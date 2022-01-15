#ifndef IPU_BATCH_AFFINE_HPP
#define IPU_BATCH_AFFINE_HPP

#ifndef KLIGN_GPU_BLOCK_SIZE
#define KLIGN_GPU_BLOCK_SIZE 20000
#endif

// Smith Waterman with static graph size.
#include <string>
#include <cmath>
#include <iostream>

#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>

#include <popops/Zero.hpp>

#include "ipulib.hpp"
#include "ipu_base.hpp"
#include "similarity.h"
#include "encoding.h"

using namespace poplar;

namespace ipu {
namespace batchaffine {

const std::string STREAM_A = "a-write";
const std::string STREAM_A_LEN = "a-len-write";
const std::string STREAM_B = "b-write";
const std::string STREAM_B_LEN = "b-len-write";
const std::string STREAM_SCORES = "scores-read";
const std::string STREAM_MISMATCHES = "mismatches-read";
const std::string STREAM_A_RANGE = "a-range-read";
const std::string STREAM_B_RANGE = "b-range-read";

/**
 * Streamable IPU graph for SW
 */
std::vector<program::Program> buildGraph(Graph& graph, unsigned long activeTiles, unsigned long bufSize, const std::string& format,
                                         const swatlib::Matrix<int8_t> similarityData) {
  program::Sequence prog;
  program::Sequence initProg;

  auto target = graph.getTarget();
  int tileCount = target.getTilesPerIPU();

  Tensor As = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "A");
  Tensor Bs = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "B");

  Tensor Alens = graph.addVariable(UNSIGNED_INT, {activeTiles}, "Alen");
  Tensor Blens = graph.addVariable(UNSIGNED_INT, {activeTiles}, "Blen");

  auto [m, n] = similarityData.shape();

  poplar::Type sType = formatToType(format);
  TypeTraits traits = typeToTrait(sType);
  void* similarityBuffer;
  convertSimilarityMatrix(target, sType, similarityData, &similarityBuffer);
  Tensor similarity = graph.addConstant(sType, {m, n}, similarityBuffer, traits, false, "similarity");
  free(similarityBuffer);

  // Tensor similarity = graph.addConstant(SIGNED_CHAR, {m, n}, similarityData.data(), "similarity");
  Tensor Scores = graph.addVariable(INT, {activeTiles}, "Scores");
  Tensor ARanges = graph.addVariable(INT, {activeTiles}, "ARanges");
  Tensor BRanges = graph.addVariable(INT, {activeTiles}, "BRanges");
  Tensor Mismatches = graph.addVariable(INT, {activeTiles}, "Mismatches");

  graph.setTileMapping(similarity, 0);
  for (int i = 0; i < activeTiles; ++i) {
    int tileIndex = i % tileCount;
    graph.setTileMapping(As[i], tileIndex);
    graph.setTileMapping(Bs[i], tileIndex);
    graph.setTileMapping(Alens[i], tileIndex);
    graph.setTileMapping(Blens[i], tileIndex);

    graph.setTileMapping(Scores[i], tileIndex);
    graph.setTileMapping(ARanges[i], tileIndex);
    graph.setTileMapping(BRanges[i], tileIndex);
    graph.setTileMapping(Mismatches[i], tileIndex);
  }

  graph.createHostWrite(STREAM_A, As);
  graph.createHostWrite(STREAM_B, Bs);
  graph.createHostWrite(STREAM_A_LEN, Alens);
  graph.createHostWrite(STREAM_B_LEN, Blens);
  graph.createHostRead(STREAM_SCORES, Scores);
  graph.createHostRead(STREAM_MISMATCHES, Mismatches);
  graph.createHostRead(STREAM_A_RANGE, ARanges);
  graph.createHostRead(STREAM_B_RANGE, BRanges);

  auto frontCs = graph.addComputeSet("front");
  for (int i = 0; i < activeTiles; ++i) {
    int tileIndex = i % tileCount;
    VertexRef vtx = graph.addVertex(frontCs, "SWAffine",
                                    {
                                        {"bufSize", bufSize},
                                        {"gapInit", 0},
                                        {"gapExt", -1},
                                        {"A", As[i]},
                                        {"Alen", Alens[i]},
                                        {"Blen", Blens[i]},
                                        {"B", Bs[i]},
                                        {"simMatrix", similarity},
                                        {"score", Scores[i]},
                                        {"mismatches", Mismatches[i]},
                                        {"ARange", ARanges[i]},
                                        {"BRange", BRanges[i]},
                                    });
    graph.setFieldSize(vtx["C"], bufSize + 1);
    graph.setFieldSize(vtx["bG"], bufSize + 1);
    graph.setTileMapping(vtx, tileIndex);
    graph.setPerfEstimate(vtx, 1);
  }
  prog.add(program::Execute(frontCs));
  return {prog, initProg};
}

class SWAlgorithm : public IPUAlgorithm {
 private:
  std::vector<char> a;
  std::vector<int16_t> a_len;
  std::vector<char> b;
  std::vector<int16_t> b_len;

  std::vector<int32_t> scores;
  std::vector<int32_t> mismatches;
  std::vector<int32_t> a_range_result;
  std::vector<int32_t> b_range_result;

 public:
  SWAlgorithm(SWConfig config, int maxAB = 300, int activeTiles = 1472)
      : IPUAlgorithm(config, activeTiles, maxAB) {
    a.reserve(maxAB * activeTiles);
    a_len.reserve(activeTiles);
    b.reserve(maxAB * activeTiles);
    b_len.reserve(activeTiles);
    scores.reserve(activeTiles);
    mismatches.reserve(activeTiles);

    Graph graph = createGraph();

    auto similarityMatrix = swatlib::selectMatrix(config.similarity, config.matchValue, config.mismatchValue);
    std::vector<program::Program> programs = buildGraph(graph, activeTiles, maxAB, "int", similarityMatrix);

    createEngine(graph, programs);
  }

  void compare(const std::vector<std::string>& A, const std::vector<std::string>& B) {
    // if (!(checkSize(A) || checkSize(B))) throw std::runtime_error("Too small buffer or number of active tiles.");
    // size_t transSize = activeTiles * bufSize * sizeof(char);

    assert(A.size() == B.size());
    assert(A.size() == activeTiles);
    auto encoder = swatlib::getEncoder(swatlib::DataType::nucleicAcid);
    auto vA = encoder.encode(A);
    auto vB = encoder.encode(B);

    for (size_t i = 0; i < A.size(); i++) {
      a_len[i] = A[i].size();
      b_len[i] = B[i].size();
    }

    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < A[i].size(); j++) {
        a[i * bufSize + j] = vA[i][j];
      }
    }
    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < B[i].size(); j++) {
        b[i * bufSize + j] = vB[i][j];
      }
    }

    engine->writeTensor(STREAM_A, &a[0], &a[a.size()]);
    engine->writeTensor(STREAM_A_LEN, &a_len[0], &a_len[a_len.size()]);
    engine->writeTensor(STREAM_B, &b[0], &b[b.size()]);
    engine->writeTensor(STREAM_B_LEN, &b_len[0], &b_len[b_len.size()]);
    cout << "Wrote" << std::endl;

    engine->run(0);
    cout << "Ran" << std::endl;

    engine->readTensor(STREAM_SCORES, &*scores.begin(), &*scores.end());
    engine->readTensor(STREAM_MISMATCHES, &*mismatches.begin(), &*mismatches.end());
    engine->readTensor(STREAM_A_RANGE, &*a_range_result.begin(), &*a_range_result.end());
    engine->readTensor(STREAM_B_RANGE, &*b_range_result.begin(), &*b_range_result.end());
    cout << "Read" << std::endl;
  }
};
}  // namespace batchaffine
}  // namespace ipu
#endif  // IPU_BATCH_AFFINE_HPP