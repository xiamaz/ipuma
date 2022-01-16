#include <string>
#include <cmath>
#include <iostream>

#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>

#include <popops/Zero.hpp>
#include "ipu_base.h"
#include "similarity.h"
#include "encoding.h"

#include "ipu_batch_affine.h"

namespace ipu {
namespace batchaffine {

/**
 * Streamable IPU graph for SW
 */
std::vector<program::Program> buildGraph(Graph& graph, unsigned long activeTiles, unsigned long bufSize, unsigned long maxBatches,
                                         const std::string& format, const swatlib::Matrix<int8_t> similarityData) {
  program::Sequence prog;
  program::Sequence initProg;

  auto target = graph.getTarget();
  int tileCount = target.getTilesPerIPU();

  Tensor As = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "A");
  Tensor Bs = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "B");

  Tensor Alens = graph.addVariable(UNSIGNED_INT, {activeTiles, maxBatches}, "Alen");
  Tensor Blens = graph.addVariable(UNSIGNED_INT, {activeTiles, maxBatches}, "Blen");

  auto [m, n] = similarityData.shape();

  poplar::Type sType = formatToType(format);
  TypeTraits traits = typeToTrait(sType);
  void* similarityBuffer;
  convertSimilarityMatrix(target, sType, similarityData, &similarityBuffer);
  Tensor similarity = graph.addConstant(sType, {m, n}, similarityBuffer, traits, false, "similarity");
  free(similarityBuffer);

  // Tensor similarity = graph.addConstant(SIGNED_CHAR, {m, n}, similarityData.data(), "similarity");
  Tensor Scores = graph.addVariable(INT, {activeTiles, maxBatches}, "Scores");
  Tensor ARanges = graph.addVariable(INT, {activeTiles, maxBatches}, "ARanges");
  Tensor BRanges = graph.addVariable(INT, {activeTiles, maxBatches}, "BRanges");
  Tensor Mismatches = graph.addVariable(INT, {activeTiles, maxBatches}, "Mismatches");

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
                                        {"maxNPerTile", maxBatches},
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

SWAlgorithm::SWAlgorithm(ipu::SWConfig config, int activeTiles, int maxAB, int maxBatches, int bufsize)
    : IPUAlgorithm(config) {
  auto totalPairs = activeTiles * maxBatches;
  a.resize(bufsize);
  a_len.resize(totalPairs);
  b.resize(bufsize);
  b_len.resize(totalPairs);
  scores.resize(totalPairs);
  mismatches.resize(totalPairs);
  a_range_result.resize(totalPairs);
  b_range_result.resize(totalPairs);
  bucket_pairs.resize(activeTiles);
  this->maxAB = maxAB;
  this->bufsize = bufsize;
  this->maxBatches = maxBatches;

  Graph graph = createGraph();

  auto similarityMatrix = swatlib::selectMatrix(config.similarity, config.matchValue, config.mismatchValue);
  std::vector<program::Program> programs = buildGraph(graph, activeTiles, maxAB, maxBatches, "int", similarityMatrix);

  createEngine(graph, programs);
}

BlockAlignmentResults SWAlgorithm::get_result() { return {scores, mismatches, a_range_result, b_range_result}; }

void SWAlgorithm::compare(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  // if (!(checkSize(A) || checkSize(B))) throw std::runtime_error("Too small buffer or number of active tiles.");
  // size_t transSize = activeTiles * bufSize * sizeof(char);

  auto encoder = swatlib::getEncoder(swatlib::DataType::nucleicAcid);
  auto vA = encoder.encode(A);
  auto vB = encoder.encode(B);

  {
    // TODO(lbb): Duplicated from klign.cpp, pass as parameter.
    size_t offset = 0;
    size_t current_bucket_a_len = 0;
    size_t current_bucket_b_len = 0;
    size_t current_bucket_elems = 0;
    size_t current_bucket_id = 0;
    for (size_t i = 0; i < A.size(); i++) {
      // Check in which bucket we are.
      bool bucket_oom = current_bucket_a_len + A[i].size()  >= bufsize || current_bucket_b_len + B[i].size() >= bufsize;
      bool bucket_ooe = current_bucket_elems >= maxBatches;
      if (bucket_oom || bucket_ooe) {
        bucket_pairs[current_bucket_id] = current_bucket_elems;
        current_bucket_id++;
        offset = bufsize * current_bucket_id;
        current_bucket_a_len = 0;
        current_bucket_b_len = 0;
        current_bucket_elems = 0;
      }
      current_bucket_elems++;
      current_bucket_a_len += A[i].size();
      current_bucket_b_len += B[i].size();

      // Assign to bucketet list
      a_len[offset] = A[i].size();
      b_len[offset] = B[i].size();
      offset++;
      if (a_len[offset] > maxAB) {
        std::cout << "A is longer then maxAB" << std::endl;
        exit(1);
      }
      if (b_len[offset] > maxAB) {
        std::cout << "B is longer then maxAB" << std::endl;
        exit(1);
      }
    }
  }

  {
    size_t offset = 0;
    size_t current_bucket_elem = 0;
    size_t current_bucket_id = 0;
    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < A[i].size(); j++) {
        a[offset] = vA[i][j];
        offset++;
      }
      current_bucket_elem++;
      if (current_bucket_elem >= bucket_pairs[current_bucket_id]) {
        offset = bufsize * current_bucket_id;
        current_bucket_elem = 0;
        current_bucket_id++;
      }
    }
  }
  {
    size_t offset = 0;
    size_t current_bucket_elem = 0;
    size_t current_bucket_id = 0;
    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < B[i].size(); j++) {
        b[offset] = vB[i][j];
        offset++;
      }
      current_bucket_elem++;
      if (current_bucket_elem >= bucket_pairs[current_bucket_id]) {
        offset = bufsize * current_bucket_id;
        current_bucket_elem = 0;
        current_bucket_id++;
      }
    }
  }

  engine->writeTensor(STREAM_A, &a[0], &a[a.size()]);
  engine->writeTensor(STREAM_A_LEN, &a_len[0], &a_len[a_len.size()]);
  engine->writeTensor(STREAM_B, &b[0], &b[b.size()]);
  engine->writeTensor(STREAM_B_LEN, &b_len[0], &b_len[b_len.size()]);

  engine->run(0);

  engine->readTensor(STREAM_SCORES, &*scores.begin(), &*scores.end());
  engine->readTensor(STREAM_MISMATCHES, &*mismatches.begin(), &*mismatches.end());
  engine->readTensor(STREAM_A_RANGE, &*a_range_result.begin(), &*a_range_result.end());
  engine->readTensor(STREAM_B_RANGE, &*b_range_result.begin(), &*b_range_result.end());
}

}  // namespace batchaffine
}  // namespace ipu