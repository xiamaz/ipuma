#include <string>
#include <cmath>
#include <iostream>

#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>

#include <popops/Zero.hpp>
#include "ipu_base.h"
#include "similarity.h"
#include "encoding.h"

#include "vector.hpp"
#include "ipu_batch_affine.h"

namespace ipu {
namespace batchaffine {

/**
 * Streamable IPU graph for SW
 */
std::vector<program::Program> buildGraph(Graph& graph, unsigned long activeTiles, unsigned long maxAB, unsigned long bufSize, unsigned long maxBatches,
                                         const std::string& format, const swatlib::Matrix<int8_t> similarityData) {
  program::Sequence prog;
  program::Sequence initProg;

  auto target = graph.getTarget();
  int tileCount = target.getTilesPerIPU();

  Tensor As = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "A");
  Tensor Bs = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "B");

  Tensor Alens = graph.addVariable(INT, {activeTiles, maxBatches}, "Alen");
  Tensor Blens = graph.addVariable(INT, {activeTiles, maxBatches}, "Blen");

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

  // graph.createHostWrite(STREAM_A, As);
  // graph.createHostWrite(STREAM_B, Bs);
  // graph.createHostWrite(STREAM_A_LEN, Alens);
  // graph.createHostWrite(STREAM_B_LEN, Blens);

  // graph.createHostRead(STREAM_SCORES, Scores);
  // graph.createHostRead(STREAM_MISMATCHES, Mismatches);
  // graph.createHostRead(STREAM_A_RANGE, ARanges);
  // graph.createHostRead(STREAM_B_RANGE, BRanges);

  auto host_stream_a     = graph.addHostToDeviceFIFO(STREAM_A, UNSIGNED_CHAR,  As.numElements());
  auto host_stream_b     = graph.addHostToDeviceFIFO(STREAM_B, UNSIGNED_CHAR, Bs.numElements());
  auto host_stream_a_len = graph.addHostToDeviceFIFO(STREAM_A_LEN, INT, Alens.numElements());
  auto host_stream_b_len = graph.addHostToDeviceFIFO(STREAM_B_LEN, INT, Blens.numElements());

  auto device_stream_scores     = graph.addDeviceToHostFIFO(STREAM_SCORES, INT, Scores.numElements());
  auto device_stream_mismatches = graph.addDeviceToHostFIFO(STREAM_MISMATCHES, INT, Mismatches.numElements());
  auto device_stream_a_range    = graph.addDeviceToHostFIFO(STREAM_A_RANGE, INT, ARanges.numElements());
  auto device_stream_b_range    = graph.addDeviceToHostFIFO(STREAM_B_RANGE, INT, BRanges.numElements());

  auto frontCs = graph.addComputeSet("SmithWaterman");
  for (int i = 0; i < activeTiles; ++i) {
    int tileIndex = i % tileCount;
    VertexRef vtx = graph.addVertex(frontCs, "SWAffine",
                                    {
                                        {"bufSize", bufSize},
                                        {"maxAB", maxAB},
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
    graph.setFieldSize(vtx["C"], maxAB + 1);
    graph.setFieldSize(vtx["bG"], maxAB + 1);
    graph.setTileMapping(vtx, tileIndex);
    graph.setPerfEstimate(vtx, 1);
  }
  auto h2d_prog = program::Sequence({
    poplar::program::Copy(host_stream_a    , As.flatten()),
    poplar::program::Copy(host_stream_b    , Bs.flatten()),
    poplar::program::Copy(host_stream_a_len, Alens.flatten()),
    poplar::program::Copy(host_stream_b_len, Blens.flatten())}
  );
  auto d2h_prog =program::Sequence({
    poplar::program::Copy(Scores.flatten()    , device_stream_scores    ),
    poplar::program::Copy(Mismatches.flatten(), device_stream_mismatches),
    poplar::program::Copy(ARanges.flatten()   , device_stream_a_range   ),
    poplar::program::Copy(BRanges.flatten()   , device_stream_b_range   )}
  );
  prog.add(h2d_prog);
  prog.add(program::Execute(frontCs));
  prog.add(d2h_prog);
  return {prog, initProg};
}

SWAlgorithm::SWAlgorithm(ipu::SWConfig config, int activeTiles, int maxAB, int maxBatches, int bufsize)
    : IPUAlgorithm(config) {
  this->maxAB = maxAB;
  this->bufsize = bufsize;
  this->maxBatches = maxBatches;
  this->tilesUsed = activeTiles;

  auto totalPairs = activeTiles * maxBatches;

  a.resize(activeTiles * bufsize);
  a_len.resize(totalPairs);
  std::fill(a_len.begin(), a_len.end(), 0);

  b.resize(activeTiles * bufsize);
  b_len.resize(totalPairs);
  std::fill(b_len.begin(), b_len.end(), 0);

  scores.resize(totalPairs);
  mismatches.resize(totalPairs);
  a_range_result.resize(totalPairs);
  b_range_result.resize(totalPairs);
  bucket_pairs.resize(activeTiles);

  Graph graph = createGraph();

  auto similarityMatrix = swatlib::selectMatrix(config.similarity, config.matchValue, config.mismatchValue);
  std::vector<program::Program> programs = buildGraph(graph, activeTiles, maxAB, bufsize, maxBatches, "int", similarityMatrix);

  createEngine(graph, programs);

  engine->connectStream(STREAM_A, a.data());
  engine->connectStream(STREAM_A_LEN, a_len.data());
  engine->connectStream(STREAM_B, b.data());
  engine->connectStream(STREAM_B_LEN, b_len.data());

  engine->connectStream(STREAM_SCORES, scores.data());
  engine->connectStream(STREAM_MISMATCHES, mismatches.data());
  engine->connectStream(STREAM_A_RANGE, a_range_result.data());
  engine->connectStream(STREAM_B_RANGE, b_range_result.data());
}

BlockAlignmentResults SWAlgorithm::get_result() { return {scores, mismatches, a_range_result, b_range_result}; }

void SWAlgorithm::fillBuckets(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  std::fill(bucket_pairs.begin(), bucket_pairs.end(), 0);
  int curBucket = 0;
  int curBucketASize = 0;
  int curBucketBSize = 0;
  for (int i = 0; i < A.size(); ++i) {
    const auto& a = A[i];
    const auto& b = B[i];
    if (a.size() > maxAB || b.size() > maxAB) {
      std::cout << "sizes of sequences: a(" << a.size() << "), b(" << b.size() << ") larger than maxAB(" << maxAB << ")\n";
      exit(1);
    }
    int newBucketASize = curBucketASize + a.size();
    int newBucketBSize = curBucketBSize + b.size();
    if ((newBucketASize > bufsize || newBucketBSize > bufsize) || bucket_pairs[curBucket] >= maxBatches) {
      curBucket++;
      curBucketASize = 0;
      curBucketBSize = 0;
    }
    bucket_pairs[curBucket]++;
  }
}

void SWAlgorithm::compare(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  // if (!(checkSize(A) || checkSize(B))) throw std::runtime_error("Too small buffer or number of active tiles.");
  // size_t transSize = activeTiles * bufSize * sizeof(char);
  if (A.size() > maxBatches * tilesUsed) {
    std::cout << "A has more elements than the maxBatchsize" << std::endl;
    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "max comparisons = " << tilesUsed << " * " << maxBatches << std::endl;
    exit(1);
  }

  auto encoder = swatlib::getEncoder(swatlib::DataType::nucleicAcid);
  auto vA = encoder.encode(A);
  auto vB = encoder.encode(B);

  size_t bufferOffset = 0;
  size_t bucketIncr = 0;
  size_t index = 0;

  fillBuckets(A, B);

  std::fill(a_len.begin(), a_len.end(), 0);
  std::fill(b_len.begin(), b_len.end(), 0);
  for (const auto& cmpCountBucket : bucket_pairs) {
    size_t bucketAOffset = 0;
    size_t bucketBOffset = 0;
    for (int cmpIndex = 0; cmpIndex < cmpCountBucket; ++cmpIndex) {
      // Copy strings A[index] and B[index] to a and b
      const auto& curA = vA[index];
      const auto& curB = vB[index];
      for (int i = 0; i < curA.size(); ++i) {
        a[bufferOffset + bucketAOffset++] = curA[i];
      }
      for (int i = 0; i < curB.size(); ++i) {
        b[bufferOffset + bucketBOffset++] = curB[i];
      }
      a_len[bucketIncr + cmpIndex] = curA.size();
      b_len[bucketIncr + cmpIndex] = curB.size();
      index++;
    }

    bufferOffset += bufsize;
    bucketIncr += maxBatches;
  }

  // engine->writeTensor(STREAM_A, &a[0], &a[a.size()]);
  // engine->writeTensor(STREAM_A_LEN, &a_len[0], &a_len[a_len.size()]);
  // engine->writeTensor(STREAM_B, &b[0], &b[b.size()]);
  // engine->writeTensor(STREAM_B_LEN, &b_len[0], &b_len[b_len.size()]);

  engine->run(0);

  // engine->readTensor(STREAM_SCORES, &*scores.begin(), &*scores.end());
  // engine->readTensor(STREAM_MISMATCHES, &*mismatches.begin(), &*mismatches.end());
  // engine->readTensor(STREAM_A_RANGE, &*a_range_result.begin(), &*a_range_result.end());
  // engine->readTensor(STREAM_B_RANGE, &*b_range_result.begin(), &*b_range_result.end());
}
}  // namespace batchaffine
}  // namespace ipu