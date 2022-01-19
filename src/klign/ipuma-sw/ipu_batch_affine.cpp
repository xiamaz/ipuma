#include <string>
#include <cmath>
#include <iostream>

#include "upcxx_utils/timers.hpp"

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

int IPUAlgoConfig::getTotalNumberOfComparisons() {
  return tilesUsed * maxBatches;
}

int IPUAlgoConfig::getTotalBufferSize() {
  return tilesUsed * bufsize;
}

std::string IPUAlgoConfig::getVertexTypeString() {
  return typeString[static_cast<int>(vtype)];
}

/**
 * Streamable IPU graph for SW
 */
std::vector<program::Program> buildGraph(Graph& graph, std::string vtype, unsigned long activeTiles, unsigned long maxAB, unsigned long bufSize, unsigned long maxBatches,
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
    VertexRef vtx = graph.addVertex(frontCs, vtype,
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

SWAlgorithm::SWAlgorithm(ipu::SWConfig config, IPUAlgoConfig algoconfig)
    : IPUAlgorithm(config), algoconfig(algoconfig) {
  // this->maxAB = maxAB;
  // this->bufsize = bufsize;
  // this->maxBatches = maxBatches;
  // this->tilesUsed = activeTiles;

  const auto totalComparisonsCount = algoconfig.getTotalNumberOfComparisons();
  const auto inputBufferSize = algoconfig.getTotalBufferSize();

  a.resize(inputBufferSize);
  a_len.resize(totalComparisonsCount);
  std::fill(a_len.begin(), a_len.end(), 0);

  b.resize(inputBufferSize);
  b_len.resize(totalComparisonsCount);
  std::fill(b_len.begin(), b_len.end(), 0);

  scores.resize(totalComparisonsCount);
  mismatches.resize(totalComparisonsCount);
  a_range_result.resize(totalComparisonsCount);
  b_range_result.resize(totalComparisonsCount);
  bucket_pairs.resize(algoconfig.tilesUsed);

  Graph graph = createGraph();

  auto similarityMatrix = swatlib::selectMatrix(config.similarity, config.matchValue, config.mismatchValue);
  std::vector<program::Program> programs = buildGraph(graph, algoconfig.getVertexTypeString(), algoconfig.tilesUsed, algoconfig.maxAB, algoconfig.bufsize, algoconfig.maxBatches, "int", similarityMatrix);

#ifdef POPLAR_DEBUG
  addCycleCount(graph, static_cast<program::Sequence&>(programs[0]));
#endif

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
    if (a.size() > algoconfig.maxAB || b.size() > algoconfig.maxAB) {
      std::cout << "sizes of sequences: a(" << a.size() << "), b(" << b.size() << ") larger than maxAB(" << algoconfig.maxAB << ")\n";
      exit(1);
    }
    int newBucketASize = curBucketASize + a.size();
    int newBucketBSize = curBucketBSize + b.size();
    if ((newBucketASize > algoconfig.bufsize || newBucketBSize > algoconfig.bufsize) || bucket_pairs[curBucket] >= algoconfig.maxBatches) {
      curBucket++;
      curBucketASize = 0;
      curBucketBSize = 0;
    }
    bucket_pairs[curBucket]++;
  }
}

void SWAlgorithm::compare(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  upcxx_utils::AsyncTimer preprocessTimer("Preprocess");
  upcxx_utils::AsyncTimer engineTimer("Engine");
  preprocessTimer.start();
  if (A.size() > algoconfig.maxBatches * algoconfig.tilesUsed) {
    std::cout << "A has more elements than the maxBatchsize" << std::endl;
    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "max comparisons = " << algoconfig.tilesUsed << " * " << algoconfig.maxBatches << std::endl;
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

    bufferOffset += algoconfig.bufsize;
    bucketIncr += algoconfig.maxBatches;
  }
  preprocessTimer.stop();

  engineTimer.start();
  engine->run(0);
  engineTimer.stop();
  SLOG("Inner comparison time: ", preprocessTimer.get_elapsed(), " engine run: ", engineTimer.get_elapsed(), "\n");

#ifdef POPLAR_DEBUG
  uint32_t cycles[2];
  engine->readTensor("cycles", &cycles, &cycles + 1);
  uint64_t totalCycles = (((uint64_t)cycles[1]) << 32) | cycles[0];
  float computedTime = (double) totalCycles / getTarget().getTileClockFrequency();
  SLOG("Poplar cycle count: ", totalCycles, " computed time (in s): ", computedTime, "\n");
#endif
}
}  // namespace batchaffine
}  // namespace ipu