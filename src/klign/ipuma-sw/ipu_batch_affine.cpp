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

static const std::string CYCLE_COUNT_OUTER = "cycle-count-outer";
static const std::string CYCLE_COUNT_INNER = "cycle-count-inner";

long long getCellCount(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  long long cellCount = 0;
  if (A.size() != B.size()) {
    SWARN("Mismatch between size of A ", A.size(), " and size of B ", B.size());
  }
  // count cells based on 1:1 comparisons
  for (int i = 0; i < A.size(); ++i) {
    cellCount += A[i].size() * B[i].size();
  }
  return cellCount;
}

int IPUAlgoConfig::getTotalNumberOfComparisons() { return tilesUsed * maxBatches; }

int IPUAlgoConfig::getTotalBufferSize() { return tilesUsed * bufsize; }

std::string vertexTypeToString(VertexType v) { return typeString[static_cast<int>(v)]; }

/**
 * Streamable IPU graph for SW
 */
std::vector<program::Program> buildGraph(Graph& graph, VertexType vtype, unsigned long activeTiles, unsigned long maxAB,
                                         unsigned long bufSize, unsigned long maxBatches,
                                         const swatlib::Matrix<int8_t> similarityData, int gapInit, int gapExt) {
  program::Sequence prog;
  program::Sequence initProg;

  auto target = graph.getTarget();
  int tileCount = target.getTilesPerIPU();

  Tensor As = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "A");
  Tensor Bs = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "B");

  Tensor Alens = graph.addVariable(INT, {activeTiles, maxBatches}, "Alen");
  Tensor Blens = graph.addVariable(INT, {activeTiles, maxBatches}, "Blen");

  auto [m, n] = similarityData.shape();

  poplar::Type sType;
  switch (vtype) {
    case VertexType::cpp: sType = INT; break;
    case VertexType::assembly: sType = FLOAT; break;
    default: break;
  }

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

  OptionFlags streamOptions({/*{"bufferingDepth", "2"}, {"splitLimit", "0"}*/});
  auto host_stream_a = graph.addHostToDeviceFIFO(STREAM_A, UNSIGNED_CHAR, As.numElements(), ReplicatedStreamMode::REPLICATE, streamOptions);
  auto host_stream_b = graph.addHostToDeviceFIFO(STREAM_B, UNSIGNED_CHAR, Bs.numElements(), ReplicatedStreamMode::REPLICATE, streamOptions);
  auto host_stream_a_len = graph.addHostToDeviceFIFO(STREAM_A_LEN, INT, Alens.numElements(), ReplicatedStreamMode::REPLICATE, streamOptions);
  auto host_stream_b_len = graph.addHostToDeviceFIFO(STREAM_B_LEN, INT, Blens.numElements(), ReplicatedStreamMode::REPLICATE, streamOptions);

  auto device_stream_scores = graph.addDeviceToHostFIFO(STREAM_SCORES, INT, Scores.numElements());
  auto device_stream_mismatches = graph.addDeviceToHostFIFO(STREAM_MISMATCHES, INT, Mismatches.numElements());
  auto device_stream_a_range = graph.addDeviceToHostFIFO(STREAM_A_RANGE, INT, ARanges.numElements());
  auto device_stream_b_range = graph.addDeviceToHostFIFO(STREAM_B_RANGE, INT, BRanges.numElements());

  auto frontCs = graph.addComputeSet("SmithWaterman");
  for (int i = 0; i < activeTiles; ++i) {
    int tileIndex = i % tileCount;
    VertexRef vtx = graph.addVertex(frontCs, vertexTypeToString(vtype),
                                    {
                                        {"bufSize", bufSize},
                                        {"maxAB", maxAB},
                                        {"gapInit", gapInit},
                                        {"gapExt", gapExt},
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
  auto h2d_prog = program::Sequence(
      {poplar::program::Copy(host_stream_a, As.flatten()), poplar::program::Copy(host_stream_b, Bs.flatten()),
       poplar::program::Copy(host_stream_a_len, Alens.flatten()), poplar::program::Copy(host_stream_b_len, Blens.flatten())});
  auto d2h_prog = program::Sequence({poplar::program::Copy(Scores.flatten(), device_stream_scores),
                                     poplar::program::Copy(Mismatches.flatten(), device_stream_mismatches),
                                     poplar::program::Copy(ARanges.flatten(), device_stream_a_range),
                                     poplar::program::Copy(BRanges.flatten(), device_stream_b_range)});
#ifdef IPUMA_DEBUG
 program::Sequence  main_prog;
  main_prog.add(program::Execute(frontCs));
  addCycleCount(graph, main_prog, CYCLE_COUNT_INNER);
#else
  auto main_prog = program::Execute(frontCs);
#endif
  prog.add(h2d_prog);
  prog.add(main_prog);
  prog.add(d2h_prog);

#ifdef IPUMA_DEBUG
  addCycleCount(graph, prog, CYCLE_COUNT_OUTER);
#endif
  return {prog, initProg};
}

SWAlgorithm::SWAlgorithm(ipu::SWConfig config, IPUAlgoConfig algoconfig)
    : IPUAlgorithm(config)
    , algoconfig(algoconfig) {
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
  std::vector<program::Program> programs =
      buildGraph(graph, algoconfig.vtype, algoconfig.tilesUsed, algoconfig.maxAB, algoconfig.bufsize, algoconfig.maxBatches,
                 similarityMatrix, config.gapInit, config.gapExtend);

  createEngine(graph, programs);

  // engine->connectStream(STREAM_A, a.data());
  // engine->connectStream(STREAM_A_LEN, a_len.data());
  // engine->connectStream(STREAM_B, b.data());
  // engine->connectStream(STREAM_B_LEN, b_len.data());

  // engine->connectStream(STREAM_SCORES, scores.data());
  // engine->connectStream(STREAM_MISMATCHES, mismatches.data());
  // engine->connectStream(STREAM_A_RANGE, a_range_result.data());
  // engine->connectStream(STREAM_B_RANGE, b_range_result.data());
}

BlockAlignmentResults SWAlgorithm::get_result() { return {scores, mismatches, a_range_result, b_range_result}; }

vector<size_t> SWAlgorithm::fillBuckets(IPUAlgoConfig& algoconfig,const std::vector<std::string>& A, const std::vector<std::string>& B) {
  // std::fill(bucket_pairs.begin(), bucket_pairs.end(), 0);
  vector<size_t> bucket_pairs(algoconfig.tilesUsed, 0);
  int curBucket = 0;
  int curBucketASize = 0;
  int curBucketBSize = 0;
  for (int i = 0; i < A.size(); ++i) {
    const auto& a = A[i];
    const auto& b = B[i];
    if (a.size() > algoconfig.maxAB || b.size() > algoconfig.maxAB) {
      std::cout << "sizes of sequences: a(" << a.size() << "), b(" << b.size() << ") larger than maxAB(" << algoconfig.maxAB
                << ")\n";
      exit(1);
    }
    int newBucketASize = curBucketASize + a.size();
    int newBucketBSize = curBucketBSize + b.size();
    if ((newBucketASize > algoconfig.bufsize || newBucketBSize > algoconfig.bufsize) ||
        bucket_pairs[curBucket] >= algoconfig.maxBatches) {
      curBucket++;
      curBucketASize = 0;
      curBucketBSize = 0;
      if (curBucket >= algoconfig.tilesUsed) {
        std::cout << "More buckets needed than available (" << algoconfig.tilesUsed << ")\n";
        exit(1);
      }
    }
    bucket_pairs[curBucket]++;
  }
  return bucket_pairs;
}

void SWAlgorithm::prepared_remote_compare(char* a, int32_t* a_len, char* b, int32_t* b_len, int32_t* scores, int32_t* mismatches,
                                          int32_t* a_range_result, int32_t* b_range_result) {
  // We have to reconnect the streams to new memory locations as the destination will be in a shared memroy region.
  engine->connectStream(STREAM_A, a);
  engine->connectStream(STREAM_A_LEN, a_len);
  engine->connectStream(STREAM_B, b);
  engine->connectStream(STREAM_B_LEN, b_len);

  engine->connectStream(STREAM_SCORES, scores);
  engine->connectStream(STREAM_MISMATCHES, mismatches);
  engine->connectStream(STREAM_A_RANGE, a_range_result);
  engine->connectStream(STREAM_B_RANGE, b_range_result);

  upcxx_utils::AsyncTimer engineTimer("Engine");
  engineTimer.start();
  engine->run(0);
  engineTimer.stop();
#ifdef IPUMA_DEBUG
  auto cyclesOuter = getTotalCycles(*engine, CYCLE_COUNT_OUTER);
  auto cyclesInner = getTotalCycles(*engine, CYCLE_COUNT_INNER);
  auto timeOuter = static_cast<double>(cyclesOuter) / getTarget().getTileClockFrequency();
  auto timeInner = static_cast<double>(cyclesInner) / getTarget().getTileClockFrequency();
  SLOG("Poplar cycle count: ", cyclesInner, "/", cyclesOuter, " computed time (in s): ", timeInner, "/", timeOuter, "\n");

  // GCUPS computation
  // auto cellCount = getCellCount(A, B);
  uint64_t cellCount = 0;
  uint64_t dataCount = 0;
  for (size_t i = 0; i < algoconfig.getTotalNumberOfComparisons(); i++) {
    cellCount += a_len[i] * b_len[i];
    dataCount += a_len[i] + b_len[i];
  }
  
  double GCUPSOuter = static_cast<double>(cellCount) / timeOuter / 1e9;
  double GCUPSInner = static_cast<double>(cellCount) / timeInner / 1e9;
  SLOG("Poplar estimated cells(", cellCount, ") GCUPS ", GCUPSInner, "/", GCUPSOuter, "\n");

  auto transferTime = timeOuter - timeInner;
  auto transferBandwidth = static_cast<double>(dataCount) / transferTime / 1e6;
  SLOG("Transfer time: ", transferTime, "s estimated bandwidth: ", transferBandwidth, "mb/s, per vertex: ", transferBandwidth / algoconfig.tilesUsed, "mb/s\n");
#endif
}

void SWAlgorithm::compare_local(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  prepare_remote(algoconfig, A, B, a.data(), a_len.data(), b.data(), b_len.data());
  prepared_remote_compare(a.data(), a_len.data(), b.data(), b_len.data(), scores.data(), mismatches.data(), a_range_result.data(),
                          b_range_result.data());
}

void SWAlgorithm::prepare_remote(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B, char* a, int32_t* a_len,
                                 char* b, int32_t* b_len) {
  upcxx_utils::AsyncTimer preprocessTimer("Preprocess");
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

  auto bucket_pairs = fillBuckets(algoconfig, A, B);

  memset(a_len, 0, algoconfig.getTotalNumberOfComparisons());
  memset(b_len, 0, algoconfig.getTotalNumberOfComparisons());
  #ifdef IPUMA_DEBUG
  for (size_t i = 0; i < algoconfig.getTotalNumberOfComparisons(); i++) {
    if (a_len[i] != 0 || b_len[i] != 0) {
      cout << "A/B_len is non-zero" << std::endl;
      exit(1);
    }
  }
  #endif
  // std::fill(a_len.begin(), a_len.end(), 0);
  // std::fill(b_len.begin(), b_len.end(), 0);
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
  // SLOG("Inner comparison time: ", preprocessTimer.get_elapsed(), " engine run: ", engineTimer.get_elapsed(), "\n");
}
}  // namespace batchaffine
}  // namespace ipu