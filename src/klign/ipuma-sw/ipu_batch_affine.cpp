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

namespace partition {

  std::vector<std::tuple<int, int>> fillFirst(const std::vector<std::string>& A, const std::vector<std::string>& B, int bucketCount, int bucketCapacity, int bucketCountCapacity) {
    std::vector<std::tuple<int, int>> mapping(A.size(), {0, 0});
    std::vector<std::tuple<int, int, int>> buckets(bucketCount, {0, 0, 0});
    int bucketIndex = 0;
    for (int i = 0; i < A.size(); ++i) {
      const auto& a = A[i];
      const auto& b = B[i];

      // find next empty bucket
      while (bucketIndex < bucketCount) {
        auto& [bN, bA, bB] = buckets[bucketIndex];
        if (bN + 1 <= bucketCountCapacity && bA + a.size() <= bucketCapacity && bB + b.size() <= bucketCapacity) {
          mapping[i] = {bucketIndex, i};
          bN++;
          bA += a.size();
          bB += b.size();
          // std::cout << "i: " << i << " bucket index: " << bucketIndex << " cap: " << bA << "/" << bucketCapacity << " n: " << bN << "/" << bucketCountCapacity << "\n";
          break;
        } else {
          bucketIndex++;
        }
      }
      if (bucketIndex >= bucketCount) {
          std::cout << "More buckets needed than available (" << buckets.size() << ")\n";
          exit(1);
      }
    }

    return mapping;
  }

  std::vector<std::tuple<int, int>> roundRobin(const std::vector<std::string>& A, const std::vector<std::string>& B, int bucketCount, int bucketCapacity, int bucketCountCapacity) {
    std::vector<std::tuple<int, int>> mapping(A.size(), {0, 0});
    std::vector<std::tuple<int, int, int>> buckets(bucketCount, {0, 0, 0});
    int bucketIndex = 0;
    for (int i = 0; i < A.size(); ++i) {
      const auto& a = A[i];
      const auto& b = B[i];

      // find next empty bucket
      int boff = 0;
      for (; boff < bucketCount; ++boff) {
        int bi = (bucketIndex + boff) % bucketCount;
        auto& [bN, bA, bB] = buckets[bi];
        if (bN + 1 > bucketCountCapacity || bA + a.size() > bucketCapacity || bB + b.size() > bucketCapacity) {
          continue;
        } else {
          mapping[i] = {bucketIndex, i};
          bN++;
          bA += a.size();
          bB += b.size();
          break;
        }
      }
      if (boff >= bucketCount) {
          std::cout << "More buckets needed than available (" << buckets.size() << ")\n";
          exit(1);
      }
      bucketIndex = (bucketIndex + 1) % bucketCount;
    }
    return mapping;
  }

  // Greedy approach in which we always put current sequence into one with lowest weight
  std::vector<std::tuple<int, int>> greedy(const std::vector<std::string>& A, const std::vector<std::string>& B, int bucketCount, int bucketCapacity, int bucketCountCapacity) {
    std::vector<std::tuple<int, int>> mapping(A.size(), {0, 0});
    std::vector<std::tuple<int, int, int, int>> buckets(bucketCount, {0, 0, 0, 0});
    for (int i = 0; i < A.size(); ++i) {
      const auto& a = A[i];
      const auto& b = B[i];

      auto weight = a.size() * b.size();
      int smallestBucket = -1;
      int smallestBucketWeight = 0;
      for (int bi = 0; bi < bucketCount; ++bi) {
        auto [bN, bA, bB, bW] = buckets[bi];
        if (!(bN + 1 > bucketCountCapacity || bA + a.size() > bucketCapacity || bB + b.size() > bucketCapacity)) {
          if (smallestBucket == -1 || smallestBucketWeight > bW) {
            smallestBucket = bi;
            smallestBucketWeight = bW;
          }
        }
      }

      if (smallestBucket == -1) {
        std::cout << "Out of buckets\n";
        exit(1);
      }

      auto& [bN, bA, bB, bW] = buckets[smallestBucket];
      bN++;
      bA += a.size();
      bB += b.size();
      bW += weight;
      mapping[i] = {smallestBucket, i};
    }
    return mapping;
  }
}

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

  Tensor Alens = graph.addVariable(INT, {activeTiles, maxBatches * 2}, "Alen");
  Tensor Blens = graph.addVariable(INT, {activeTiles, maxBatches * 2}, "Blen");

  auto [m, n] = similarityData.shape();

  poplar::Type sType;
  int workerMultiplier = 1;
  switch (vtype) {
    case VertexType::cpp: sType = INT; break;
    case VertexType::assembly: sType = FLOAT; break;
    case VertexType::multi: sType = INT; workerMultiplier = target.getNumWorkerContexts(); break;
    case VertexType::multiasm: sType = FLOAT; workerMultiplier = target.getNumWorkerContexts(); break;
    case VertexType::stripedasm: sType = HALF; break;
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
  // OptionFlags streamOptions({{"bufferingDepth", "2"}, {"splitLimit", "0"}});
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
    graph.setFieldSize(vtx["C"], maxAB * workerMultiplier);
    graph.setFieldSize(vtx["bG"], maxAB * workerMultiplier);
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
  a_len.resize(totalComparisonsCount * 2);
  std::fill(a_len.begin(), a_len.end(), 0);

  b.resize(inputBufferSize);
  b_len.resize(totalComparisonsCount * 2);
  std::fill(b_len.begin(), b_len.end(), 0);

  scores.resize(totalComparisonsCount);
  mismatches.resize(totalComparisonsCount);
  a_range_result.resize(totalComparisonsCount);
  b_range_result.resize(totalComparisonsCount);

  Graph graph = createGraph();

  auto similarityMatrix = swatlib::selectMatrix(config.similarity, config.matchValue, config.mismatchValue, config.ambiguityValue);
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

void SWAlgorithm::checkSequenceSizes(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B) {
  if (A.size() != B.size()) {
    std::cout << "Mismatched size A " << A.size() << " != B " << B.size() << "\n";
    exit(1);
  }
  if (A.size() > algoconfig.maxBatches * algoconfig.tilesUsed) {
    std::cout << "A has more elements than the maxBatchsize" << std::endl;
    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "max comparisons = " << algoconfig.tilesUsed << " * " << algoconfig.maxBatches << std::endl;
    exit(1);
  }

  for (const auto& a : A) {
    if (a.size() > algoconfig.maxAB) {
      std::cout << "Sequence size in a " << a.size() << " > " << algoconfig.maxAB << "\n";
      exit(1);
    }
  }
  for (const auto& b : B) {
    if (b.size() > algoconfig.maxAB) {
      std::cout << "Sequence size in a " << b.size() << " > " << algoconfig.maxAB << "\n";
      exit(1);
    }
  }
}

vector<std::tuple<int, int>> SWAlgorithm::fillBuckets(IPUAlgoConfig& algoconfig,const std::vector<std::string>& A, const std::vector<std::string>& B) {
  vector<std::tuple<int, int>> bucket_pairs;
  switch (algoconfig.fillAlgo) {
  case partition::Algorithm::fillFirst:
    bucket_pairs = partition::fillFirst(A, B, algoconfig.tilesUsed, algoconfig.bufsize, algoconfig.maxBatches);
    break;
  case partition::Algorithm::roundRobin:
    bucket_pairs = partition::roundRobin(A, B, algoconfig.tilesUsed, algoconfig.bufsize, algoconfig.maxBatches);
    break;
  case partition::Algorithm::greedy:
    bucket_pairs = partition::greedy(A, B, algoconfig.tilesUsed, algoconfig.bufsize, algoconfig.maxBatches);
    break;
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
    cellCount += a_len[2*i] * b_len[2*i];
    dataCount += a_len[2*i] + b_len[2*i];
  }
  
  double GCUPSOuter = static_cast<double>(cellCount) / timeOuter / 1e9;
  double GCUPSInner = static_cast<double>(cellCount) / timeInner / 1e9;
  SLOG("Poplar estimated cells(", cellCount, ") GCUPS ", GCUPSInner, "/", GCUPSOuter, "\n");

  // dataCount - actual data content transferred
  // totalTransferSize - size of buffer being transferred
  double totalTransferSize = algoconfig.getTotalBufferSize() * 2;

  auto transferTime = timeOuter - timeInner;
  auto transferInfoRatio = static_cast<double>(dataCount) / totalTransferSize * 100;
  auto transferBandwidth = totalTransferSize / transferTime / 1e6;
  SLOG("Transfer time: ", transferTime, "s transfer ratio: ", transferInfoRatio, "% estimated bandwidth: ", transferBandwidth, "mb/s, per vertex: ", transferBandwidth / algoconfig.tilesUsed, "mb/s\n");
#endif
}

void SWAlgorithm::compare_local(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  std::vector<int> mapping;
  prepare_remote(algoconfig, A, B, a.data(), a_len.data(), b.data(), b_len.data(), mapping);
  std::vector<int32_t> unord_scores(scores), unord_mismatches(mismatches), unord_a_range(a_range_result), unord_b_range(b_range_result);
  prepared_remote_compare(a.data(), a_len.data(), b.data(), b_len.data(), unord_scores.data(), unord_mismatches.data(), unord_a_range.data(),
                          unord_b_range.data());
  // std::cout << swatlib::printVector(unord_scores) << "\n";
  // reorder results based on mapping
  for (int i = 0; i < mapping.size(); ++i) {
    scores[i] = unord_scores[mapping[i]];
    mismatches[i] = unord_mismatches[mapping[i]];
    a_range_result[i] = unord_a_range[mapping[i]];
    b_range_result[i] = unord_b_range[mapping[i]];
  }
}

void SWAlgorithm::prepare_remote(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B, char* a, int32_t* a_len,
                                 char* b, int32_t* b_len, std::vector<int>& seqMapping) {
  upcxx_utils::AsyncTimer preprocessTimer("Preprocess");
  preprocessTimer.start();
  checkSequenceSizes(algoconfig, A, B);

  auto encoder = swatlib::getEncoder(swatlib::DataType::nucleicAcid);
  auto vA = encoder.encode(A);
  auto vB = encoder.encode(B);

  memset(a_len, 0, algoconfig.getTotalNumberOfComparisons() * sizeof(*a_len) * 2);
  memset(b_len, 0, algoconfig.getTotalNumberOfComparisons() * sizeof(*b_len) * 2);

  #ifdef IPUMA_DEBUG
  for (size_t i = 0; i < algoconfig.getTotalNumberOfComparisons(); i++) {
    if (a_len[i] != 0 || b_len[i] != 0) {
      cout << "A/B_len is non-zero" << std::endl;
      exit(1);
    }
  }
  #endif

  auto mapping = fillBuckets(algoconfig, A, B);
  std::vector<std::tuple<int, int, int>> buckets(algoconfig.tilesUsed, {0, 0, 0});

  seqMapping = std::vector<int>(A.size(), 0);
  for (const auto [bucket, i] : mapping) {
    auto& [bN, bA, bB] = buckets[bucket];
    auto aSize = vA[i].size();
    auto bSize = vB[i].size();

    size_t offsetBuffer = bucket * algoconfig.bufsize;
    size_t offsetLength = bucket * algoconfig.maxBatches;

    a_len[(offsetLength + bN)*2] = aSize;
    b_len[(offsetLength + bN)*2] = bSize;
    a_len[(offsetLength + bN)*2+1] = bA;
    b_len[(offsetLength + bN)*2+1] = bB;
    seqMapping[i] = offsetLength + bN;

    memcpy(a + offsetBuffer + bA, vA[i].data(), aSize);
    memcpy(b + offsetBuffer + bB, vB[i].data(), bSize);

    bN++;
    bA += aSize;
    bB += bSize;
  }

  preprocessTimer.stop();

#ifdef IPUMA_DEBUG
  int emptyBuckets = 0;
  std::vector<int> bucketCmps;
  std::map<int, int> occurence;
  for (auto [n, bA, bB] : buckets) {
    if (n == 0) emptyBuckets++;
    occurence[n]++;
    bucketCmps.push_back(n);
  }
  std::stringstream ss;
  ss << "Map[";
  for (auto [k ,v] : occurence) {
    ss << k << ": " << v << ",";
  }
  ss << "]";
  // SLOG(swatlib::printVector(bucketCmps), "\n");
  SLOG("Total number of buckets: ", buckets.size(), " empty buckets: ", emptyBuckets, "\n");
  SLOG("Bucket size occurence: ", ss.str(), "\n");
#endif
  // SLOG("Inner comparison time: ", preprocessTimer.get_elapsed(), " engine run: ", engineTimer.get_elapsed(), "\n");
}
}  // namespace batchaffine
}  // namespace ipu