#include <string>
#include <cmath>
#include <iostream>

#include <plog/Log.h>

#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Zero.hpp>

#include "ipu_base.h"
#include "similarity.h"
#include "encoding.h"
#include "timing.hpp"
#include "vector.hpp"
#include "ipu_batch_affine.h"

namespace ipu {
namespace batchaffine {
namespace partition {
  int fillFirst(std::vector<std::tuple<int, int>>& mapping, const std::vector<std::string>& A, const std::vector<std::string>& B, int bucketCount, int bucketCapacity, int bucketCountCapacity) {
    mapping = std::vector<std::tuple<int, int>>(A.size(), {0, 0});
    std::vector<BucketData> buckets(bucketCount, {0, 0, 0, 0});
    int bucketIndex = 0;
    for (int i = 0; i < A.size(); ++i) {
      const auto& a = A[i];
      const auto& b = B[i];

      // find next empty bucket
      while (bucketIndex < bucketCount) {
        auto& [bN, bA, bB, _] = buckets[bucketIndex];
        if (bN + 1 <= bucketCountCapacity && bA + a.size() <= bucketCapacity && bB + b.size() <= bucketCapacity) {
          mapping[i] = {bucketIndex, i};
          bN++;
          bA += a.size();
          bB += b.size();
          // PLOGD << "i: " << i << " bucket index: " << bucketIndex << " cap: " << bA << "/" << bucketCapacity << " n: " << bN << "/" << bucketCountCapacity << "\n";
          break;
        } else {
          bucketIndex++;
        }
      }
      if (bucketIndex >= bucketCount) {
          return 1;
      }
    }

    return 0;
  }

  int roundRobin(std::vector<std::tuple<int, int>>& mapping, const std::vector<std::string>& A, const std::vector<std::string>& B, int bucketCount, int bucketCapacity, int bucketCountCapacity) {
    mapping = std::vector<std::tuple<int, int>>(A.size(), {0, 0});
    std::vector<BucketData> buckets(bucketCount, {0, 0, 0, 0});
    int bucketIndex = 0;
    for (int i = 0; i < A.size(); ++i) {
      const auto& a = A[i];
      const auto& b = B[i];

      // find next empty bucket
      int boff = 0;
      for (; boff < bucketCount; ++boff) {
        int bi = (bucketIndex + boff) % bucketCount;
        auto& [bN, bA, bB, _] = buckets[bi];
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
          return 1;
      }
      bucketIndex = (bucketIndex + 1) % bucketCount;
    }
    return 0;
  }

  // Greedy approach in which we always put current sequence into one with lowest weight
  int greedy(std::vector<std::tuple<int, int>>& mapping, const std::vector<std::string>& A, const std::vector<std::string>& B, int bucketCount, int bucketCapacity, int bucketCountCapacity) {
    mapping = std::vector<std::tuple<int, int>>(A.size(), {0, 0});
    std::vector<BucketData> buckets(bucketCount, {0, 0, 0, 0});
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
        return 1;
      }

      auto& [bN, bA, bB, bW] = buckets[smallestBucket];
      bN++;
      bA += a.size();
      bB += b.size();
      bW += weight;
      mapping[i] = {smallestBucket, i};
    }
    return 0;
    // return mapping;
  }
}

static const std::string CYCLE_COUNT_OUTER = "cycle-count-outer";
static const std::string CYCLE_COUNT_INNER = "cycle-count-inner";

long long getCellCount(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  long long cellCount = 0;
  if (A.size() != B.size()) {
    PLOGW << "Mismatch between size of A " << A.size() << " and size of B " << B.size();
  }
  // count cells based on 1:1 comparisons
  for (int i = 0; i < A.size(); ++i) {
    cellCount += A[i].size() * B[i].size();
  }
  return cellCount;
}

int IPUAlgoConfig::getTotalNumberOfComparisons() { return tilesUsed * maxBatches; }

int IPUAlgoConfig::getTotalBufferSize() { return tilesUsed * bufsize; }

int IPUAlgoConfig::getInputBufferSize() { return std::ceil(getTotalBufferSize() / 4) * 2 + getTotalNumberOfComparisons() * 2 * 2; }

std::string vertexTypeToString(VertexType v) { return typeString[static_cast<int>(v)]; }

/**
 * Streamable IPU graph for SW
 */
std::vector<program::Program> buildGraph(Graph& graph, VertexType vtype, unsigned long activeTiles, unsigned long maxAB,
                                         unsigned long bufSize, unsigned long maxBatches,
                                         const swatlib::Matrix<int8_t> similarityData, int gapInit, int gapExt) {
  program::Sequence prog;

  auto target = graph.getTarget();
  int tileCount = target.getTilesPerIPU();

  Tensor As = graph.addVariable(INT, {activeTiles, static_cast<size_t>(std::ceil(bufSize / 4))}, "A");
  Tensor Bs = graph.addVariable(INT, {activeTiles, static_cast<size_t>(std::ceil(bufSize / 4))}, "B");

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

  auto inputs_tensor = concat({As.flatten(), Alens.flatten(), Bs.flatten(), Blens.flatten()});
  auto outputs_tensor = concat({Scores.flatten(), ARanges.flatten(), BRanges.flatten()});

  auto host_stream_concat = graph.addHostToDeviceFIFO(HOST_STREAM_CONCAT, INT, inputs_tensor.numElements());
  auto device_stream_concat = graph.addDeviceToHostFIFO(STREAM_CONCAT_ALL, INT, Scores.numElements() + ARanges.numElements() + BRanges.numElements());
  auto h2d_prog_concat = program::Sequence({poplar::program::Copy(host_stream_concat, inputs_tensor)});
  auto d2h_prog_concat = program::Sequence({poplar::program::Copy(outputs_tensor, device_stream_concat)});
#ifdef IPUMA_DEBUG
 program::Sequence  main_prog;
  main_prog.add(program::Execute(frontCs));
  addCycleCount(graph, main_prog, CYCLE_COUNT_INNER);
#else
  auto main_prog = program::Execute(frontCs);
#endif
  prog.add(h2d_prog_concat);
  prog.add(main_prog);
  prog.add(d2h_prog_concat);

#ifdef IPUMA_DEBUG
  addCycleCount(graph, prog, CYCLE_COUNT_OUTER);
#endif
  return {prog, d2h_prog_concat};
}

SWAlgorithm::SWAlgorithm(ipu::SWConfig config, IPUAlgoConfig algoconfig, int thread_id) : IPUAlgorithm(config), algoconfig(algoconfig), thread_id(thread_id) {
  const auto totalComparisonsCount = algoconfig.getTotalNumberOfComparisons();
  const auto inputBufferSize = algoconfig.getTotalBufferSize();

  scores.resize(totalComparisonsCount);
  a_range_result.resize(totalComparisonsCount);
  b_range_result.resize(totalComparisonsCount);

  Graph graph = createGraph();

  auto similarityMatrix = swatlib::selectMatrix(config.similarity, config.matchValue, config.mismatchValue, config.ambiguityValue);
  std::vector<program::Program> programs =
      buildGraph(graph, algoconfig.vtype, algoconfig.tilesUsed, algoconfig.maxAB, algoconfig.bufsize, algoconfig.maxBatches,
                 similarityMatrix, config.gapInit, config.gapExtend);

  createEngine(graph, programs);
}

SWAlgorithm::SWAlgorithm(ipu::SWConfig config, IPUAlgoConfig algoconfig) : SWAlgorithm::SWAlgorithm(config, algoconfig, 0) {}

BlockAlignmentResults SWAlgorithm::get_result() { return {scores, a_range_result, b_range_result}; }

void SWAlgorithm::checkSequenceSizes(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B) {
  if (A.size() != B.size()) {
    PLOGW << "Mismatched size A " << A.size() << " != B " << B.size();
    exit(1);
  }
  if (A.size() > algoconfig.maxBatches * algoconfig.tilesUsed) {
    PLOGW << "A has more elements than the maxBatchsize";
    PLOGW << "A.size() = " << A.size();
    PLOGW << "max comparisons = " << algoconfig.tilesUsed << " * " << algoconfig.maxBatches;
    exit(1);
  }

  for (const auto& a : A) {
    if (a.size() > algoconfig.maxAB) {
      PLOGW << "Sequence size in a " << a.size() << " > " << algoconfig.maxAB;
      exit(1);
    }
  }
  for (const auto& b : B) {
    if (b.size() > algoconfig.maxAB) {
      PLOGW << "Sequence size in a " << b.size() << " > " << algoconfig.maxAB;
      exit(1);
    }
  }
}

std::vector<std::tuple<int, int>> SWAlgorithm::fillBuckets(const std::vector<std::string>& A, const std::vector<std::string>& B, int& err) {
  return fillBuckets(algoconfig, A, B, err);
}

std::vector<std::tuple<int, int>> SWAlgorithm::fillBuckets(IPUAlgoConfig& algoconfig,const std::vector<std::string>& A, const std::vector<std::string>& B, int& err) {
  std::vector<std::tuple<int, int>> bucket_pairs;
  switch (algoconfig.fillAlgo) {
  case partition::Algorithm::fillFirst:
    err = partition::fillFirst(bucket_pairs, A, B, algoconfig.tilesUsed, algoconfig.bufsize, algoconfig.maxBatches);
    break;
  case partition::Algorithm::roundRobin:
    err = partition::roundRobin(bucket_pairs, A, B, algoconfig.tilesUsed, algoconfig.bufsize, algoconfig.maxBatches);
    break;
  case partition::Algorithm::greedy:
    err = partition::greedy(bucket_pairs, A, B, algoconfig.tilesUsed, algoconfig.bufsize, algoconfig.maxBatches);
    break;
  }
  return bucket_pairs;
}

void SWAlgorithm::prepared_remote_compare(int32_t* inputs_begin, int32_t* inputs_end, int32_t* results_begin, int32_t* results_end) {
  // We have to reconnect the streams to new memory locations as the destination will be in a shared memroy region.
  engine->connectStream(HOST_STREAM_CONCAT, inputs_begin);
  engine->connectStream(STREAM_CONCAT_ALL, results_begin);

  swatlib::TickTock t;
  t.tick();
  engine->run(0);
  t.tock();

#ifdef IPUMA_DEBUG
  auto cyclesOuter = getTotalCycles(*engine, CYCLE_COUNT_OUTER);
  auto cyclesInner = getTotalCycles(*engine, CYCLE_COUNT_INNER);
  auto timeOuter = static_cast<double>(cyclesOuter) / getTarget().getTileClockFrequency();
  auto timeInner = static_cast<double>(cyclesInner) / getTarget().getTileClockFrequency();
  PLOGD << "Poplar cycle count: " << cyclesInner << "/" << cyclesOuter << " computed time (in s): " << timeInner << "/" << timeOuter;

  size_t alen_offset = std::ceil(algoconfig.getTotalBufferSize() / 4);
  size_t blen_offset = alen_offset + (algoconfig.getTotalNumberOfComparisons() * 2) + std::ceil(algoconfig.getTotalBufferSize() / 4);

  int32_t* a_len = inputs_begin + alen_offset;
  int32_t* b_len = inputs_begin + blen_offset;

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
  PLOGD << "Poplar estimated cells(" << cellCount << ") GCUPS " << GCUPSInner << "/" << GCUPSOuter;

  // dataCount - actual data content transferred
  // totalTransferSize - size of buffer being transferred
  double totalTransferSize = algoconfig.getTotalBufferSize() * 2;

  auto transferTime = timeOuter - timeInner;
  auto transferInfoRatio = static_cast<double>(dataCount) / totalTransferSize * 100;
  auto transferBandwidth = totalTransferSize / transferTime / 1e6;
  PLOGD << "Transfer time: " << transferTime << "s transfer ratio: " << transferInfoRatio << "% estimated bandwidth: " << transferBandwidth << "mb/s, per vertex: " << transferBandwidth / algoconfig.tilesUsed << "mb/s";
#endif
}

void SWAlgorithm::compare_local(const std::vector<std::string>& A, const std::vector<std::string>& B) {
  std::vector<int> mapping;
  size_t inputs_size = algoconfig.getInputBufferSize();
  std::vector<int32_t> inputs(inputs_size + 4);

  size_t results_size = scores.size() + a_range_result.size() + b_range_result.size();
  std::vector<int32_t> results(results_size + 4);

  inputs[0] = 0xDEADBEEF;
  inputs[1] = 0xDEADBEEF;
  inputs[inputs_size + (1) + 1] = 0xFEEBDAED;
  inputs[inputs_size + (1) + 2] = 0xFEEBDAED;
  prepare_remote(algoconfig, A, B, &*inputs.begin() + 2, &*inputs.end() - 2, mapping);

  if (inputs[0] != 0xDEADBEEF || inputs[1] != 0xDEADBEEF) {
    std::vector<int32_t> subset(inputs.begin(), inputs.begin() + 10);
    PLOGW << "Canary begin overwritten " << swatlib::printVector(subset);
    exit(1);
  }
  if (inputs[inputs_size + 2] != 0xFEEBDAED || inputs[inputs_size + 3] != 0xFEEBDAED) {
    std::vector<int32_t> subset(inputs.end() - 10, inputs.end());
    PLOGW << "Canary end overwritten " << swatlib::printVector(subset);
    exit(1);
  }

  // std::vector<int32_t> unord_scores(scores), unord_mismatches(mismatches), unord_a_range(a_range_result), unord_b_range(b_range_result);
  results[0] = 0xDEADBEEF;
  results[1] = 0xDEADBEEF;
  results[results_size + (1) + 1] = 0xFEEBDAED;
  results[results_size + (1) + 2] = 0xFEEBDAED;
  // prepared_remote_compare(a.data(), a_len.data(), b.data(), b_len.data(), unord_scores.data(), unord_mismatches.data(), unord_a_range.data(),
  //                         unord_b_range.data());
  prepared_remote_compare(&*inputs.begin() + 2, &*inputs.end() - 2, &*results.begin() + 2, &*results.end() - 2);

  // check canaries
  if (results[0] != 0xDEADBEEF || results[1] != 0xDEADBEEF) {
    std::vector<int32_t> subset(results.begin(), results.begin() + 10);
    PLOGW << "Canary begin overwritten " << swatlib::printVector(subset);
    exit(1);
  }
  if (results[results_size + 2] != 0xFEEBDAED || results[results_size + 3] != 0xFEEBDAED) {
    std::vector<int32_t> subset(results.end() - 10, results.end());
    PLOGW << "Canary end overwritten " << swatlib::printVector(subset);
    exit(1);
  }

  // reorder results based on mapping
  int nthTry = 0;
  int sc;
  retry:
  nthTry++;
  sc = 0;
  for (size_t i = 0; i < mapping.size(); ++i) {
    size_t mapped_i = mapping[i];
    scores[i] = results[mapped_i + 2];
    if (scores[i] >= KLIGN_IPU_MAXAB_SIZE) {
      PLOGW << "ERROR Expected " << A.size() << " valid comparisons. But got " << i << " instead.";
      PLOGW.printf("ERROR Thread %d received wrong data FIRST, try again data=%d, map_translate=%d\n", thread_id, scores[i], mapping[i] + 2);
      engine->run(1);
      goto retry;
    }
    sc += scores[i] > 0;
    size_t a_range_offset = scores.size();
    size_t b_range_offset = a_range_offset + a_range_result.size();
    a_range_result[i] = results[a_range_offset + mapped_i + 2];
    b_range_result[i] = results[b_range_offset + mapped_i + 2];
  }
  if ((double)sc/A.size() < 0.5) {
    PLOGW << "ERROR Too many scores are 0, retry number " << (nthTry - 1);
    engine->run(1);
    goto retry;
  }
  // return;
// retry:
//   prepared_remote_compare(a.data(), a_len.data(), b.data(), b_len.data(), unord_scores.data(), unord_mismatches.data(), unord_a_range.data(),
//                           unord_b_range.data());
//   // reorder results based on mapping
//   for (int i = 0; i < mapping.size(); ++i) {
//     scores[i] = unord_scores[mapping[i]];
//     if (scores[i] >= KLIGN_IPU_MAXAB_SIZE) {
//       printf("Thread %d received wrong data AGAIN, FATAL data=%d, map_translate=%d\n", -1, scores[i], mapping[i]);
//       exit(1);
//     }
//     mismatches[i] = unord_mismatches[mapping[i]];
//     a_range_result[i] = unord_a_range[mapping[i]];
//     b_range_result[i] = unord_b_range[mapping[i]];
//   }
}

void SWAlgorithm::prepare_remote(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B, int32_t* inputs_begin, int32_t* inputs_end, std::vector<int>& seqMapping) {
  swatlib::TickTock preprocessTimer;
  preprocessTimer.tick();
  checkSequenceSizes(algoconfig, A, B);

  auto encoder = swatlib::getEncoder(swatlib::DataType::nucleicAcid);
  auto vA = encoder.encode(A);
  auto vB = encoder.encode(B);

  size_t input_elems = inputs_end - inputs_begin;
  memset(inputs_begin, 0, input_elems * sizeof(int32_t));

  #ifdef IPUMA_DEBUG
  for (int32_t* it = inputs_begin; it != inputs_end; ++it) {
    if (*it != 0) {
      PLOGW << "Results are not zero";
    }
  }
  #endif

  int errval = 0;
  auto mapping = fillBuckets(algoconfig, A, B, errval);

  if (errval) {
    PLOGW << "Bucket filling failed.";
    exit(1);
  }

  std::vector<std::tuple<int, int, int>> buckets(algoconfig.tilesUsed, {0, 0, 0});

  seqMapping = std::vector<int>(A.size(), 0);
  size_t a_offset = 0;
  size_t alen_offset = std::ceil(algoconfig.getTotalBufferSize() / 4);
  size_t b_offset = alen_offset + (algoconfig.getTotalNumberOfComparisons() * 2);
  size_t blen_offset = b_offset + std::ceil(algoconfig.getTotalBufferSize() / 4);

  int8_t* a = (int8_t*)inputs_begin;
  int32_t* a_len = inputs_begin + alen_offset;
  int8_t* b = (int8_t*)(inputs_begin + b_offset);
  int32_t* b_len = inputs_begin + blen_offset;

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

  preprocessTimer.tock();

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
  PLOGD << "Total number of buckets: " << buckets.size() << " empty buckets: " << emptyBuckets;
  PLOGD << "Bucket size occurence: " << ss.str();
#endif
  // SLOG("Inner comparison time: ", preprocessTimer.get_elapsed(), " engine run: ", engineTimer.get_elapsed(), "\n");
}
}  // namespace batchaffine
}  // namespace ipu