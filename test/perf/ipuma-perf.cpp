#include <sstream>
#include <string>
#include <vector>
#include <tuple>

#include "ssw.hpp"
#include "gtest/gtest.h"

#include "klign/klign.hpp"
#include "klign/aligner_cpu.hpp"

#ifdef ENABLE_IPUS
#include "klign/ipuma-sw/ipu_batch_affine.h"
#include "klign/ipuma-sw/vector.hpp"
#endif

using std::max;
using std::min;
using std::string;
using std::vector;
using std::tuple;

using namespace StripedSmithWaterman;
vector<tuple<string, string>> INPUT_BATCHS = {
  {"/global/D1/projects/ipumer/inputs_ab/batch_0_A.txt", "/global/D1/projects/ipumer/inputs_ab/batch_0_B.txt"},
  {"/global/D1/projects/ipumer/inputs_ab/batch_1_A.txt", "/global/D1/projects/ipumer/inputs_ab/batch_1_B.txt"},
  {"/global/D1/projects/ipumer/inputs_ab/batch_2_A.txt", "/global/D1/projects/ipumer/inputs_ab/batch_2_B.txt"},
};

string aln2string(Alignment &aln) {
  std::stringstream ss;
  ss << "score=" << aln.sw_score << " score2=" << aln.sw_score_next_best;
  ss << " rbegin=" << aln.ref_begin << " rend=" << aln.ref_end;
  ss << " qbegin=" << aln.query_begin << " qend=" << aln.query_end;
  ss << " rend2=" << aln.ref_end_next_best << " mismatches=" << aln.mismatches;
  ss << " cigarstr=" << aln.cigar_string;
  return ss.str();
}

AlnScoring aln_scoring = {.match = ALN_MATCH_SCORE,
                          .mismatch = ALN_MISMATCH_COST,
                          .gap_opening = ALN_GAP_OPENING_COST,
                          .gap_extending = ALN_GAP_EXTENDING_COST,
                          .ambiguity = ALN_AMBIGUITY_COST};
AlnScoring cigar_aln_scoring = {.match = 2, .mismatch = 4, .gap_opening = 4, .gap_extending = 2, .ambiguity = 1};

std::vector<std::string> loadSequences(const std::string& path) {
  std::vector<std::string> sequences;
  std::ifstream seqFile(path);
  string line;
  while (std::getline(seqFile, line)) {
    sequences.push_back(line);
  }
  return sequences;
}

#ifdef ENABLE_IPUS

class PerformanceBase : public ::testing::Test {
protected:
  vector<string> refs, queries;
};

class AlgoPerformance : public PerformanceBase, public ::testing::WithParamInterface<ipu::batchaffine::VertexType> {
};

TEST_P(AlgoPerformance, RunOptimal) {
  auto algotype = GetParam();
  int numWorkers = 8832;
  int numCmps = 40;
  int strlen = 150;
  if (algotype == ipu::batchaffine::VertexType::multi) {
    numWorkers = numWorkers / 6;
    numCmps = numCmps * 6;
  }
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, numCmps * strlen, algotype});

  // generate input strings
  for (int i = 0; i < numCmps * numWorkers; ++i) {
    refs.push_back(string(strlen, 'A'));
    queries.push_back(string(strlen, 'T'));
  }
  driver.compare_local(queries, refs);
  // auto alns = driver.get_result();
  // for (int i = 0; i < numWorkers * numCmps; ++i) {
  //   std::cout << alns.scores[i] << " ";
  // }
  // std::cout << "\n";
}

INSTANTIATE_TEST_SUITE_P(
  VertexTypePerformance,
  AlgoPerformance,
  testing::Values(ipu::batchaffine::VertexType::cpp, ipu::batchaffine::VertexType::assembly, ipu::batchaffine::VertexType::multi)
  );

class PartitionPerformance : public PerformanceBase, public ::testing::WithParamInterface<ipu::batchaffine::partition::Algorithm> {
};

TEST_P(PartitionPerformance, RealBatches) {
  int numWorkers = 8832;
  int numCmps = 40;
  int strlen = 200;

  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, numCmps * strlen, ipu::batchaffine::VertexType::assembly, GetParam()});
  for (auto& [path_a, path_b] : INPUT_BATCHS) {
    refs = loadSequences(path_a);
    queries = loadSequences(path_b);
    std::cout << "Len A: " << refs.size() << " Len B: " << queries.size() << "\n";
    driver.compare_local(refs, queries);
  }
}

INSTANTIATE_TEST_SUITE_P(
  PartitionTests,
  PartitionPerformance,
  testing::Values(ipu::batchaffine::partition::Algorithm::fillFirst, ipu::batchaffine::partition::Algorithm::roundRobin, ipu::batchaffine::partition::Algorithm::greedy)
  );
#endif