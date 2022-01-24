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
TEST(MHMTest, DISABLED_ipumaperfasm) {
  int numWorkers = 8832;
  int numCmps = 40;
  int strlen = 150;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, numCmps * strlen, ipu::batchaffine::VertexType::assembly});
  vector<string> refs, queries;

  // generate input strings
  for (int i = 0; i < numCmps * numWorkers; ++i) {
    refs.push_back(string(strlen, 'A'));
    queries.push_back(string(strlen, 'T'));
  }
  driver.compare_local(queries, refs);
}

TEST(MHMTest, DISABLED_ipumaperfcpp) {
  int numWorkers = 8832;
  int numCmps = 40;
  int strlen = 150;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, numCmps * strlen, ipu::batchaffine::VertexType::cpp});
  vector<string> refs, queries;

  // generate input strings
  for (int i = 0; i < numCmps * numWorkers; ++i) {
    refs.push_back(string(strlen, 'A'));
    queries.push_back(string(strlen, 'T'));
  }
  driver.compare_local(queries, refs);
}

TEST(MHMTest, DISABLED_ipupartitionfill) {
  int numWorkers = 8832;
  int numCmps = 30;
  int strlen = 200;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, numCmps * strlen, ipu::batchaffine::VertexType::assembly, ipu::batchaffine::partition::Algorithm::fillFirst});
  vector<string> refs, queries;
  vector<tuple<string, string>> paths = {
    {"/global/D1/projects/ipumer/inputs_ab/batch_0_A.txt", "/global/D1/projects/ipumer/inputs_ab/batch_0_B.txt"} 
  };
  for (auto& [path_a, path_b] : paths) {
    refs = loadSequences(path_a);
    queries = loadSequences(path_b);
  }
  std::cout << "Len A: " << refs.size() << " Len B: " << queries.size() << "\n";
  driver.compare_local(refs, queries);
}

TEST(MHMTest, DISABLED_ipupartitionroundrobin) {
  int numWorkers = 8832;
  int numCmps = 30;
  int strlen = 200;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, numCmps * strlen, ipu::batchaffine::VertexType::assembly, ipu::batchaffine::partition::Algorithm::roundRobin});
  vector<string> refs, queries;
  vector<tuple<string, string>> paths = {
    {"/global/D1/projects/ipumer/inputs_ab/batch_0_A.txt", "/global/D1/projects/ipumer/inputs_ab/batch_0_B.txt"} 
  };
  for (auto& [path_a, path_b] : paths) {
    refs = loadSequences(path_a);
    queries = loadSequences(path_b);
  }
  std::cout << "Len A: " << refs.size() << " Len B: " << queries.size() << "\n";
  driver.compare_local(refs, queries);
}

TEST(MHMTest, ipupartitiongreedy) {
  int numWorkers = 8832;
  int numCmps = 30;
  int strlen = 200;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, numCmps * strlen, ipu::batchaffine::VertexType::assembly, ipu::batchaffine::partition::Algorithm::greedy});
  vector<string> refs, queries;
  vector<tuple<string, string>> paths = {
    {"/global/D1/projects/ipumer/inputs_ab/batch_0_A.txt", "/global/D1/projects/ipumer/inputs_ab/batch_0_B.txt"} 
  };
  for (auto& [path_a, path_b] : paths) {
    refs = loadSequences(path_a);
    queries = loadSequences(path_b);
  }
  std::cout << "Len A: " << refs.size() << " Len B: " << queries.size() << "\n";
  driver.compare_local(refs, queries);
}
#endif