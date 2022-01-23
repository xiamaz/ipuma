#include <sstream>
#include <string>
#include <vector>

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

#ifdef ENABLE_IPUS
TEST(MHMTest, ipumaperfasm) {
  int numWorkers = 8832;
  int numCmps = 200;
  int strlen = 150;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, 40000, ipu::batchaffine::VertexType::assembly});
  vector<string> refs, queries;

  // generate input strings
  for (int i = 0; i < numCmps * numWorkers; ++i) {
    refs.push_back(string(strlen, 'A'));
    queries.push_back(string(strlen, 'T'));
  }
  driver.compare_local(queries, refs);
}

TEST(MHMTest, ipumaperfcpp) {
  int numWorkers = 8832;
  int numCmps = 200;
  int strlen = 150;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {numWorkers, strlen, numCmps, 40000, ipu::batchaffine::VertexType::cpp});
  vector<string> refs, queries;

  // generate input strings
  for (int i = 0; i < numCmps * numWorkers; ++i) {
    refs.push_back(string(strlen, 'A'));
    queries.push_back(string(strlen, 'T'));
  }
  driver.compare_local(queries, refs);
}
#endif