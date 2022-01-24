#include <sstream>
#include <string>
#include <vector>

#include "ssw.hpp"
#include "gtest/gtest.h"

#include "klign/klign.hpp"
#include "klign/aligner_cpu.hpp"
#include "klign/adept-sw/driver.hpp"

#ifdef ENABLE_IPUS
#include "klign/ipuma-sw/ipu_batch_affine.h"
#include "klign/ipuma-sw/vector.hpp"
#endif

using std::max;
using std::min;
using std::string;
using std::vector;

using namespace StripedSmithWaterman;

#ifdef ENABLE_IPUS
void test_aligns_ipu(vector<Alignment> &alns, vector<string> query, vector<string> ref, ipu::batchaffine::SWAlgorithm &algo) {
  alns.reserve(query.size());

  algo.compare_local(query, ref);
  auto aln_results = algo.get_result();

  for (int i = 0; i < query.size(); i++) {
    alns.push_back({});
    Alignment &aln = alns[i];

    auto uda = aln_results.a_range_result[i];
    int16_t query_begin = uda & 0xffff;
    int16_t query_end = uda >> 16;

    auto udb = aln_results.b_range_result[i];
    int16_t ref_begin = udb & 0xffff;
    int16_t ref_end = udb >> 16;

    aln.query_end = query_end;
    aln.query_begin = query_begin;
    aln.ref_begin = ref_begin;
    aln.ref_end = ref_end;
    aln.sw_score = aln_results.scores[i];
    aln.sw_score_next_best = 0;
    aln.mismatches = 0;
  }
}

void check_alns_ipu(vector<Alignment> &alns, vector<int> qstart, vector<int> qend, vector<int> rstart, vector<int> rend) {
  int i = 0;

  EXPECT_TRUE(alns.size() == qstart.size()) << "Number of alignments does not equal number of comparisons (" << qstart.size() <<")";

  for (Alignment &aln : alns) {
    if (i == 15) {  // mismatch test
      EXPECT_TRUE(aln.ref_end - aln.ref_begin <= 3) << "adept.ref_begin:" << aln.ref_begin << "\tadept.ref_end:" << aln.ref_end;
      EXPECT_TRUE(aln.query_end - aln.query_begin <= 3)
          << "\tadept.query_begin:" << aln.query_begin << "\tadept.query_end:" << aln.query_end;
      EXPECT_TRUE(aln.sw_score <= 4);
      EXPECT_TRUE(aln.sw_score_next_best == 0);
    } else {
      EXPECT_EQ(aln.ref_begin, rstart[i]) << i << ": adept.ref_begin:" << aln.ref_begin << "\t"
                                          << "correct ref_begin:" << rstart[i];
      EXPECT_EQ(aln.ref_end, rend[i]) << "\tadept.ref_end:" << aln.ref_end << "\tcorrect ref_end:" << rend[i];
      EXPECT_EQ(aln.query_begin, qstart[i]) << "\tadept.query_begin:" << aln.query_begin << "\tcorrect query_begin:" << qstart[i];
      EXPECT_EQ(aln.query_end, qend[i]) << "\tadept.query_end:" << aln.query_end << "\tcorrect query end:" << qend[i];
    }
    i++;
  }
}

// TEST(MHMTest, ipuasmtest) {
//   int numWorkers = 1;
//   int numCmps = 3;
//   int strlen = 300;
//   int bufsize = 600;
//   auto driver = ipu::batchaffine::SWAlgorithm({
//     // .gapInit = -ALN_GAP_OPENING_COST,
//     .gapInit = 0,
//     .gapExtend = -ALN_GAP_EXTENDING_COST,
//     .matchValue = ALN_MATCH_SCORE,
//     .mismatchValue = -ALN_MISMATCH_COST,
//     .similarity = swatlib::Similarity::nucleicAcid,
//     .datatype = swatlib::DataType::nucleicAcid,
//   }, {numWorkers, strlen, numCmps, bufsize, ipu::batchaffine::VertexType::cpp});
//   vector<string> refs, queries;
// 
//        // refs.push_back("AAAATTTTCCCCGGGG");
//        // queries.push_back("GCTAGCTAGCTAGCTA");
//         // auto ref =   "AAAATTTTCCCCGGGG";
//         // auto query = "AAAATTTCCCCGGGG";
//         auto ref =   "AATTCC";
//         auto query = "AATCC";
//         refs.push_back(ref);
//         queries.push_back(query);
//   // generate input strings
//   //for (int i = 0; i < numCmps * numWorkers; ++i) {
//     // refs.push_back(string(strlen, 'A'));
//     // queries.push_back(string(strlen, 'T'));
//     // refs.push_back("AACGT");
//     // queries.push_back("ACGT");
//     // refs.push_back("AAAATTTTCCCCGGGG");
//     // queries.push_back("GCTAGCTAGCTAGCTA");
//     //     auto ref =   "AAAATTTTCCCCGGGG";
//     //     auto query = "AAAATTTCCCCGGGG";
//     //     refs.push_back(ref);
//     //     queries.push_back(query);
//   // }
//   driver.compare(queries, refs);
//   auto aln_results = driver.get_result();
//   for (int i = 0; i < numCmps * numWorkers; ++i) {
//     auto uda = aln_results.a_range_result[i];
//     int16_t query_begin = uda & 0xffff;
//     int16_t query_end = uda >> 16;
// 
//     auto udb = aln_results.b_range_result[i];
//     int16_t ref_begin = udb & 0xffff;
//     int16_t ref_end = udb >> 16;
//     std::cout << "Scores: " << aln_results.scores[i] << "\n";
//     std::cout << "Query Begin: " << query_begin << " End: " <<  query_end << "\n";
//     std::cout << "Refer Begin: " << ref_begin << " End: " <<  ref_end << "\n";
// 
// 
//     vector<Alignment> alns(6);
//     test_aligns(alns, queries[i], refs[i]);
//     for (auto& aln : alns) {
//       std::cout << "Score: " << aln.sw_score << " Alingment: Ref: " << aln.ref_begin << " " << aln.ref_end << " : Query: " << aln.query_begin << " " << aln.query_end <<  "\n";
//       std::cout << "cigar: " << aln.cigar_string << "\n";
//     }
//   }
// }

inline string aln2string(Alignment &aln) {
  std::stringstream ss;
  ss << "score=" << aln.sw_score << " score2=" << aln.sw_score_next_best;
  ss << " rbegin=" << aln.ref_begin << " rend=" << aln.ref_end;
  ss << " qbegin=" << aln.query_begin << " qend=" << aln.query_end;
  ss << " rend2=" << aln.ref_end_next_best << " mismatches=" << aln.mismatches;
  ss << " cigarstr=" << aln.cigar_string;
  return ss.str();
}

TEST(MHMTest, ipuparity) {
  vector<string> queries = {
    "AATGAGAATGATGTCGTTCGAAATTTGACCAGTCAAACCGCGGGCAATAAGGTCTTCGTTCAGGGCATAGACCTTAATGGGGGCATTACGCAGACTTTCA",
    "ATCTGGCAGGTAAAGATGAGCTCAACAAAGTGATCCAGCATTTTGGCAAAGGAGGCTTTGATGTGATTACTCGCGGTCAGGTGCCACCTAACCCGTCTGA"
  };
  vector<string> refs = {
    "AATGAGAATGATGTCNTTCNAAATTTGACCAGTCAAACCGCGGGCAATAAGGTCTTCGTTCAGGGCATAGACCTTAATGGGGGCATTACGCAGACTTTCA",
    "ATCTGGCAGGTAAAGATGAGCTCAACAAAGTGATCCAGCATTTTGGCAAAGGAGGCTTTGATGTGATTACTCGCGGTCAGGTGCCACCTAANNCGTCTGA"
  };
  AlnScoring aln_scoring = {.match = ALN_MATCH_SCORE,
                            .mismatch = ALN_MISMATCH_COST,
                            .gap_opening = ALN_GAP_OPENING_COST,
                            .gap_extending = ALN_GAP_EXTENDING_COST,
                            .ambiguity = ALN_AMBIGUITY_COST};

  Aligner cpu_aligner(aln_scoring.match, aln_scoring.mismatch, aln_scoring.gap_opening, aln_scoring.gap_extending,
                            aln_scoring.ambiguity);
  Filter ssw_filter(true, false, 0, 32767);

  int numWorkers = 1;
  int numCmps = 10;
  int strlen = 120;
  int bufsize = 1000;
  auto driver = ipu::batchaffine::SWAlgorithm({
    .gapInit = 0,
    .gapExtend = -ALN_GAP_EXTENDING_COST,
    .matchValue = ALN_MATCH_SCORE,
    .mismatchValue = -ALN_MISMATCH_COST,
    .ambiguityValue = -ALN_AMBIGUITY_COST,
    .similarity = swatlib::Similarity::nucleicAcid,
    .datatype = swatlib::DataType::nucleicAcid,
  }, {numWorkers, strlen, numCmps, bufsize, ipu::batchaffine::VertexType::assembly});

  vector<Alignment> alns_ipu(queries.size());
  test_aligns_ipu(alns_ipu, queries, refs, driver);

  vector<Alignment> alns(queries.size());
  for (int i = 0; i < queries.size(); ++i) {
    auto reflen = refs[i].size();
    auto qlen = queries[i].size();
    auto masklen = max((int)min(reflen, qlen) / 2, 15);
    cpu_aligner.Align(queries[i].c_str(), refs[i].c_str(), qlen, ssw_filter, &alns[i], masklen);

    EXPECT_EQ(alns[i].sw_score, alns_ipu[i].sw_score) << i << ": IPU score result does not match CPU SSW";
    EXPECT_EQ(alns[i].ref_begin, alns_ipu[i].ref_begin) << i << ": IPU reference start result does not match CPU SSW";
    EXPECT_EQ(alns[i].ref_end, alns_ipu[i].ref_end) << i << ": IPU reference end result does not match CPU SSW";
    EXPECT_EQ(alns[i].query_begin, alns_ipu[i].query_begin) << i << ": IPU query start result does not match CPU SSW";
    EXPECT_EQ(alns[i].query_end, alns_ipu[i].query_end) << i << ": IPU query end result does not match CPU SSW";
  }
}

TEST(MHMTest, DISABLED_ipumultivert) {
  int numWorkers = 1;
  int numCmps = 30;
  int strlen = 20;
  int bufsize = 1000;
  auto driver = ipu::batchaffine::SWAlgorithm({
    .gapInit = 0,
    .gapExtend = -ALN_GAP_EXTENDING_COST,
    .matchValue = ALN_MATCH_SCORE,
    .mismatchValue = -ALN_MISMATCH_COST,
    .ambiguityValue = -ALN_AMBIGUITY_COST,
    .similarity = swatlib::Similarity::nucleicAcid,
    .datatype = swatlib::DataType::nucleicAcid,
  }, {numWorkers, strlen, numCmps, bufsize, ipu::batchaffine::VertexType::multi});

  vector<string> queries, refs;
  queries.push_back("AAAAAA");
  refs.push_back("AAAAAA");
  driver.compare_local(queries, refs);
  auto aln_results = driver.get_result();
  std::cout << "Aln results:\n";
  std::cout << swatlib::printVector(aln_results.scores) << "\n";
}

TEST(MHMTest, ipumaasm) {
  int numWorkers = 1;
  int numCmps = 30;
  int strlen = 20;
  int bufsize = 1000;
  auto driver = ipu::batchaffine::SWAlgorithm({
    .gapInit = 0,
    .gapExtend = -ALN_GAP_EXTENDING_COST,
    .matchValue = ALN_MATCH_SCORE,
    .mismatchValue = -ALN_MISMATCH_COST,
    .ambiguityValue = -ALN_AMBIGUITY_COST,
    .similarity = swatlib::Similarity::nucleicAcid,
    .datatype = swatlib::DataType::nucleicAcid,
  }, {numWorkers, strlen, numCmps, bufsize, ipu::batchaffine::VertexType::assembly});
  std::cout << "Initialized IPU\n";

  vector<Alignment> alns;
  vector<string> refs, queries;
  vector<int> qstarts, qends, rstarts, rends;
  // first test
  string ref = "ACGT";
  string query = ref;
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // second test
  ref = "AACGT";
  query = "ACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(1);
  rends.push_back(4);
  // third test
  ref = "ACGTT";
  query = "ACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // fourth test
  ref = "ACGT";
  query = "TACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(1);
  qends.push_back(4);
  rstarts.push_back(0);
  rends.push_back(3);
  // fifth test
  ref = "ACGT";
  query = "TTACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(2);
  qends.push_back(5);
  rstarts.push_back(0);
  rends.push_back(3);
  // sixth test
  ref = "ACGT";
  query = "ACGTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // seventh test
  ref = "ACGT";
  query = "ACGTTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // eighth test
  ref = "ACGT";
  query = "TACGTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(1);
  qends.push_back(4);
  rstarts.push_back(0);
  rends.push_back(3);
  // ninth test
  ref = "ACGT";
  query = "TTACGTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(2);
  qends.push_back(5);
  rstarts.push_back(0);
  rends.push_back(3);
  // tenth test
  ref = "ACGT";
  query = "TACGTTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(1);
  qends.push_back(4);
  rstarts.push_back(0);
  rends.push_back(3);
  // eleventh test
  ref = "ACGT";
  query = "TTACGTTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(2);
  qends.push_back(5);
  rstarts.push_back(0);
  rends.push_back(3);
  // twelvth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTCCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(15);
  rstarts.push_back(0);
  rends.push_back(15);
  // thirteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTACCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(15);
  rstarts.push_back(0);
  rends.push_back(15);
  // 1 insert // fourteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTACCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(16);
  rstarts.push_back(0);
  rends.push_back(15);
  // 1 del // fifteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTCCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(14);
  rstarts.push_back(0);
  rends.push_back(15);

  // no match // sixteenth
  ref = "AAAATTTTCCCCGGGG";
  query = "GCTAGCTAGCTAGCTA";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(3);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(0);

  // soft clip start // seventeenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "GCTAAAATTTTCCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(3);
  qends.push_back(18);
  rstarts.push_back(0);
  rends.push_back(15);
  // soft clip end // eighteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTCCCCGGGGACT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(15);
  rstarts.push_back(0);
  rends.push_back(15);

  test_aligns_ipu(alns, queries, refs, driver);
  check_alns_ipu(alns, qstarts, qends, rstarts, rends);
}

TEST(MHMTest, ipuma) {
  double time_to_initialize;
  int device_count;
  size_t total_mem;
  auto driver = ipu::batchaffine::SWAlgorithm({}, {
    .tilesUsed = 2,
    .maxAB = 300,
    .maxBatches = 20,
    .bufsize = 3000,
    .vtype = ipu::batchaffine::VertexType::cpp,
    .fillAlgo = ipu::batchaffine::partition::Algorithm::roundRobin
  });
  std::cout << "Initialized IPU\n";

  vector<Alignment> alns;
  vector<string> refs, queries;
  vector<int> qstarts, qends, rstarts, rends;
  // first test
  string ref = "ACGT";
  string query = ref;
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // second test
  ref = "AACGT";
  query = "ACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(1);
  rends.push_back(4);
  // third test
  ref = "ACGTT";
  query = "ACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // fourth test
  ref = "ACGT";
  query = "TACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(1);
  qends.push_back(4);
  rstarts.push_back(0);
  rends.push_back(3);
  // fifth test
  ref = "ACGT";
  query = "TTACGT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(2);
  qends.push_back(5);
  rstarts.push_back(0);
  rends.push_back(3);
  // sixth test
  ref = "ACGT";
  query = "ACGTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // seventh test
  ref = "ACGT";
  query = "ACGTTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(3);
  rstarts.push_back(0);
  rends.push_back(3);
  // eighth test
  ref = "ACGT";
  query = "TACGTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(1);
  qends.push_back(4);
  rstarts.push_back(0);
  rends.push_back(3);
  // ninth test
  ref = "ACGT";
  query = "TTACGTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(2);
  qends.push_back(5);
  rstarts.push_back(0);
  rends.push_back(3);
  // tenth test
  ref = "ACGT";
  query = "TACGTTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(1);
  qends.push_back(4);
  rstarts.push_back(0);
  rends.push_back(3);
  // eleventh test
  ref = "ACGT";
  query = "TTACGTTT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(2);
  qends.push_back(5);
  rstarts.push_back(0);
  rends.push_back(3);
  // twelvth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTCCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(15);
  rstarts.push_back(0);
  rends.push_back(15);
  // thirteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTACCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(15);
  rstarts.push_back(0);
  rends.push_back(15);
  // 1 insert // fourteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTACCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(16);
  rstarts.push_back(0);
  rends.push_back(15);
  // 1 del // fifteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTCCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(14);
  rstarts.push_back(0);
  rends.push_back(15);
  // no match // sixteenth
  ref = "AAAATTTTCCCCGGGG";
  query = "GCTAGCTAGCTAGCTA";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(14);
  rstarts.push_back(0);
  rends.push_back(15);

  // soft clip start // seventeenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "GCTAAAATTTTCCCCGGGG";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(3);
  qends.push_back(18);
  rstarts.push_back(0);
  rends.push_back(15);
  // soft clip end // eighteenth test
  ref = "AAAATTTTCCCCGGGG";
  query = "AAAATTTTCCCCGGGGACT";
  queries.push_back(query);
  refs.push_back(ref);
  qstarts.push_back(0);
  qends.push_back(15);
  rstarts.push_back(0);
  rends.push_back(15);

  test_aligns_ipu(alns, queries, refs, driver);
  check_alns_ipu(alns, qstarts, qends, rstarts, rends);
}
#endif