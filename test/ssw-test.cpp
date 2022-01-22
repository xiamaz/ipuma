#include <sstream>
#include <string>
#include <vector>

#include "ssw.hpp"
#include "gtest/gtest.h"

#include "klign/klign.hpp"
#include "klign/aligner_cpu.hpp"
#include "klign/adept-sw/driver.hpp"

#ifdef ENABLE_GPUS
#include "gpu-utils/gpu_utils.hpp"
#endif

#ifdef ENABLE_IPUS
#include "klign/ipuma-sw/ipu_batch_affine.h"
#include "klign/ipuma-sw/vector.hpp"
#endif

using std::max;
using std::min;
using std::string;
using std::vector;

using namespace StripedSmithWaterman;

void translate_adept_to_ssw(Alignment &aln, const adept_sw::AlignmentResults &aln_results, int idx) {
  aln.sw_score = aln_results.top_scores[idx];
  aln.sw_score_next_best = 0;
#ifndef REF_IS_QUERY
  aln.ref_begin = aln_results.ref_begin[idx];
  aln.ref_end = aln_results.ref_end[idx];
  aln.query_begin = aln_results.query_begin[idx];
  aln.query_end = aln_results.query_end[idx];
#else
  aln.query_begin = aln_results.ref_begin[idx];
  aln.query_end = aln_results.ref_end[idx];
  aln.ref_begin = aln_results.query_begin[idx];
  aln.ref_end = aln_results.query_end[idx];
#endif
  aln.ref_end_next_best = 0;
  aln.mismatches = 0;
  aln.cigar_string.clear();
  aln.cigar.clear();
}

#ifdef ENABLE_GPUS
void test_aligns_gpu(vector<Alignment> &alns, vector<string> query, vector<string> ref, adept_sw::GPUDriver &gpu_driver) {
  alns.reserve(query.size());
  unsigned max_q_len = 0, max_ref_len = 0;
  for (int i = 0; i < query.size(); i++) {
    if (max_q_len < query[i].size()) max_q_len = query[i].size();
    if (max_ref_len < ref[i].size()) max_ref_len = ref[i].size();
  }
  gpu_driver.run_kernel_forwards(query, ref, max_q_len, max_ref_len);
  gpu_driver.kernel_block();
  gpu_driver.run_kernel_backwards(query, ref, max_q_len, max_ref_len);
  gpu_driver.kernel_block();

  auto aln_results = gpu_driver.get_aln_results();

  for (int i = 0; i < query.size(); i++) {

    Alignment &alignment = alns[i];
    translate_adept_to_ssw(alignment, aln_results, i);
  }
}

void check_alns_gpu(vector<Alignment> &alns, vector<int> qstart, vector<int> qend, vector<int> rstart, vector<int> rend) {
  int i = 0;
  for (Alignment &aln : alns) {
    if (i == 15) {  // mismatch test
      EXPECT_TRUE(aln.ref_end - aln.ref_begin <= 3) << "adept.ref_begin:" << aln.ref_begin << "\tadept.ref_end:" << aln.ref_end;
      EXPECT_TRUE(aln.query_end - aln.query_begin <= 3)
          << "\tadept.query_begin:" << aln.query_begin << "\tadept.query_end:" << aln.query_end;
      EXPECT_TRUE(aln.sw_score <= 4);
      EXPECT_TRUE(aln.sw_score_next_best == 0);
    } else {
      EXPECT_EQ(aln.ref_begin, rstart[i]) << "adept.ref_begin:" << aln.ref_begin << "\t"
                                          << "correct ref_begin:" << rstart[i];
      EXPECT_EQ(aln.ref_end, rend[i]) << "\tadept.ref_end:" << aln.ref_end << "\tcorrect ref_end:" << rend[i];
      EXPECT_EQ(aln.query_begin, qstart[i]) << "\tadept.query_begin:" << aln.query_begin << "\tcorrect query_begin:" << qstart[i];
      EXPECT_EQ(aln.query_end, qend[i]) << "\tadept.query_end:" << aln.query_end << "\tcorrect query end:" << qend[i];
    }
    i++;
  }
}
#endif

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

#endif

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

Aligner ssw_aligner;
Aligner ssw_aligner_mhm2(aln_scoring.match, aln_scoring.mismatch, aln_scoring.gap_opening, aln_scoring.gap_extending,
                         aln_scoring.ambiguity);
Aligner ssw_aligner_cigar(cigar_aln_scoring.match, cigar_aln_scoring.mismatch, cigar_aln_scoring.gap_opening,
                          cigar_aln_scoring.gap_extending, cigar_aln_scoring.ambiguity);

Filter ssw_filter(true, false, 0, 32767), ssw_filter_cigar(true, true, 0, 32767);

void test_aligns(vector<Alignment> &alns, string query, string ref) {
  alns.resize(6);
  auto reflen = ref.size();
  auto qlen = query.size();
  auto masklen = max((int)min(reflen, qlen) / 2, 15);
  ssw_aligner.Align(query.c_str(), ref.c_str(), reflen, ssw_filter, &alns[0], masklen);
  ssw_aligner.Align(query.c_str(), ref.c_str(), reflen, ssw_filter_cigar, &alns[1], masklen);

  ssw_aligner_mhm2.Align(query.c_str(), ref.c_str(), reflen, ssw_filter, &alns[2], masklen);
  ssw_aligner_mhm2.Align(query.c_str(), ref.c_str(), reflen, ssw_filter_cigar, &alns[3], masklen);

  ssw_aligner_cigar.Align(query.c_str(), ref.c_str(), reflen, ssw_filter, &alns[4], masklen);
  ssw_aligner_cigar.Align(query.c_str(), ref.c_str(), reflen, ssw_filter_cigar, &alns[5], masklen);
}

void check_alns(vector<Alignment> &alns, int qstart, int qend, int rstart, int rend, int mismatches, string query = "",
                string ref = "", string cigar = "") {
  for (Alignment &aln : alns) {
    EXPECT_EQ(aln.ref_begin, rstart) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    EXPECT_EQ(aln.ref_end, rend) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    EXPECT_EQ(aln.query_begin, qstart) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    EXPECT_EQ(aln.query_end, qend) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    if (!aln.cigar_string.empty()) {  // mismatches should be recorded...
      EXPECT_EQ(aln.mismatches, mismatches) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
      if (!cigar.empty())
        EXPECT_STREQ(aln.cigar_string.c_str(), cigar.c_str()) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    }
  }
}
void check_not_alns(vector<Alignment> &alns, string query = "", string ref = "") {
  for (Alignment &aln : alns) {
    EXPECT_TRUE(aln.ref_end - aln.ref_begin <= 2) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    EXPECT_TRUE(aln.query_end - aln.query_begin <= 2) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    EXPECT_TRUE(aln.sw_score <= 4) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
    EXPECT_TRUE(aln.sw_score_next_best == 0) << "query=" << query << " ref=" << ref << " aln=" << aln2string(aln);
  }
}

TEST(MHMTest, ssw) {
  // arrange
  // act
  // assert

  EXPECT_EQ(ssw_filter.report_cigar, false);
  EXPECT_EQ(ssw_filter_cigar.report_cigar, true);

  vector<Alignment> alns;
  string ref = "ACGT";
  string query = ref;
  test_aligns(alns, query, ref);
  check_alns(alns, 0, 3, 0, 3, 0, query, ref, "4=");
  ref = "AACGT";
  test_aligns(alns, query, ref);
  check_alns(alns, 0, 3, 1, 4, 0, query, ref, "4=");
  ref = "ACGTT";
  test_aligns(alns, query, ref);
  check_alns(alns, 0, 3, 0, 3, 0, query, ref, "4=");

  ref = "ACGT";

  query = "TACGT";
  test_aligns(alns, query, ref);
  check_alns(alns, 1, 4, 0, 3, 0, query, ref, "1S4=");
  query = "TTACGT";
  test_aligns(alns, query, ref);
  check_alns(alns, 2, 5, 0, 3, 0, query, ref, "2S4=");
  query = "ACGTT";
  test_aligns(alns, query, ref);
  check_alns(alns, 0, 3, 0, 3, 0, query, ref, "4=1S");
  query = "ACGTTT";
  test_aligns(alns, query, ref);
  check_alns(alns, 0, 3, 0, 3, 0, query, ref, "4=2S");

  query = "TACGTT";
  test_aligns(alns, query, ref);
  check_alns(alns, 1, 4, 0, 3, 0, query, ref, "1S4=1S");
  query = "TTACGTT";
  test_aligns(alns, query, ref);
  check_alns(alns, 2, 5, 0, 3, 0, query, ref, "2S4=1S");
  query = "TACGTTT";
  test_aligns(alns, query, ref);
  check_alns(alns, 1, 4, 0, 3, 0, query, ref, "1S4=2S");
  query = "TTACGTTT";
  test_aligns(alns, query, ref);
  check_alns(alns, 2, 5, 0, 3, 0, query, ref, "2S4=2S");

  string r = "AAAATTTTCCCCGGGG";
  string q = "AAAATTTTCCCCGGGG";
  test_aligns(alns, q, r);
  check_alns(alns, 0, 15, 0, 15, 0, q, r, "16=");

  // 1 subst
  q = "AAAATTTTACCCGGGG";
  test_aligns(alns, q, r);
  check_alns(alns, 0, 15, 0, 15, 1, q, r, "8=1X7=");

  // 1 insert
  q = "AAAATTTTACCCCGGGG";
  test_aligns(alns, q, r);
  check_alns(alns, 0, 16, 0, 15, 1, q, r, "8=1I8=");

  // 1 del
  q = "AAAATTTCCCCGGGG";
  test_aligns(alns, q, r);
  check_alns(alns, 0, 14, 0, 15, 1, q, r, "4=1D11=");

  // no match
  q = "GCTAGCTAGCTAGCTA";
  test_aligns(alns, q, r);
  check_not_alns(alns, q, r);

  // soft clip start
  q = "GCTAAAATTTTCCCCGGGG";
  test_aligns(alns, q, r);
  check_alns(alns, 3, 18, 0, 15, 0, q, r, "3S16=");

  // soft clip end
  q = "AAAATTTTCCCCGGGGACT";
  test_aligns(alns, q, r);
  check_alns(alns, 0, 15, 0, 15, 0, q, r, "16=3S");
}

TEST(MHMTest, AdeptSW) {
  // arrange
  // act
  // assert

  double time_to_initialize;
  int device_count;
  size_t total_mem;
#ifdef ENABLE_GPUS
  gpu_utils::initialize_gpu(time_to_initialize, 0);
//  if (device_count > 0) {
//    EXPECT_TRUE(total_mem > 32 * 1024 * 1024);  // >32 MB
//  }

  double init_time = 0;
  adept_sw::GPUDriver gpu_driver(0, 1, (short)aln_scoring.match, (short)-aln_scoring.mismatch, (short)-aln_scoring.gap_opening,
                                   (short)-aln_scoring.gap_extending, 300, init_time);
  std::cout << "Initialized gpu in " << time_to_initialize << "s and " << init_time << "s\n";
#endif

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

#ifdef ENABLE_GPUS
  // run kernel
  test_aligns_gpu(alns, queries, refs, gpu_driver);
  // verify results
  check_alns_gpu(alns, qstarts, qends, rstarts, rends);
  // cuda tear down happens in driver destructor
#endif

}


#ifdef ENABLE_IPUS
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
  auto driver = ipu::batchaffine::SWAlgorithm({}, {});
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