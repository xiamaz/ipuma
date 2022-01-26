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

struct SequenceTestData {
  string ref;
  string query;
  int qstart;
  int qend;
  int rstart;
  int rend;
  int score;
};

const vector<SequenceTestData> simpleData = {
  {"ACGT", "ACGT", 0, 3, 0, 3, 4},
  {"AACGT", "ACGT", 0, 3, 1, 4, 4},
  {"ACGTT", "ACGT", 0, 3, 0, 3, 4},
  {"ACGT", "TACGT", 1, 4, 0, 3, 4},
  {"ACGT", "TTACGT", 2, 5, 0, 3, 4},
  {"ACGT", "ACGTTT", 0, 3, 0, 3, 4},
  {"ACGT", "TACGTT", 1, 4, 0, 3, 4},
  {"ACGT", "TTACGTT", 2, 5, 0, 3, 4},
  {"ACGT", "TACGTTT", 1, 4, 0, 3, 4},
  {"ACGT", "TTACGTTT", 2, 5, 0, 3, 4},
  {"ACGT", "TTACGTTT", 2, 5, 0, 3, 4},
  {"AAAATTTTCCCCGGGG", "AAAATTTTCCCCGGGG", 0, 15, 0, 15, 16},
  {"AAAATTTTCCCCGGGG", "AAAATTTTACCCGGGG", 0, 15, 0, 15, 14},
  {"AAAATTTTCCCCGGGG", "AAAATTTTACCCCGGGG", 0, 16, 0, 15, 15},
  {"AAAATTTTCCCCGGGG", "AAAATTTCCCCGGGG", 0, 14, 0, 15, 14},
  {"AAAATTTTCCCCGGGG", "GCTAGCTAGCTAGCTA", 3, 3, 0, 0, 1},
  {"AAAATTTTCCCCGGGG", "GCTAAAATTTTCCCCGGGG", 3, 18, 0, 15, 16},
  {"AAAATTTTCCCCGGGG", "AAAATTTTCCCCGGGGACT", 0, 15, 0, 15, 16},
};

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
    "ATCTGGCAGGTAAAGATGAGCTCAACAAAGTGATCCAGCATTTTGGCAAAGGAGGCTTTGATGTGATTACTCGCGGTCAGGTGCCACCTAACCCGTCTGA",
    "GATTACGCAAGGCCTGCAAATACGCATCCAGTTGCTGGCTCTCTTTTTCCGCCAGCTCTGAGCGTAAGCGCGCTAATTCCTGGCGGGTATTGGGAGCACG",
    "CCCCGCACCCGCAAGCCGCCGAGAAAAAAAGGATGAGGGCGATACGGATCAGGATATCTACGGTTTTCTGCCCCGCGCCGTTTTGCAGCCAGTTCCAGAA",
    "AATAATAATAATGTCGCAGTCGTCTTCCATGTCATGCCCCAGATATCCAGAACACAACACCCTAACATAGCGTTACTTAAGGGAAATTGACCGCCGAACA",
    "CGTGCTCCCAATACCCGCCAGGAATTAGCGCGCTTACGCTCAGAGCTGGCGGAAAAAGAGAGCCAGCAACTGGATGCGTATTTGCAGGCCTTGCGTAATC",
    "CCCCGCACCCGCAAGCCGCCGAGAAAAAAAGGATGAGGGCGATACGGATCAGGATATCTACGGTTTTCTGCCCCGCGCCGTTTTGCAGCCAGTTCCAGAA",
    "AATAATAATAATGTCGCAGTCGTCTTCCATGTCATGCCCCAGATATCCAGAACACAACACCCTAACATAGCGTTACTTAAGGGAAATTGACCGCCGAACA",
    "ATATCATCACTCCGATGGACGTTTCGATTATCGGCTGCGTGGGGAATGGCCCAGGTGAGGCGCTGGTTTCTACACTCGGCGTCACCGGCGGGACGAAACA",
    "CCGGCGGTGGGCGCGTCCGCCAGTGCCGGCGCGAGCAGGACGGCGTAGAGCCGGATGACGTGATCCTGCCGCCGGAGAAGGCAAAAGCGAAAGGGCTGAC",
    "GATTACGCAAGGCCTGCAAATACGCATCCAGTTGCTGGCTCTCTTTTTCCGCCAGCTCTGAGCGTAAGCGCGCTAATTCCTGGCGGGTATTGGGAGCACG",
    "TCCGGCTGGCAGAACTTGACCAGTGCCGATAAAGAAAAGATGTGGCAGGCGCAGGGGCGAATGCTCACCGCACAAAGCCTGAAGATTAATGCGTTGCTGC",
    "CCGCCCCCGCGCACACGGTGCGGCCTGTCCCGCGTATACTCGACCAGCGGCGTCCCGCCCAGCTTCATTCCCGCCAGGTAACCGCTGCCATACGTCAGCC",
    "AAACCATTGCGAGGAAGTGGTTCTACTTGCTGTCGCCGCGGGAGAACAGGTTGTGCCTGCCTCTGAACTTGCTGCCGCCATGAAGCAGATTAAAGAACTC",
    "CCTTCCCCCTAACTTTCCGCCCGCCATGAAGCAGATAAAAGAACTCCAGCGCCTGCTCGGAAAGAAAACGATGGAAAATGAACTCCTCAAAGAAGCCGTT",
    "AGATGTGCCGGTCATTAAGCATAAAGCCGATGGTTTCTCCCCGCACTTGCCGCCAGTGACGCCACGGCCAGTCAGAGAAGATCATAACAACCGCTCCAGT",
    "CATCGCCCGATTTTCACGTTCGAGAGCGGCGGAGCGGATCGCTCCTTGTTCTTTTTGCCAGGCCCGTAGTTCTTCACCCGTTTTGAATTCGGGTTTGTAT",
    "GCCAGGCAAAATCGGCGTTTCTGGCGGCGATGAGCCATGAGATCCGCACACCGCTGTACGGTATTCTCGGCACTGCTCACTTGATGGCAGATAACGCGCC",
  };
  vector<string> refs = {
    "AATGAGAATGATGTCNTTCNAAATTTGACCAGTCAAACCGCGGGCAATAAGGTCTTCGTTCAGGGCATAGACCTTAATGGGGGCATTACGCAGACTTTCA",
    "ATCTGGCAGGTAAAGATGAGCTCAACAAAGTGATCCAGCATTTTGGCAAAGGAGGCTTTGATGTGATTACTCGCGGTCAGGTGCCACCTAANNCGTCTGA",
    "GATTACGCAAGGCCTGCAAATACGCATCCAGTTGCTGGCTCTCTTTTTCCGCCAGCTCTGAGCGTAAGCGCGCTAATTCCTGGCGGTTATTGGCAGACAG",
    "GCACCGTCCAGCCAACCGCCGAGAAGAAAAGAATGAGTGCGATACGGATCAGGATATCTACGGTTTTCTGCCCCGCGCCGTTTTGCAGCCAGTTCCAGAA",
    "AATAATAATAATGTCGCAGTCGTCTTCCATGTCATGCCCCAGATATCCAGAACACAACACCCTAACATAGCGTTACTTAAGGGAAATTGACCGCCGACAC",
    "CTGTCTGCCAATAACCGCCAGGAATTAGCGCGCTTACGCTCAGAGCTGGCGGAAAAAGAGAGCCAGCAACTGGATGCGTATTTGCAGGCCTTGCGTAATC",
    "GCACCGTCCAGCCAACCGCCGAGAAGAAAAGAATGAGTGCGATACGGATCAGGATATCTACGGTTTTCTGCCCCGCGCCGTTTTGCAGCCAGTTCCAGAA",
    "AATAATAATAATGTCGCAGTCGTCTTCCATGTCATGCCCCAGATATCCAGAACACAACACCCTAACATAGCGTTACTTAAGGGAAATTGACCGCCGACAC",
    "TGGTTTCTACACTCGGCGTCACCGGCGGCAACAAGAA",
    "AGCGCCGGGCGCGCTTCCGCCAGTGCCTGCGCGAGCAGGACGGCGTAGAGCCGGATGACGTGATCCTGCCGCCGGAGAAGGCAAAAGCGAAAGGGCTGAC",
    "GATTACGCAAGGCCTGCAAATACGCATCCAGTTGCTGGCTCTCTTTTTCCGCCAGCTCTGAGCGTAAGCGCGCTAATTCCTGGCGGTTATTGGCAGACAG",
    "TTCGCCGCGCAGAACCTGACCAGTGCCGATAACGAAAAGATGTGGCAGGCGCAGGGGCGAATGCTCACCGCACAAAGCCTGAAGATTAATGCGTTGCTGC",
    "CTGCGCACCGTCTCACGGTGCAGCCTGTCCCGCGTATACTCGACCAGCGGCGTCCCGCCCAGCTTCATTCCCGCCAGGTAACCGCTGCCATACGTCAGC",
    "TAAGCAATACCAGGAAGGAAGTCTTACTGCTGTCGCCGCCGGAGAACAGGTTGTTCCTGCCTCTGAACTTGCTGCCGCCATGAAGCAGATTAAAGAACTC",
    "TCCTGCCTCTGAACTTGCTGCCGCCATGAAGCAGATTAAAGAACTCCAGCGCCTGCTCGGCAAGAAAACGATGGAAAATGAACTCCTCAAAGAAGCCGTT",
    "TGAGTTGCTCGTCATTAAGACGTAAGGCGATGGTTTCTCCCCGCACTTGCCGCCAGTG",
    "CATCGCCCGATTTTCACGTTCGAGAGCGGCGGAGCGGATCGCTCCTTGTTCTTTTTGCCAGGCCAGTAGTTCTTCACCCGTTTTGAATGCGGGTTTGATA",
    "GCCAGGCAAAATCGGCGTTTCTGGCGGCGATGAGCCATGAGATCCGCACACCGCTGTACGGTATTCTCGGCACTGCTCAACTGCTGGCAGATAACCCCGC",
  };
  AlnScoring aln_scoring = {.match = ALN_MATCH_SCORE,
                            .mismatch = ALN_MISMATCH_COST,
                            .gap_opening = ALN_GAP_OPENING_COST,
                            .gap_extending = ALN_GAP_EXTENDING_COST,
                            .ambiguity = ALN_AMBIGUITY_COST};

  Aligner cpu_aligner(aln_scoring.match, aln_scoring.mismatch, aln_scoring.gap_opening, aln_scoring.gap_extending,
                            aln_scoring.ambiguity);
  Filter ssw_filter(true, false, 0, 32767);

  int numWorkers = 10;
  int numCmps = 10;
  int strlen = 120;
  int bufsize = 1000;
  auto driver = ipu::batchaffine::SWAlgorithm({
    .gapInit = -(ALN_GAP_OPENING_COST-ALN_GAP_EXTENDING_COST),
    .gapExtend = -ALN_GAP_EXTENDING_COST,
    .matchValue = ALN_MATCH_SCORE,
    .mismatchValue = -ALN_MISMATCH_COST,
    .ambiguityValue = -ALN_AMBIGUITY_COST,
    .similarity = swatlib::Similarity::nucleicAcid,
    .datatype = swatlib::DataType::nucleicAcid,
  }, {numWorkers, strlen, numCmps, bufsize, ipu::batchaffine::VertexType::assembly, ipu::batchaffine::partition::Algorithm::greedy});

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

TEST(IPUDev, MultiVertexSeparate) {
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
  }, {numWorkers, strlen, numCmps, bufsize, ipu::batchaffine::VertexType::multiasm});

  vector<string> queries, refs;
  for (int i = 0; i < 6; ++i) {
    queries.push_back("AAAAAA");
    refs.push_back("AAAAAA");
  }
  refs[1] = "TTAAAA";
  refs[4] = "TTTTTT";
  driver.compare_local(queries, refs);
  auto aln_results = driver.get_result();
  std::cout << "Aln results:\n";
  std::cout << swatlib::printVector(aln_results.scores) << "\n";
}

class SimpleCorrectnessTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::transform(testData.begin(), 
                   testData.end(), 
                   std::back_inserter(queries), 
                  [](const SequenceTestData& t) { return t.query; });
    std::transform(testData.begin(), 
                   testData.end(), 
                   std::back_inserter(refs), 
                  [](const SequenceTestData& t) { return t.ref; });
  }

  void checkResults(ipu::batchaffine::BlockAlignmentResults results) {
    for (int i = 0; i < testData.size(); ++i) {
      const auto& [ref, query, qstart, qend, rstart, rend, score] = testData[i];

      auto uda = results.a_range_result[i];
      int16_t query_begin = uda & 0xffff;
      int16_t query_end = uda >> 16;

      auto udb = results.b_range_result[i];
      int16_t ref_begin = udb & 0xffff;
      int16_t ref_end = udb >> 16;

      EXPECT_EQ(results.scores[i], score) << i << ": Alignment score does not match expected value";
      EXPECT_EQ(ref_begin, rstart) << i << ": Alignment reference start does not match expected value";
      EXPECT_EQ(ref_end, rend) << i << ": Alignment reference end does not match expected value";
      EXPECT_EQ(query_begin, qstart) << i << ": Alignment query start does not match expected value";
      EXPECT_EQ(query_end, qend) << i << ": Alignment query end does not match expected value";
    }
  }

  const vector<SequenceTestData>& testData = simpleData;
  vector<string> queries;
  vector<string> refs;
};

TEST_F(SimpleCorrectnessTest, UseAssemblyVertex) {
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

  driver.compare_local(queries, refs);
  auto aln_results = driver.get_result();
  checkResults(aln_results);
}

TEST_F(SimpleCorrectnessTest, UseCppVertex) {
  auto driver = ipu::batchaffine::SWAlgorithm({}, {
    .tilesUsed = 2,
    .maxAB = 300,
    .maxBatches = 20,
    .bufsize = 3000,
    .vtype = ipu::batchaffine::VertexType::cpp,
    .fillAlgo = ipu::batchaffine::partition::Algorithm::roundRobin
  });

  driver.compare_local(queries, refs);
  auto aln_results = driver.get_result();
  checkResults(aln_results);
}

TEST_F(SimpleCorrectnessTest, UseCppMultiVertex) {
  auto driver = ipu::batchaffine::SWAlgorithm({}, {
    .tilesUsed = 2,
    .maxAB = 300,
    .maxBatches = 20,
    .bufsize = 3000,
    .vtype = ipu::batchaffine::VertexType::multi,
    .fillAlgo = ipu::batchaffine::partition::Algorithm::roundRobin
  });

  driver.compare_local(queries, refs);
  auto aln_results = driver.get_result();
  checkResults(aln_results);
}

TEST_F(SimpleCorrectnessTest, UseAsmMultiVertex) {
  auto driver = ipu::batchaffine::SWAlgorithm({}, {
    .tilesUsed = 2,
    .maxAB = 300,
    .maxBatches = 20,
    .bufsize = 3000,
    .vtype = ipu::batchaffine::VertexType::multiasm,
    .fillAlgo = ipu::batchaffine::partition::Algorithm::roundRobin
  });

  driver.compare_local(queries, refs);
  auto aln_results = driver.get_result();
  checkResults(aln_results);
}
#endif