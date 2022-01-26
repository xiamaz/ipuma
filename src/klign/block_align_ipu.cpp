#include "ssw.hpp"
#include "klign.hpp"
#include "kmer.hpp"
#include "aligner_cpu.hpp"
#include "ipuma-sw/driver.hpp"
#include "ipuma-sw/ipu_batch_affine.h"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

// upcxx::global_ptr<char> a;
// upcxx::global_ptr<char> b;
// upcxx::global_ptr<int32_t> a_len;
// upcxx::global_ptr<int32_t> b_len;

// upcxx::global_ptr<int32_t> scores;
// upcxx::global_ptr<int32_t> mismatches;
// upcxx::global_ptr<int32_t> a_range_result;
// upcxx::global_ptr<int32_t> b_range_result;

ipu::batchaffine::IPUAlgoConfig algoconfig = {
    KLIGN_IPU_TILES, KLIGN_IPU_MAXAB_SIZE, KLIGN_IPU_MAX_BATCHES, KLIGN_IPU_BUFSIZE, ipu::batchaffine::VertexType::assembly,
};

//   void checkResults(const vector<Alignment>& alns_ipu) {
//     AlnScoring aln_scoring = {.match = ALN_MATCH_SCORE,
//                             .mismatch = ALN_MISMATCH_COST,
//                             .gap_opening = ALN_GAP_OPENING_COST,
//                             .gap_extending = ALN_GAP_EXTENDING_COST,
//                             .ambiguity = ALN_AMBIGUITY_COST};

//     vector<Alignment> alns(queries.size());
//     for (int i = 0; i < queries.size(); ++i) {
//       auto reflen = refs[i].size();
//       auto qlen = queries[i].size();
//       auto masklen = max((int)min(reflen, qlen) / 2, 15);
//       Aligner cpu_aligner(aln_scoring.match, aln_scoring.mismatch, aln_scoring.gap_opening, aln_scoring.gap_extending,
//                           aln_scoring.ambiguity);
//       Filter ssw_filter(true, false, 0, 32767);
//       cpu_aligner.Align(queries[i].c_str(), refs[i].c_str(), reflen, ssw_filter, &alns[i], masklen);

//       EXPECT_EQ(alns[i].sw_score, alns_ipu[i].sw_score) << i << ": IPU score result does not match CPU SSW";
//       EXPECT_EQ(alns[i].ref_begin, alns_ipu[i].ref_begin) << i << ": IPU reference start result does not match CPU SSW";
//       EXPECT_EQ(alns[i].ref_end, alns_ipu[i].ref_end) << i << ": IPU reference end result does not match CPU SSW";
//       EXPECT_EQ(alns[i].query_begin, alns_ipu[i].query_begin) << i << ": IPU query start result does not match CPU SSW";
//       EXPECT_EQ(alns[i].query_end, alns_ipu[i].query_end) << i << ": IPU query end result does not match CPU SSW";
//     }
//   }
// };

void insert_ipu_result_block(shared_ptr<AlignBlockData> aln_block_data, Alns *alns, int32_t *a_range, int32_t *b_range,
                             int32_t *scores) {
  for (int i = 0; i < aln_block_data->kernel_alns.size(); i++) {
    auto uda = a_range[i];
    int16_t contig_begin = uda & 0xffff;
    int16_t contig_end = uda >> 16;

    auto udb = b_range[i];
    int16_t read_begin = udb & 0xffff;
    int16_t read_end = udb >> 16;

    Aln &aln = aln_block_data->kernel_alns[i];
    aln.rstop = aln.rstart + read_end + 1;
    aln.rstart += read_begin;
    aln.cstop = aln.cstart + contig_end + 1;
    aln.cstart += contig_begin;
    if (aln.orient == '-') switch_orient(aln.rstart, aln.rstop, aln.rlen);
    aln.score1 = scores[i];
    // FIXME: needs to be set to the second best
    aln.score2 = 0;
    // aln.mismatches = aln_results.mismatches[i];  // ssw_aln.mismatches;
    aln.mismatches = 0;
    aln.identity = 100 * aln.score1 / aln_block_data->aln_scoring.match / aln.rlen;
    // aln.identity = (unsigned)100 * (unsigned)ssw_aln.sw_score / (unsigned)aln_scoring.match / (unsigned)aln.rlen;
    if (aln.identity < 0) {
      printf("[%d]\tnegative identity\n", rank_me());
      printf("[%d]\tA/B %s / %s\n",rank_me(), aln_block_data->read_seqs[i].c_str(), aln_block_data->ctg_seqs[i].c_str());
      printf("[%d]\t%d\n",rank_me(), 100 * aln.score1);
      printf("[%d]\t%d\n", rank_me(), aln_block_data->aln_scoring.match);
      printf("[%d]\t%d\n", rank_me(), aln.rlen);
      // std::ofstream oa("./dump_a.txt");
      // std::ofstream oa_len("./dump_a_len.txt");
      // std::ofstream ob("./dump_b.txt");
      // std::ofstream ob_len("./dump_b_len.txt");
      // for (size_t i = 0; i < aln_block_data->read_seqs.size(); i++) {
      //   oa_len << aln_block_data->read_seqs[i].size() << std::endl;
      //   ob_len << aln_block_data->ctg_seqs[i].size() << std::endl;
      // }
      // for (size_t i = 0; i < aln_block_data->read_seqs.size(); i++) {
      //   oa << aln_block_data->read_seqs[i] << std::endl;
      //   ob << aln_block_data->ctg_seqs[i] << std::endl;
      // }
      // oa_len.close();
      // ob_len.close();
      // oa.close();
      // ob.close();
      exit(1);
    }
    aln.read_group_id = aln_block_data->read_group_id;
    // FIXME: need to get cigar
    // if (report_cigar) set_sam_string(aln, "*", "*");  // FIXME until there is a valid:ssw_aln.cigar_string);
    aln_block_data->alns->add_aln(aln);
  }
  alns->append(*(aln_block_data->alns));
}

upcxx::future<> ipu_align_block(shared_ptr<AlignBlockData> aln_block_data, Alns *alns, CPUAligner &cpu_aligner) {
  // CPUAligner cpu(false);
  // auto fu = cpu.ssw_align_block(aln_block_data, alns);
  // fu.wait();
  auto [sc, mis, ar, br] = rpc(
                               local_team(), local_team().rank_me() % KLIGN_IPUS_LOCAL,
                               [](int sender_rank, vector<string> A, vector<string> B) {
                                 cout << "Launching rpc on " << rank_me() << " from " << sender_rank << endl;
                                 getDriver()->compare_local(A, B);
                                 auto res = getDriver()->get_result();
                                 vector<int32_t> sc(res.scores);
                                 vector<int32_t> mis(res.mismatches);
                                 vector<int32_t> ar(res.a_range_result);
                                 vector<int32_t> br(res.b_range_result);
                                 cout << "Exiting rpc on " << rank_me() << " from " << sender_rank << endl;
                                 return make_tuple(sc, mis, ar, br);
                               },
                               local_team().rank_me(), aln_block_data->ctg_seqs, aln_block_data->read_seqs)
                               .wait();
  // cout << sc.size() << " " << endl;
  insert_ipu_result_block(aln_block_data, alns, ar.data(), br.data(), sc.data());

  return execute_in_thread_pool([]() { });
  // auto [scores, mismatches, a_range, b_range] = fuu2.wait();
  // return rpc(local_team(), local_team().rank_me() % KLIGN_IPUS_LOCAL, [](){ cout << "done" << endl; });
}

void init_aligner(AlnScoring &aln_scoring, int rlen_limit) {
  // if (a.is_null())  {
  //   SLOG_VERBOSE("Initialize global array s\n");
  //   a = new_array<char>(algoconfig.getTotalBufferSize());
  //   b = new_array<char>(algoconfig.getTotalBufferSize());
  //   a_len = new_array<int32_t>(algoconfig.getTotalNumberOfComparisons());
  //   b_len = new_array<int32_t>(algoconfig.getTotalNumberOfComparisons());

  //   scores = new_array<int32_t>(algoconfig.getTotalNumberOfComparisons());
  //   mismatches = new_array<int32_t>(algoconfig.getTotalNumberOfComparisons());
  //   a_range_result = new_array<int32_t>(algoconfig.getTotalNumberOfComparisons());
  //   b_range_result = new_array<int32_t>(algoconfig.getTotalNumberOfComparisons());
  // }
  // double init_time;
  // SLOG_VERBOSE("Initialized ipuma driver in ", init_time, " s\n");
}

void cleanup_aligner() {
  // delete ipu_driver;
}
void kernel_align_block(CPUAligner &cpu_aligner, vector<Aln> &kernel_alns, vector<string> &ctg_seqs, vector<string> &read_seqs,
                        Alns *alns, future<> &active_kernel_fut, int read_group_id, int max_clen, int max_rlen,
                        IntermittentTimer &aln_kernel_timer) {
  if (!kernel_alns.empty()) {
    active_kernel_fut.wait();  // should be ready already

    vector<Aln> kernel_alns_copy(kernel_alns);
    vector<string> ctg_seqs_copy(ctg_seqs);
    vector<string> read_seqs_copy(read_seqs);

    vector<Aln> kernel_alns_2(kernel_alns_copy);
    vector<string> ctg_seqs_2(ctg_seqs_copy);
    vector<string> read_seqs_2(read_seqs_copy);

    vector<Aln> kernel_alns_3(kernel_alns_copy);
    vector<string> ctg_seqs_3(ctg_seqs_copy);
    vector<string> read_seqs_3(read_seqs_copy);

    auto N = kernel_alns.size();
    // Normal
    shared_ptr<AlignBlockData> aln_block_data = make_shared<AlignBlockData>(kernel_alns, ctg_seqs, read_seqs, max_clen, max_rlen, read_group_id, cpu_aligner.aln_scoring);
    assert(kernel_alns.empty());
    assert(max_clen != 0);
    active_kernel_fut = cpu_aligner.ssw_align_block(aln_block_data, alns);
    active_kernel_fut.wait();

    shared_ptr<AlignBlockData> aln_block_data2 = make_shared<AlignBlockData>(kernel_alns_2, ctg_seqs_2, read_seqs_2, max_clen, max_rlen, read_group_id, cpu_aligner.aln_scoring);
    shared_ptr<AlignBlockData> aln_block_data3 = make_shared<AlignBlockData>(kernel_alns_3, ctg_seqs_3, read_seqs_3, max_clen, max_rlen, read_group_id, cpu_aligner.aln_scoring);
    for (int i = 0; i < aln_block_data->kernel_alns.size(); i++) {
        // if (aln_block_data2->alns.get)) 
        auto& x1 = aln_block_data2->kernel_alns[i];
        auto& x2 = aln_block_data3->kernel_alns[i];
        if (x1.rstart != x2.rstart || x1.cstart != x2.cstart || x1.cstop != x2.cstop) {
          cout << "prelim is false" << endl;
          exit(1);
        }
    }

    // Hahaha
    auto aa_CPU = new Alns();
    active_kernel_fut = cpu_aligner.ssw_align_block(aln_block_data2, aa_CPU);
    active_kernel_fut.wait();

    auto aa_IPU = new Alns();
    active_kernel_fut = ipu_align_block(aln_block_data3, aa_IPU, cpu_aligner);
    active_kernel_fut.wait();

    assert(aa_CPU->size() == N);
    assert(aa_IPU->size() == N);
    // if (rank_me() == 70) 
    for (int i = 0; i < aa_CPU->size(); i++) {
      // Same score
      assert(aa_CPU->get_aln(i).score1 == aa_IPU->get_aln(i).score1);
      // Same a,b pair
      assert(aa_CPU->get_aln(i).rlen == aa_IPU->get_aln(i).rlen);
      assert(aa_CPU->get_aln(i).clen == aa_IPU->get_aln(i).clen);

      if (aa_CPU->get_aln(i).cstart != aa_IPU->get_aln(i).cstart || aa_CPU->get_aln(i).cstop != aa_IPU->get_aln(i).cstop ||
          aa_CPU->get_aln(i).rstart != aa_IPU->get_aln(i).rstart || aa_CPU->get_aln(i).rstop != aa_IPU->get_aln(i).rstop) {
        printf("mismatch A/B: %s / %s\n", read_seqs_copy[i].c_str(), ctg_seqs_copy[i].c_str());
        if (aa_CPU->get_aln(i).cstart != aa_IPU->get_aln(i).cstart) {
          printf("\tmismatch want %d cstart %d got %d\n", i, aa_CPU->get_aln(i).cstart, aa_IPU->get_aln(i).cstart);
        }
        if (aa_CPU->get_aln(i).cstop != aa_IPU->get_aln(i).cstop) {
          printf("\tmismatch want %d cstop %d got %d\n", i, aa_CPU->get_aln(i).cstop, aa_IPU->get_aln(i).cstop);
        }
        if (aa_CPU->get_aln(i).rstart != aa_IPU->get_aln(i).rstart) {
          printf("\tmismatch want %d rstart %d got %d\n", i, aa_CPU->get_aln(i).rstart, aa_IPU->get_aln(i).rstart);
        }
        if (aa_CPU->get_aln(i).rstop != aa_IPU->get_aln(i).rstop) {
          printf("\tmismatch want %d rstop %d got %d\n", i, aa_CPU->get_aln(i).rstop, aa_IPU->get_aln(i).rstop);
        }
        if (aa_CPU->get_aln(i).identity != aa_IPU->get_aln(i).identity) {
          printf("\tmismatch want %d identity %d got %d\n", i, aa_CPU->get_aln(i).identity, aa_IPU->get_aln(i).identity);
        }
      }
    }
  }
}

// void kernel_align_block(CPUAligner &cpu_aligner, vector<Aln> &kernel_alns, vector<string> &ctg_seqs, vector<string> &read_seqs,
//                         Alns *alns, future<> &active_kernel_fut, int read_group_id, int max_clen, int max_rlen,
//                         IntermittentTimer &aln_kernel_timer) {
//   while (!active_kernel_fut.ready() || kernel_alns.empty()) {
//     progress();
//   }
//   if (!kernel_alns.empty()) {
//     assert(active_kernel_fut.ready() && "active_kernel_fut should already be ready");
//     active_kernel_fut.wait();  // should be ready already
//     shared_ptr<AlignBlockData> aln_block_data =
//         make_shared<AlignBlockData>(kernel_alns, ctg_seqs, read_seqs, max_clen, max_rlen, read_group_id,
//         cpu_aligner.aln_scoring);

//     if (cpu_aligner.ssw_filter.report_cigar) {
//       SWARN("Not implemented cigar for IPUs");
//       exit(1);
//     }

//     assert(kernel_alns.empty());
//     active_kernel_fut = ipu_align_block(aln_block_data, alns, cpu_aligner.ssw_filter.report_cigar, aln_kernel_timer);
//     // // for now, the GPU alignment doesn't support cigars
//     // if (cpu_aligner.ssw_filter.report_cigar) {
//     //   SWARN("Not implemented cigar for IPUs");
//     //   exit(1);
//     //   // active_kernel_fut = gpu_align_block(aln_block_data, alns, cpu_aligner.ssw_filter.report_cigar, aln_kernel_timer);
//     // } else {
//     //   active_kernel_fut = cpu_aligner.ssw_align_block(aln_block_data, alns);
//     // }
//   } else {
//     SWARN("Block shall not be empty");
//     exit(1);
//   }
// }}