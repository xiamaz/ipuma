#include <plog/Log.h>

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

std::tuple<int16_t, int16_t> convertPackedRange(int32_t packed) {
    int16_t begin = packed & 0xffff;
    int16_t end = packed >> 16;
    return {begin, end};
}

int convertIpuToAln(Aln& aln, int32_t score, int32_t a_range, int32_t b_range) {
    auto [contig_begin, contig_end] = convertPackedRange(a_range);
    auto [read_begin, read_end] = convertPackedRange(b_range);

    aln.rstop = aln.rstart + read_end + 1;
    aln.rstart += read_begin;
    aln.cstop = aln.cstart + contig_end + 1;
    aln.cstart += contig_begin;
    if (aln.orient == '-') switch_orient(aln.rstart, aln.rstop, aln.rlen);
    aln.score1 = score;
    // FIXME: needs to be set to the second best
    aln.score2 = 0;
    // aln.mismatches = aln_results.mismatches[i];  // ssw_aln.mismatches;
    aln.mismatches = 0;
    return 0;
}

void insert_ipu_result_block(shared_ptr<AlignBlockData> aln_block_data, Alns *alns, std::vector<int32_t>& a_range, std::vector<int32_t>& b_range,
                             std::vector<int32_t>& scores) {
  for (int i = 0; i < aln_block_data->kernel_alns.size(); i++) {
    Aln &aln = aln_block_data->kernel_alns[i];
    if (convertIpuToAln(aln, scores[i], a_range[i], b_range[i])){
      PLOGW << "Conversion of IPU to Aln failed";
      exit(1);
    }
    aln.identity = 100 * aln.score1 / aln_block_data->aln_scoring.match / aln.rlen;
    // aln.identity = (unsigned)100 * (unsigned)ssw_aln.sw_score / (unsigned)aln_scoring.match / (unsigned)aln.rlen;
    if (aln.identity < 0) {
      PLOGW.printf("[%d]\tnegative identity\n", rank_me());
      PLOGW.printf("[%d]\tA/B %s / %s\n",rank_me(), aln_block_data->read_seqs[i].c_str(), aln_block_data->ctg_seqs[i].c_str());
      PLOGW.printf("[%d]\t%d\n",rank_me(), 100 * aln.score1);
      PLOGW.printf("[%d]\t%d\n", rank_me(), aln_block_data->aln_scoring.match);
      PLOGW.printf("[%d]\t%d\n", rank_me(), aln.rlen);
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

upcxx::future<> ipu_align_block(shared_ptr<AlignBlockData> aln_block_data, Alns *alns) {
  // auto fu = cpu.ssw_align_block(aln_block_data, alns);
  // fu.wait();
  auto [sc, ar, br] = rpc(
                               local_team(), local_team().rank_me() % KLIGN_IPUS_LOCAL,
                               [](int sender_rank, vector<string> A, vector<string> B) {
                                 PLOGD << "Launching rpc on " << rank_me() << " from " << sender_rank;
                                 getDriver()->compare_local(A, B);
                                 auto res = getDriver()->get_result();
                                 vector<int32_t> sc(res.scores);
                                 vector<int32_t> ar(res.a_range_result);
                                 vector<int32_t> br(res.b_range_result);
                                 PLOGD << "Exiting rpc on " << rank_me() << " from " << sender_rank;
                                 return make_tuple(sc, ar, br);
                               },
                               local_team().rank_me(), aln_block_data->ctg_seqs, aln_block_data->read_seqs)
                               .wait();
  insert_ipu_result_block(aln_block_data, alns, ar, br, sc);

  return execute_in_thread_pool([]() { });
  // auto [scores, mismatches, a_range, b_range] = fuu2.wait();
  // return rpc(local_team(), local_team().rank_me() % KLIGN_IPUS_LOCAL, [](){ PLOGD << "done"; });
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

shared_ptr<AlignBlockData> copyAlignBlock(shared_ptr<AlignBlockData> al) {
    vector<Aln> kernel_alns(al->kernel_alns);
    vector<string> ctg_seqs(al->ctg_seqs);
    vector<string> read_seqs(al->read_seqs);
    shared_ptr<AlignBlockData> al_copy = make_shared<AlignBlockData>(
      kernel_alns, ctg_seqs, read_seqs, al->max_clen, al->max_rlen, al->read_group_id, al->aln_scoring
    );
    return al_copy;
}

void validate_align_block(future<>& fut, CPUAligner& cpu_aligner, shared_ptr<AlignBlockData> al_copy, vector<Aln>& alns_ipu) {
    auto alns = new Alns();
    fut = cpu_aligner.ssw_align_block(al_copy, alns);
    fut.wait();

    int mismatches = 0;
    for (int i = 0; i < alns->size(); i++) {
      // Same score
      // Same a,b pair
      const auto& aln_ipu = alns_ipu[i];
      const auto& aln_cpu = alns->get_aln(i);
      assert(aln_cpu.rlen == aln_ipu.rlen);
      assert(aln_cpu.clen == aln_ipu.clen);

      if (aln_cpu.score1 != aln_ipu.score1) {
        PLOGW.printf("\tmismatch want %d score %d got %d\n", i, aln_cpu.score1, aln_ipu.score1);
      }

      if (aln_cpu.cstart != aln_ipu.cstart || aln_cpu.cstop != aln_ipu.cstop ||
          aln_cpu.rstart != aln_ipu.rstart || aln_cpu.rstop != aln_ipu.rstop || 
          aln_cpu.identity != aln_ipu.identity) {
        PLOGW.printf("mismatch A/B: %s / %s\n", al_copy->read_seqs[i].c_str(), al_copy->ctg_seqs[i].c_str());
        if (aln_cpu.cstart != aln_ipu.cstart) {
          PLOGW.printf("\tmismatch want %d cstart %d got %d\n", i, aln_cpu.cstart, aln_ipu.cstart);
        }
        if (aln_cpu.cstop != aln_ipu.cstop) {
          PLOGW.printf("\tmismatch want %d cstop %d got %d\n", i, aln_cpu.cstop, aln_ipu.cstop);
        }
        if (aln_cpu.rstart != aln_ipu.rstart) {
          PLOGW.printf("\tmismatch want %d rstart %d got %d\n", i, aln_cpu.rstart, aln_ipu.rstart);
        }
        if (aln_cpu.rstop != aln_ipu.rstop) {
          PLOGW.printf("\tmismatch want %d rstop %d got %d\n", i, aln_cpu.rstop, aln_ipu.rstop);
        }
        if (aln_cpu.identity != aln_ipu.identity) {
          PLOGW.printf("\tmismatch want %d identity %d got %d\n", i, aln_cpu.identity, aln_ipu.identity);
        }
        mismatches++;
      }
    }
    if (mismatches) {
      exit(1);
    }
}

void kernel_align_block(CPUAligner &cpu_aligner, vector<Aln> &kernel_alns, vector<string> &ctg_seqs, vector<string> &read_seqs,
                        Alns *alns, future<> &active_kernel_fut, int read_group_id, int max_clen, int max_rlen,
                        IntermittentTimer &aln_kernel_timer) {
  if (!kernel_alns.empty()) {
    active_kernel_fut.wait();  // should be ready already

    auto N = kernel_alns.size();
    auto alns_offset = alns->size();

    // Normal
    shared_ptr<AlignBlockData> aln_block_data = make_shared<AlignBlockData>(kernel_alns, ctg_seqs, read_seqs, max_clen, max_rlen, read_group_id, cpu_aligner.aln_scoring);
    auto aln_block_copy = copyAlignBlock(aln_block_data);
    active_kernel_fut = ipu_align_block(aln_block_data, alns);
    active_kernel_fut.wait();

    validate_align_block(active_kernel_fut, cpu_aligner, aln_block_copy, aln_block_data->kernel_alns);
  }
}