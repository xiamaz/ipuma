#include <plog/Log.h>
#undef LOG

#include "ssw.hpp"
#include "klign.hpp"
#include "kmer.hpp"
#include "aligner_cpu.hpp"
#include "driver.hpp"
#include "ipu_batch_affine.h"
#include "ipu_base.h"
#include "ipu_batch_affine.h"
#include <atomic>
#include <thread>
#include <queue>
#include <tuple>

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

upcxx::global_ptr<int> g_mapping;
upcxx::global_ptr<int32_t> g_input;
upcxx::global_ptr<int32_t> g_output;
const bool do_test = false;

std::tuple<int16_t, int16_t> convertPackedRange(int32_t packed) {
  int16_t begin = packed & 0xffff;
  int16_t end = packed >> 16;
  return {begin, end};
}

inline void convertIpuToAln(shared_ptr<AlignBlockData> aln_block_data, Aln &aln, int32_t score, int32_t a_range, int32_t b_range) {
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
  aln.identity = 100 * aln.score1 / aln_block_data->aln_scoring.match / aln.rlen;
  aln.read_group_id = aln_block_data->read_group_id;
}

void insert_ipu_result_block(shared_ptr<AlignBlockData> aln_block_data, std::vector<int32_t> &a_range,
                             std::vector<int32_t> &b_range, std::vector<int32_t> &scores) {
  for (int i = 0; i < aln_block_data->kernel_alns.size(); i++) {
    Aln &aln = aln_block_data->kernel_alns[i];
    convertIpuToAln(aln_block_data, aln, scores[i], a_range[i], b_range[i]);
    if (do_test && aln.identity < 0) {
      PLOGW.printf("[%d]\tnegative identity\n", rank_me());
      PLOGW.printf("[%d]\tA/B %s / %s\n", rank_me(), aln_block_data->read_seqs[i].c_str(), aln_block_data->ctg_seqs[i].c_str());
      PLOGW.printf("[%d]\t%d\n", rank_me(), 100 * aln.score1);
      PLOGW.printf("[%d]\t%d\n", rank_me(), aln_block_data->aln_scoring.match);
      PLOGW.printf("[%d]\t%d\n", rank_me(), aln.rlen);
      exit(1);
    }
    aln_block_data->alns->add_aln(aln);
  }
}

bool done_upload = true;

upcxx::future<> uprunner;
bool done_uprunner;
upcxx::persona progress_persona;
std::queue<std::tuple<int32_t *, int32_t *, int, int>> backlog;

upcxx::persona resume_persona;
bool got_result = true;

upcxx::future<> ipu_align_block(shared_ptr<AlignBlockData> aln_block_data, Alns *alns, IntermittentTimer &aln_kernel_timer) {
  auto algoconfig = ALGO_CONFIGURATION;
  auto swconfig = SW_CONFIGURATION;

  int maxb = ipu::batchaffine::SWAlgorithm::prepare_remote(swconfig, algoconfig, aln_block_data->ctg_seqs, aln_block_data->read_seqs,
                                                &g_input.local()[0], &g_input.local()[algoconfig.getInputBufferSize32b()],
                                                &g_mapping.local()[0]);

  const int ipu_executor_rank = (local_team().rank_me()) % KLIGN_IPUS_LOCAL;
  rpc(
      ipu_executor_rank,
      [](global_ptr<int32_t> in, global_ptr<int32_t> out, int src, int max_bucket) {
        // PLOGD.printf("lpc backlog");
        return progress_persona.lpc([in = in.local(), out = out.local(), src = src, mb=max_bucket] {
          backlog.push({in, out, src, mb});
          // PLOGD.printf("Backlog pushed");
        });
      },
      g_input, g_output, rank_me(), maxb)
      .wait();
  got_result = false;
  auto fut = upcxx_utils::execute_in_thread_pool([]() {
    upcxx::persona_scope scope(resume_persona);
    PLOGD.printf("Wait for completion of execution semaphore locally");
    while (!got_result) upcxx::progress();
  });
  fut = fut.then([alns = alns, aln_block_data, mapping=g_mapping.local()]() {
    PLOGD.printf("merge data on %d from RPC on %d", rank_me(), local_team().rank_me() % KLIGN_IPUS_LOCAL);
    auto algoconfig = ALGO_CONFIGURATION;
    const auto totalComparisonsCount = algoconfig.getTotalNumberOfComparisons();

    vector<int32_t> ars(algoconfig.getTotalNumberOfComparisons());
    vector<int32_t> brs(algoconfig.getTotalNumberOfComparisons());
    vector<int32_t> scs(algoconfig.getTotalNumberOfComparisons());

    size_t a_range_offset = scs.size();
    size_t b_range_offset = a_range_offset + ars.size();

    for (size_t i = 0; i < aln_block_data->ctg_seqs.size(); ++i) {
      size_t mapped_i = mapping[i];
      scs[i] = g_output.local()[mapped_i];
      ars[i] = g_output.local()[a_range_offset + mapped_i];
      brs[i] = g_output.local()[b_range_offset + mapped_i];
    }
    insert_ipu_result_block(aln_block_data, ars, brs, scs);
  });
  fut = fut.then([alns = alns, aln_block_data]() {
    PLOGD << "appending and returning " << aln_block_data->alns->size();
    alns->append(*(aln_block_data->alns));
  });
  return fut;
}

void init_aligner(AlnScoring &aln_scoring, int rlen_limit, IntermittentTimer &aln_kernel_timer) {
  done_uprunner = false;
  if (rank_me() < KLIGN_IPUS_LOCAL) {
    uprunner = upcxx_utils::execute_in_new_thread([&aln_kernel_timer]() {
      // push progress_persona onto this thread's persona stack
      upcxx::persona_scope scope(progress_persona);
      // progress thread drains progress until work is done
      PLOGD.printf("Start Executer");
      while (true) {
        if (backlog.empty() && done_uprunner) {
          PLOGD.printf("Exit Uploader thread");
          break;
        }
        if (!backlog.empty()) {
          auto [in, out, src, mb] = backlog.front();
          auto driver = getDriver();
          if (!driver->slot_available(mb)) {
            sched_yield();
            upcxx::progress();
            continue;
          }
          backlog.pop();
          // PLOGD.printf("Going to aquire slot");
          int slot = driver->queue_slot(mb);
          // PLOGD.printf("Going to upload on slot%d", slot);
          int *inptr = (int *)in;
          PLOGD.printf("Upload+Execute data from rank(%d)", src);
          // PLOGD.printf("Call uploading slot %d", slot);
          getDriver()->upload(&inptr[0], &inptr[getDriver()->algoconfig.getInputBufferSize32b()], slot);
          // PLOGD.printf("Called uploading slot %d", slot);
          aln_kernel_timer.start();
          driver->prepared_remote_compare(&in[0], &in[driver->algoconfig.getInputBufferSize32b()], &out[0],
                                          &out[driver->algoconfig.getTotalNumberOfComparisons() * 3], slot);
          aln_kernel_timer.stop();
          PLOGD.printf("Execution done");
          rpc_ff(src, []() { resume_persona.lpc_ff([]() { got_result = true; }); });
        }
        sched_yield();
        upcxx::progress();
      }
    });
  }
  SWARN("Initialize global array\n");
  auto algoconfig = ALGO_CONFIGURATION;
  size_t inputs_size = algoconfig.getInputBufferSize32b();
  size_t results_size = algoconfig.getTotalNumberOfComparisons() * 3;
  g_input = new_array<int32_t>(inputs_size);
  g_output = new_array<int32_t>(results_size);
  g_mapping = new_array<int>(algoconfig.getTotalNumberOfComparisons());
}

void cleanup_aligner() {
  barrier();
  if (rank_me() < KLIGN_IPUS_LOCAL) {
    progress_persona.lpc([]() { done_uprunner = true; }).wait();
    uprunner.wait();
  }
  SWARN("Delete global array\n");
  delete_array(g_input);
  delete_array(g_output);
  delete_array(g_mapping);
}

shared_ptr<AlignBlockData> copyAlignBlock(shared_ptr<AlignBlockData> al) {
  vector<Aln> kernel_alns(al->kernel_alns);
  vector<string> ctg_seqs(al->ctg_seqs);
  vector<string> read_seqs(al->read_seqs);
  shared_ptr<AlignBlockData> al_copy =
      make_shared<AlignBlockData>(kernel_alns, ctg_seqs, read_seqs, al->max_clen, al->max_rlen, al->read_group_id, al->aln_scoring);
  return al_copy;
}

void validate_align_block(future<> &fut, CPUAligner &cpu_aligner, shared_ptr<AlignBlockData> al_copy, vector<Aln> &alns_ipu) {
  auto alns = new Alns();
  fut = cpu_aligner.ssw_align_block(al_copy, alns);
  fut.wait();

  int mismatches = 0;
  for (int i = 0; i < alns->size(); i++) {
    // Same score
    // Same a,b pair
    const auto &aln_ipu = alns_ipu[i];
    const auto &aln_cpu = alns->get_aln(i);
    assert(aln_cpu.rlen == aln_ipu.rlen);
    assert(aln_cpu.clen == aln_ipu.clen);

    if (aln_cpu.score1 != aln_ipu.score1) {
      PLOGW.printf("\tmismatch want %d score %d got %d", i, aln_cpu.score1, aln_ipu.score1);
    }

    if (aln_cpu.cstart != aln_ipu.cstart || aln_cpu.cstop != aln_ipu.cstop || aln_cpu.rstart != aln_ipu.rstart ||
        aln_cpu.rstop != aln_ipu.rstop || aln_cpu.identity != aln_ipu.identity) {
      PLOGW.printf("mismatch A/B: %s / %s", al_copy->read_seqs[i].c_str(), al_copy->ctg_seqs[i].c_str());
      if (aln_cpu.cstart != aln_ipu.cstart) {
        PLOGW.printf("\tmismatch want %d cstart %d got %d", i, aln_cpu.cstart, aln_ipu.cstart);
      }
      exit(1);
    }
  }
  if (mismatches) {
    int matches = alns->size() - mismatches;
    PLOGW << "Total number of mismatches/matches: " << mismatches << "/" << matches;
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
    shared_ptr<AlignBlockData> aln_block_data =
        make_shared<AlignBlockData>(kernel_alns, ctg_seqs, read_seqs, max_clen, max_rlen, read_group_id, cpu_aligner.aln_scoring);
    if (do_test) {
      PLOGW << "Do parity check";
      auto aln_block_copy = copyAlignBlock(aln_block_data);
      active_kernel_fut = ipu_align_block(aln_block_data, alns, aln_kernel_timer);
      active_kernel_fut.wait();
      validate_align_block(active_kernel_fut, cpu_aligner, aln_block_copy, aln_block_data->kernel_alns);
    } else {
      active_kernel_fut = ipu_align_block(aln_block_data, alns, aln_kernel_timer);
    }
  }
}