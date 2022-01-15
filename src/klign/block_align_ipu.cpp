#include "klign.hpp"
#include "kmer.hpp"
#include "aligner_cpu.hpp"

#include "ipuma-sw/ipu_batch_affine.h"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

static ipu::batchaffine::SWAlgorithm *ipu_driver = nullptr;

static upcxx::future<> ipu_align_block(shared_ptr<AlignBlockData> aln_block_data,
                                       Alns *alns,
                                       bool report_cigar,
                                       IntermittentTimer &aln_kernel_timer) {
  future<> fut = upcxx_utils::execute_in_thread_pool([aln_block_data, report_cigar, &aln_kernel_timer] {
    DBG_VERBOSE("Starting _ipu_align_block_kernel of ", aln_block_data->kernel_alns.size(), "\n");
    aln_kernel_timer.start();
    ipu_driver->compare(aln_block_data->read_seqs, aln_block_data->ctg_seqs);
    auto aln_results = ipu_driver->get_result();
    for (int i = 0; i < aln_block_data->kernel_alns.size(); i++) {
      auto uda = aln_results.a_range_result[i];
      int16_t query_begin = uda & 0xffff;
      int16_t query_end = uda >> 16;

      auto udb = aln_results.b_range_result[i];
      int16_t ref_begin = udb & 0xffff;
      int16_t ref_end = udb >> 16;

      Aln &aln = aln_block_data->kernel_alns[i];
      aln.rstop = aln.rstart + query_end + 1;
      aln.rstart += query_begin;
      aln.cstop = aln.cstart + ref_end + 1;
      aln.cstart += ref_begin;
      if (aln.orient == '-') switch_orient(aln.rstart, aln.rstop, aln.rlen);
      aln.score1 = aln_results.scores[i];
      // FIXME: needs to be set to the second best
      aln.score2 = 0;
      aln.mismatches = aln_results.mismatches[i];  // ssw_aln.mismatches;
      aln.identity = 100 * aln.score1 / aln_block_data->aln_scoring.match / aln.rlen;
      aln.read_group_id = aln_block_data->read_group_id;
      // FIXME: need to get cigar
      if (report_cigar) set_sam_string(aln, "*", "*");  // FIXME until there is a valid:ssw_aln.cigar_string);
      aln_block_data->alns->add_aln(aln);
    }

    aln_kernel_timer.stop();
  });
  fut = fut.then([alns = alns, aln_block_data]() {
    DBG_VERBOSE("appending and returning ", aln_block_data->alns->size(), "\n");
    alns->append(*(aln_block_data->alns));
  });

  return fut;
}

void init_aligner(AlnScoring &aln_scoring, int rlen_limit) {
    SWARN("Assuming 1 IPU per rank, TODO change");
    if (upcxx::rank_me() > 64) {
      SWARN("More ranks than IPUs");
      exit(1);
    }
    double init_time;
    if (ipu_driver == NULL) {
      ipu::SWConfig config = {1, 1, 2, -2, swatlib::Similarity::nucleicAcid, swatlib::DataType::nucleicAcid};
      ipu_driver = new ipu::batchaffine::SWAlgorithm(config, 300, 1472 * 6);
      // local_team().rank_me(), local_team().rank_n(), (short)aln_scoring.match,
      //                                    (short)-aln_scoring.mismatch, (short)-aln_scoring.gap_opening,
      //                                    (short)-aln_scoring.gap_extending, rlen_limit, init_time);
      SLOG_VERBOSE("Initialized ipuma driver in ", init_time, " s\n");
    }
}

void cleanup_aligner() {
  // delete ipu_driver;
}

void kernel_align_block(CPUAligner &cpu_aligner, vector<Aln> &kernel_alns, vector<string> &ctg_seqs, vector<string> &read_seqs,
                        Alns *alns, future<> &active_kernel_fut, int read_group_id, int max_clen, int max_rlen,
                        IntermittentTimer &aln_kernel_timer) {
//   BaseTimer steal_t("CPU work steal");
//   steal_t.start();
//   auto num = kernel_alns.size();
//   // steal work from this kernel block if the previous kernel is still active
//   // if true, this balances the block size that will be sent to the kernel
//   while ((!gpu_utils::gpus_present() || !active_kernel_fut.ready()) && !kernel_alns.empty()) {
//     assert(!ctg_seqs.empty());
//     assert(!read_seqs.empty());
// #ifndef NO_KLIGN_CPU_WORK_STEAL
//     // steal one from the block
//     cpu_aligner.ssw_align_read(alns, kernel_alns.back(), ctg_seqs.back(), read_seqs.back(), read_group_id);
//     kernel_alns.pop_back();
//     ctg_seqs.pop_back();
//     read_seqs.pop_back();
// #endif
//     progress();
//   }
//   steal_t.stop();
//   auto steal_secs = steal_t.get_elapsed();
//   if (num != kernel_alns.size()) {
//     auto num_stole = num - kernel_alns.size();
//     LOG("Stole from kernel block ", num_stole, " alignments in ", steal_secs, "s (",
//         (steal_secs > 0 ? num_stole / steal_secs : 0.0), " aln/s), while waiting for previous block to complete",
//         (kernel_alns.empty() ? " - THE ENTIRE BLOCK" : ""), "\n");
//   } else if (steal_secs > 0.01) {
//     LOG("Waited ", steal_secs, "s for previous block to complete\n");
//   }
  while (!active_kernel_fut.ready() || kernel_alns.empty()) {
    progress();
  }
  if (!kernel_alns.empty()) {
    assert(active_kernel_fut.ready() && "active_kernel_fut should already be ready");
    active_kernel_fut.wait();  // should be ready already
    shared_ptr<AlignBlockData> aln_block_data = make_shared<AlignBlockData>(kernel_alns, ctg_seqs, read_seqs, max_clen, max_rlen, read_group_id, cpu_aligner.aln_scoring);

    if (cpu_aligner.ssw_filter.report_cigar) {
      SWARN("Not implemented cigar for IPUs");
      exit(1);
    }
    
    assert(kernel_alns.empty());
    active_kernel_fut = ipu_align_block(aln_block_data, alns, cpu_aligner.ssw_filter.report_cigar, aln_kernel_timer);
    // // for now, the GPU alignment doesn't support cigars
    // if (cpu_aligner.ssw_filter.report_cigar) {
    //   SWARN("Not implemented cigar for IPUs");
    //   exit(1);
    //   // active_kernel_fut = gpu_align_block(aln_block_data, alns, cpu_aligner.ssw_filter.report_cigar, aln_kernel_timer);
    // } else {
    //   active_kernel_fut = cpu_aligner.ssw_align_block(aln_block_data, alns);
    // }
  } else {
    SWARN("Block shall not be empty");  
    exit(1);
  }
}