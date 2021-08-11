/*
 HipMer v 2.0, Copyright (c) 2020, The Regents of the University of California,
 through Lawrence Berkeley National Laboratory (subject to receipt of any required
 approvals from the U.S. Dept. of Energy).  All rights reserved."

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley National
 Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to
 endorse or promote products derived from this software without specific prior
 written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
 to the features, functionality or performance of the source code ("Enhancements") to
 anyone; however, if you choose to make your Enhancements available either publicly,
 or directly to Lawrence Berkeley National Laboratory, without imposing a separate
 written license agreement for such Enhancements, then you hereby grant the following
 license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
 prepare derivative works, incorporate into other computer software, distribute, and
 sublicense such enhancements or derivative works thereof, in binary and source code
 form.
*/

#include "upcxx_utils.hpp"
#include "utils.hpp"
#include "kcount.hpp"

#include "gpu-utils/gpu_utils.hpp"
#include "kcount-gpu/parse_and_pack.hpp"

//#define DBG_ADD_KMER DBG
#define DBG_ADD_KMER(...)

using namespace std;
using namespace upcxx_utils;
using namespace upcxx;

static kcount_gpu::ParseAndPackGPUDriver *pnp_gpu_driver;

template <int MAX_K>
static void process_block_gpu(unsigned kmer_len, string &seq_block, const vector<kmer_count_t> &depth_block,
                              dist_object<KmerDHT<MAX_K>> &kmer_dht, int64_t &num_Ns, int64_t &num_kmers, int64_t &num_gpu_waits,
                              int64_t &bytes_kmers_sent, int64_t &bytes_supermers_sent) {
  bool from_ctgs = !depth_block.empty();
  unsigned int num_valid_kmers = 0;
  if (!pnp_gpu_driver->process_seq_block(seq_block, num_valid_kmers))
    DIE("seq length is too high, ", seq_block.length(), " >= ", KCOUNT_GPU_SEQ_BLOCK_SIZE);
  bytes_kmers_sent += sizeof(KmerAndExt<MAX_K>) * num_valid_kmers;
  while (!pnp_gpu_driver->kernel_is_done()) {
    num_gpu_waits++;
    progress();
  }
  pnp_gpu_driver->pack_seq_block(seq_block);
  int num_targets = (int)pnp_gpu_driver->supermers.size();
  for (int i = 0; i < num_targets; i++) {
    auto target = pnp_gpu_driver->supermers[i].target;
    auto offset = pnp_gpu_driver->supermers[i].offset;
    auto len = pnp_gpu_driver->supermers[i].len;
    Supermer supermer;
    int packed_len = len / 2;
    if (offset % 2 || len % 2) packed_len++;
    supermer.seq = pnp_gpu_driver->packed_seqs.substr(offset / 2, packed_len);
    if (offset % 2) supermer.seq[0] &= 15;
    if ((offset + len) % 2) supermer.seq[supermer.seq.length() - 1] &= 240;
    supermer.count = (from_ctgs ? depth_block[offset + 1] : (kmer_count_t)1);
    bytes_supermers_sent += supermer.get_bytes();
    kmer_dht->add_supermer(supermer, target);
    progress();
  }
}

template <int MAX_K>
static void count_kmers(unsigned kmer_len, int qual_offset, vector<PackedReads *> &packed_reads_list,
                        dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_reads = 0;
  int64_t num_lines = 0;
  int64_t num_kmers = 0;
  int64_t num_bad_quals = 0;
  int64_t num_Ns = 0;
  int64_t bytes_supermers_sent = 0;
  int64_t bytes_kmers_sent = 0;

  string progbar_prefix = "";
  IntermittentTimer t_pp(__FILENAME__ + string(":kmer parse and pack"));
  kmer_dht->set_pass(READ_KMERS_PASS);
  barrier();
  int64_t num_gpu_waits = 0;
  int num_read_blocks = 0;
  string seq_block;
  seq_block.reserve(KCOUNT_GPU_SEQ_BLOCK_SIZE);
  if (gpu_utils::get_num_node_gpus() <= 0) {
    DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  } else {
    double init_time;
    pnp_gpu_driver = new kcount_gpu::ParseAndPackGPUDriver(rank_me(), rank_n(), qual_offset, kmer_len, Kmer<MAX_K>::get_N_LONGS(),
                                                           kmer_dht->get_minimizer_len(), init_time);
    SLOG_GPU("Initialized PnP GPU driver in ", fixed, setprecision(3), init_time, " s\n");
  }

  int64_t tot_num_local_reads = 0;
  for (auto packed_reads : packed_reads_list) {
    tot_num_local_reads += packed_reads->get_local_num_reads();
  }
  ProgressBar progbar(tot_num_local_reads, "Processing reads to count kmers");

  for (auto packed_reads : packed_reads_list) {
    packed_reads->reset();
    string id, seq, quals;
    while (true) {
      if (!packed_reads->get_next_read(id, seq, quals)) break;
      num_reads++;
      progbar.update();
      if (seq.length() < kmer_len) continue;
      for (int i = 0; i < seq.length(); i++) {
        if (quals[i] < qual_offset + KCOUNT_QUAL_CUTOFF) {
          seq[i] = tolower(seq[i]);
          num_bad_quals++;
        }
      }
      if (seq_block.length() + 1 + seq.length() >= KCOUNT_GPU_SEQ_BLOCK_SIZE) {
        process_block_gpu(kmer_len, seq_block, {}, kmer_dht, num_Ns, num_kmers, num_gpu_waits, bytes_kmers_sent,
                          bytes_supermers_sent);
        seq_block.clear();
        num_read_blocks++;
      }
      seq_block += seq;
      seq_block += "_";
      progress();
    }
  }
  if (!seq_block.empty()) {
    process_block_gpu(kmer_len, seq_block, {}, kmer_dht, num_Ns, num_kmers, num_gpu_waits, bytes_kmers_sent, bytes_supermers_sent);
    num_read_blocks++;
  }
  progbar.done();
  SLOG_GPU("Number of calls to progress while PnP GPU driver was running: ", num_gpu_waits, "\n");
  auto [gpu_time_tot, gpu_time_kernel] = pnp_gpu_driver->get_elapsed_times();
  SLOG_GPU("Number of calls to PnP GPU kernel: ", num_read_blocks, "\n");
  SLOG_GPU("Elapsed times for PnP GPU: ", fixed, setprecision(3), " total ", gpu_time_tot, ", kernel ", gpu_time_kernel, "\n");
  delete pnp_gpu_driver;
  pnp_gpu_driver = nullptr;

  kmer_dht->flush_updates();
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  DBG("This rank processed ", num_reads, " reads\n");
  auto all_num_reads = reduce_one(num_reads, op_fast_add, 0).wait();
  auto all_num_bad_quals = reduce_one(num_bad_quals, op_fast_add, 0).wait();
  auto all_num_Ns = reduce_one(num_Ns, op_fast_add, 0).wait();
  if (all_num_bad_quals) SLOG_VERBOSE("Found ", perc_str(all_num_bad_quals, all_num_kmers), " bad quality positions\n");
  if (all_num_Ns) SLOG_VERBOSE("Found ", perc_str(all_num_Ns, all_num_kmers), " kmers with Ns\n");
  auto tot_supermers_bytes_sent = reduce_one(bytes_supermers_sent, op_fast_add, 0).wait();
  auto tot_kmers_bytes_sent = reduce_one(bytes_kmers_sent, op_fast_add, 0).wait();
  SLOG_VERBOSE("Total bytes sent in compressed supermers ", get_size_str(tot_supermers_bytes_sent),
               " and what would have been sent in kmers ", get_size_str(tot_kmers_bytes_sent), " compression is ", fixed,
               setprecision(3), (double)tot_kmers_bytes_sent / tot_supermers_bytes_sent, "\n");
};

template <int MAX_K>
static void add_ctg_kmers(unsigned kmer_len, unsigned prev_kmer_len, Contigs &ctgs, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_kmers = 0;
  int64_t num_prev_kmers = kmer_dht->get_num_kmers();
  int64_t num_Ns = 0;
  int64_t bytes_supermers_sent = 0;
  int64_t bytes_kmers_sent = 0;

  ProgressBar progbar(ctgs.size(), "Adding extra contig kmers from kmer length " + to_string(prev_kmer_len));
  kmer_dht->set_pass(CTG_KMERS_PASS);
  auto start_local_num_kmers = kmer_dht->get_local_num_kmers();

  int64_t num_gpu_waits = 0;
  int num_ctg_blocks = 0;
  string seq_block = "";
  vector<kmer_count_t> depth_block;
  seq_block.reserve(KCOUNT_GPU_SEQ_BLOCK_SIZE);
  depth_block.reserve(KCOUNT_GPU_SEQ_BLOCK_SIZE);
  if (gpu_utils::get_num_node_gpus() <= 0) {
    DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  } else {
    double init_time;
    pnp_gpu_driver = new kcount_gpu::ParseAndPackGPUDriver(rank_me(), rank_n(), 0, kmer_len, Kmer<MAX_K>::get_N_LONGS(),
                                                           kmer_dht->get_minimizer_len(), init_time);
    SLOG_GPU("Initialized parse and pack GPU driver in ", fixed, setprecision(3), init_time, " s\n");
  }
  // estimate number of kmers from ctgs
  int64_t max_kmers = 0;
  for (auto &ctg : ctgs) {
    if (ctg.seq.length() > kmer_len) max_kmers += ctg.seq.length() - kmer_len + 1;
  }
  int64_t all_max_kmers = reduce_all(max_kmers, op_fast_add).wait();
  kmer_dht->init_ctg_kmers(all_max_kmers / rank_n());

  for (auto it = ctgs.begin(); it != ctgs.end(); ++it) {
    auto ctg = it;
    progbar.update();
    if (ctg->seq.length() < kmer_len + 2) continue;
    if (seq_block.length() + 1 + ctg->seq.length() >= KCOUNT_GPU_SEQ_BLOCK_SIZE) {
      process_block_gpu(kmer_len, seq_block, depth_block, kmer_dht, num_Ns, num_kmers, num_gpu_waits, bytes_kmers_sent,
                        bytes_supermers_sent);
      seq_block.clear();
      depth_block.clear();
      num_ctg_blocks++;
    }
    if (ctg->seq.length() >= KCOUNT_GPU_SEQ_BLOCK_SIZE)
      DIE("Oh dear, my laziness is revealed: the ctg seq is too long ", ctg->seq.length(), " for this GPU implementation ",
          KCOUNT_GPU_SEQ_BLOCK_SIZE);
    seq_block += ctg->seq;
    seq_block += "_";
    depth_block.insert(depth_block.end(), ctg->seq.length() + 1, ctg->get_uint16_t_depth());
  }
  progbar.done();
  if (!seq_block.empty()) {
    process_block_gpu(kmer_len, seq_block, depth_block, kmer_dht, num_Ns, num_kmers, num_gpu_waits, bytes_kmers_sent,
                      bytes_supermers_sent);
    num_ctg_blocks++;
  }
  SLOG_GPU("Number of calls to progress while GPU PnP driver was running: ", num_gpu_waits, "\n");
  auto [gpu_time_tot, gpu_time_kernel] = pnp_gpu_driver->get_elapsed_times();
  SLOG_GPU("Number of calls to GPU PnP kernel ", num_ctg_blocks, " times\n");
  SLOG_GPU("PnP GPU times: ", fixed, setprecision(3), " total ", gpu_time_tot, ", kernel ", gpu_time_kernel, "\n");
  delete pnp_gpu_driver;
  pnp_gpu_driver = nullptr;
  kmer_dht->flush_updates();
  DBG("This rank processed ", ctgs.size(), " contigs and ", num_kmers, " kmers\n");
  auto all_num_ctgs = reduce_one(ctgs.size(), op_fast_add, 0).wait();
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_ctgs, " contigs and ", all_num_kmers, " kmers\n");
  auto tot_supermers_bytes_sent = reduce_one(bytes_supermers_sent, op_fast_add, 0).wait();
  auto tot_kmers_bytes_sent = reduce_one(bytes_kmers_sent, op_fast_add, 0).wait();
  SLOG_VERBOSE("Total bytes sent in compressed supermers ", get_size_str(tot_supermers_bytes_sent),
               " and what would have been sent in kmers ", get_size_str(tot_kmers_bytes_sent), " compression is ", fixed,
               setprecision(3), (double)tot_kmers_bytes_sent / tot_supermers_bytes_sent, "\n");
};

template <int MAX_K>
void analyze_kmers(unsigned kmer_len, unsigned prev_kmer_len, int qual_offset, vector<PackedReads *> &packed_reads_list,
                   int dmin_thres, Contigs &ctgs, dist_object<KmerDHT<MAX_K>> &kmer_dht, bool dump_kmers) {
  BarrierTimer timer(__FILEFUNC__);
  auto fut_has_contigs = upcxx::reduce_all(ctgs.size(), upcxx::op_fast_max).then([](size_t max_ctgs) { return max_ctgs > 0; });
  _dmin_thres = dmin_thres;

  count_kmers(kmer_len, qual_offset, packed_reads_list, kmer_dht);
  barrier();
  if (fut_has_contigs.wait()) {
    add_ctg_kmers(kmer_len, prev_kmer_len, ctgs, kmer_dht);
    barrier();
  }
  kmer_dht->compute_kmer_exts();
  if (dump_kmers) kmer_dht->dump_kmers();
  barrier();
  kmer_dht->clear_stores();
  auto [bytes_sent, max_bytes_sent] = kmer_dht->get_bytes_sent();
  SLOG_VERBOSE("Total bytes sent ", get_size_str(bytes_sent), " balance ", (double)bytes_sent / (rank_n() * max_bytes_sent), "\n");
};
