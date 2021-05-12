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

#ifdef ENABLE_KCOUNT_GPUS
#include "gpu-utils/gpu_utils.hpp"
#include "kcount-gpu/parse_and_pack.hpp"

static kcount_gpu::ParseAndPackGPUDriver *pnp_gpu_driver;

#endif

//#define DBG_ADD_KMER DBG
#define DBG_ADD_KMER(...)

using namespace std;
using namespace upcxx_utils;
using namespace upcxx;

#ifdef ENABLE_KCOUNT_GPUS

template <int MAX_K>
static void process_block_gpu(unsigned kmer_len, int qual_offset, const string &seq_block, const string &quals_block,
                              const vector<kmer_count_t> &depth_block, dist_object<KmerDHT<MAX_K>> &kmer_dht, int64_t &num_Ns,
                              int64_t &num_kmers, int64_t &num_gpu_waits) {
  bool from_ctgs = quals_block.empty();
  int qual_cutoff = KCOUNT_QUAL_CUTOFF;
  if (!pnp_gpu_driver->process_seq_block(seq_block, num_Ns))
    DIE("seq length is too high, ", seq_block.length(), " >= ", KCOUNT_GPU_SEQ_BLOCK_SIZE);
  while (!pnp_gpu_driver->kernel_is_done()) {
    num_gpu_waits++;
    progress();
  }
  int num_kmer_longs = Kmer<MAX_K>::get_N_LONGS();
  for (int i = 0; i < (int)pnp_gpu_driver->host_kmer_targets.size(); i++) {
    // invalid kmer
    if (pnp_gpu_driver->host_kmer_targets[i] == -1) continue;
    char left_base = '0', right_base = '0';
    kmer_count_t depth = 1;
    if (from_ctgs) {
      if (i > 0 && seq_block[i - 1] != '_') left_base = seq_block[i - 1];
      if (i + kmer_len < seq_block.size() && seq_block[i + kmer_len] != '_') right_base = seq_block[i + kmer_len];
      if (right_base == '0' || left_base == '0') continue;
      depth = depth_block[i];
    } else {
      int qthres = qual_offset + qual_cutoff;
      if (i > 0 && quals_block[i - 1] >= qthres && seq_block[i - 1] != '_') left_base = seq_block[i - 1];
      if (i + kmer_len < quals_block.size() && quals_block[i + kmer_len] >= qthres && seq_block[i + kmer_len] != '_')
        right_base = seq_block[i + kmer_len];
    }
    if (pnp_gpu_driver->host_is_rcs[i]) {
      swap(left_base, right_base);
      left_base = comp_nucleotide(left_base);
      right_base = comp_nucleotide(right_base);
    }
    Kmer<MAX_K> kmer(&(pnp_gpu_driver->host_kmers[i * num_kmer_longs]));
#ifdef DEBUG
    auto cpu_target = kmer_dht->get_kmer_target_rank(kmer);
    if (cpu_target != pnp_gpu_driver->host_kmer_targets[i])
      DIE("cpu target is ", cpu_target, " but gpu target is ", pnp_gpu_driver->host_kmer_targets[i]);
#endif
    kmer_dht->add_kmer(kmer, left_base, right_base, depth, pnp_gpu_driver->host_kmer_targets[i]);
    DBG_ADD_KMER("kcount add_kmer ", kmer.to_string(), " count ", 1, "\n");
    num_kmers++;
  }
}

#else

template <int MAX_K>
static void process_seq(unsigned kmer_len, int qual_offset, const string &seq, const string &quals, int depth,
                        dist_object<KmerDHT<MAX_K>> &kmer_dht, int64_t &num_Ns, int64_t &num_kmers, vector<Kmer<MAX_K>> &kmers) {
  int qual_cutoff = KCOUNT_QUAL_CUTOFF;
  bool from_ctgs = quals.empty();
  // split into kmers
  Kmer<MAX_K>::get_kmers(kmer_len, seq, kmers);
  // skip kmers that contain an N
  size_t found_N_pos = seq.length();
  if (!from_ctgs) {
    found_N_pos = seq.find_first_of('N');
    if (found_N_pos == string::npos)
      found_N_pos = seq.length();
    else
      num_Ns++;
  }
  for (int i = 0; i < (int)kmers.size(); i++) {
    // skip kmers that contain an N
    if (i + kmer_len > found_N_pos) {
      i = found_N_pos;  // skip
      // find the next N
      found_N_pos = seq.find_first_of('N', found_N_pos + 1);
      if (found_N_pos == string::npos)
        found_N_pos = seq.length();
      else
        num_Ns++;
      continue;
    }
    char left_base = '0', right_base = '0';
    if (from_ctgs) {
      if (i > 0) left_base = seq[i - 1];
      if (i + kmer_len < seq.length()) right_base = seq[i + kmer_len];
      if (left_base == '0' || right_base == '0') continue;
    } else {
      int qthres = qual_offset + qual_cutoff;
      if (i > 0 && quals[i - 1] >= qthres) left_base = seq[i - 1];
      if (i + kmer_len < quals.length() && quals[i + kmer_len] >= qthres) right_base = seq[i + kmer_len];
    }
    kmer_dht->add_kmer(kmers[i], left_base, right_base, depth);
    DBG_ADD_KMER("kcount add_kmer ", kmers[i].to_string(), " count ", depth, "\n");
    num_kmers++;
  }
}

#endif

template <int MAX_K>
static void count_kmers(unsigned kmer_len, int qual_offset, vector<PackedReads *> &packed_reads_list,
                        dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  BarrierTimer timer(__FILEFUNC__);
  // probability of an error is P = 10^(-Q/10) where Q is the quality cutoff
  // so we want P = 0.5*1/k (i.e. 50% chance of 1 error)
  // and Q = -10 log10(P)
  // eg qual_cutoff for k=21 is 16, for k=99 is 22.
  // int qual_cutoff = -10 * log10(0.5 / kmer_len);
  // SLOG_VERBOSE("Using quality cutoff ", qual_cutoff, "\n");
  int qual_cutoff = KCOUNT_QUAL_CUTOFF;
  int64_t num_reads = 0;
  int64_t num_lines = 0;
  int64_t num_kmers = 0;
  int64_t num_bad_quals = 0;
  int64_t num_Ns = 0;
  string progbar_prefix = "";
  IntermittentTimer t_pp(__FILENAME__ + string(":kmer parse and pack"));
  kmer_dht->set_pass(READ_KMERS_PASS);
  barrier();
#ifdef ENABLE_KCOUNT_GPUS
  int64_t num_gpu_waits = 0;
  int num_read_blocks = 0;
  string seq_block, quals_block;
  seq_block.reserve(KCOUNT_GPU_SEQ_BLOCK_SIZE);
  quals_block.reserve(KCOUNT_GPU_SEQ_BLOCK_SIZE);
  if (gpu_utils::get_num_node_gpus() <= 0) {
    DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  } else {
    double init_time;
    pnp_gpu_driver = new kcount_gpu::ParseAndPackGPUDriver(rank_me(), rank_n(), kmer_len, Kmer<MAX_K>::get_N_LONGS(),
                                                           kmer_dht->get_minimizer_len(), init_time);
    SLOG(KLMAGENTA, "Initialized PnP GPU driver in ", fixed, setprecision(3), init_time, " s", KNORM, "\n");
  }
#else
  vector<Kmer<MAX_K>> kmers;
#endif

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
#ifdef ENABLE_KCOUNT_GPUS
      if (seq_block.length() + 1 + seq.length() >= KCOUNT_GPU_SEQ_BLOCK_SIZE) {
        process_block_gpu(kmer_len, qual_offset, seq_block, quals_block, {}, kmer_dht, num_Ns, num_kmers, num_gpu_waits);
        seq_block.clear();
        quals_block.clear();
        num_read_blocks++;
      }
      seq_block += seq;
      seq_block += "_";
      quals_block += quals;
      quals_block += '\0';
#else
      process_seq(kmer_len, qual_offset, seq, quals, 1, kmer_dht, num_Ns, num_kmers, kmers);
#endif
      progress();
    }
  }
#ifdef ENABLE_KCOUNT_GPUS
  if (!seq_block.empty()) {
    process_block_gpu(kmer_len, qual_offset, seq_block, quals_block, {}, kmer_dht, num_Ns, num_kmers, num_gpu_waits);
    num_read_blocks++;
  }
  progbar.done();
  SLOG(KLMAGENTA, "Number of calls to progress while PnP GPU driver was running: ", num_gpu_waits, KNORM, "\n");
  auto [gpu_time_tot, gpu_time_malloc, gpu_time_cp, gpu_time_kernel] = pnp_gpu_driver->get_elapsed_times();
  SLOG(KLMAGENTA, "Number of calls to PnP GPU kernel: ", num_read_blocks, KNORM, "\n");
  SLOG(KLMAGENTA, "Elapsed times for PnP GPU: ", fixed, setprecision(3), " total ", gpu_time_tot, ", malloc ", gpu_time_malloc,
       ", cp ", gpu_time_cp, ", kernel ", gpu_time_kernel, KNORM, "\n");
  delete pnp_gpu_driver;
  pnp_gpu_driver = nullptr;
#else
  progbar.done();
#endif

  kmer_dht->flush_updates();
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  DBG("This rank processed ", num_reads, " reads\n");
  auto all_num_reads = reduce_one(num_reads, op_fast_add, 0).wait();
  auto all_num_bad_quals = reduce_one(num_bad_quals, op_fast_add, 0).wait();
  auto all_num_Ns = reduce_one(num_Ns, op_fast_add, 0).wait();
  auto all_distinct_kmers = kmer_dht->get_num_kmers();
  SLOG_VERBOSE("Processed a total of ", all_num_reads, " reads\n");
  SLOG_VERBOSE("Found ", perc_str(all_distinct_kmers, all_num_kmers), " unique kmers\n");
  if (all_num_bad_quals) SLOG_VERBOSE("Found ", perc_str(all_num_bad_quals, all_num_kmers), " bad quality positions\n");
  if (all_num_Ns) SLOG_VERBOSE("Found ", perc_str(all_num_Ns, all_num_kmers), " kmers with Ns\n");
  auto tot_kmers_stored = reduce_one(kmer_dht->get_local_num_kmers(), op_fast_add, 0).wait();
  auto max_kmers_stored = reduce_one(kmer_dht->get_local_num_kmers(), op_fast_max, 0).wait();
  if (!rank_me()) {
    auto avg_kmers_stored = tot_kmers_stored / rank_n();
    SLOG_VERBOSE("Avg kmers in hash table per rank ", avg_kmers_stored, " max ", max_kmers_stored, " load balance ",
                 (double)avg_kmers_stored / max_kmers_stored, "\n");
  }
};

template <int MAX_K>
static void add_ctg_kmers(unsigned kmer_len, unsigned prev_kmer_len, Contigs &ctgs, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_kmers = 0;
  int64_t num_prev_kmers = kmer_dht->get_num_kmers();
  int64_t num_Ns = 0;
  ProgressBar progbar(ctgs.size(), "Adding extra contig kmers from kmer length " + to_string(prev_kmer_len));
  kmer_dht->set_pass(CTG_KMERS_PASS);
  auto start_local_num_kmers = kmer_dht->get_local_num_kmers();

#ifdef ENABLE_KCOUNT_GPUS
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
    pnp_gpu_driver = new kcount_gpu::ParseAndPackGPUDriver(rank_me(), rank_n(), kmer_len, Kmer<MAX_K>::get_N_LONGS(),
                                                           kmer_dht->get_minimizer_len(), init_time);
    SLOG(KLMAGENTA, "Initialized parse and pack GPU driver in ", fixed, setprecision(3), init_time, " s", KNORM, "\n");
  }
#else
  vector<Kmer<MAX_K>> kmers;
#endif

  for (auto it = ctgs.begin(); it != ctgs.end(); ++it) {
    auto ctg = it;
    progbar.update();
    if (ctg->seq.length() < kmer_len + 2) continue;
#ifdef ENABLE_KCOUNT_GPUS
    if (seq_block.length() + 1 + ctg->seq.length() >= KCOUNT_GPU_SEQ_BLOCK_SIZE) {
      process_block_gpu(kmer_len, 0, seq_block, {}, depth_block, kmer_dht, num_Ns, num_kmers, num_gpu_waits);
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
#else
    process_seq(kmer_len, 0, ctg->seq, {}, ctg->get_uint16_t_depth(), kmer_dht, num_Ns, num_kmers, kmers);
    progress();
#endif
  }
  progbar.done();
#ifdef ENABLE_KCOUNT_GPUS
  if (!seq_block.empty()) {
    process_block_gpu(kmer_len, 0, seq_block, {}, depth_block, kmer_dht, num_Ns, num_kmers, num_gpu_waits);
    num_ctg_blocks++;
  }
  SLOG(KLMAGENTA, "Number of calls to progress while GPU PnP driver was running: ", num_gpu_waits, KNORM, "\n");
  auto [gpu_time_tot, gpu_time_malloc, gpu_time_cp, gpu_time_kernel] = pnp_gpu_driver->get_elapsed_times();
  SLOG(KLMAGENTA, "Number of calls to GPU PnP kernel ", num_ctg_blocks, " times", KNORM, "\n");
  SLOG(KLMAGENTA, "PnP GPU times (secs): ", fixed, setprecision(3), " total ", gpu_time_tot, ", malloc ", gpu_time_malloc, ", cp ",
       gpu_time_cp, ", kernel ", gpu_time_kernel, KNORM, "\n");
  delete pnp_gpu_driver;
  pnp_gpu_driver = nullptr;
#endif
  kmer_dht->flush_updates();
  DBG("This rank processed ", ctgs.size(), " contigs and ", num_kmers, " kmers\n");
  auto all_num_ctgs = reduce_one(ctgs.size(), op_fast_add, 0).wait();
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_ctgs, " contigs and ", all_num_kmers, " kmers\n");
  SLOG_VERBOSE("Found ", perc_str(kmer_dht->get_num_kmers() - num_prev_kmers, all_num_kmers), " additional unique kmers\n");
  auto local_kmers = kmer_dht->get_local_num_kmers() - start_local_num_kmers;
  auto tot_kmers_stored = reduce_one(local_kmers, op_fast_add, 0).wait();
  auto max_kmers_stored = reduce_one(local_kmers, op_fast_max, 0).wait();
  if (!rank_me()) {
    auto avg_kmers_stored = tot_kmers_stored / rank_n();
    SLOG_VERBOSE("add ctgs: avg kmers in hash table per rank ", avg_kmers_stored, " max ", max_kmers_stored, " load balance ",
                 (double)avg_kmers_stored / max_kmers_stored, "\n");
  }
};

template <int MAX_K>
void analyze_kmers(unsigned kmer_len, unsigned prev_kmer_len, int qual_offset, vector<PackedReads *> &packed_reads_list,
                   int dmin_thres, Contigs &ctgs, dist_object<KmerDHT<MAX_K>> &kmer_dht, bool dump_kmers) {
  BarrierTimer timer(__FILEFUNC__);
  auto fut_has_contigs = upcxx::reduce_all(ctgs.size(), upcxx::op_fast_max).then([](size_t max_ctgs) { return max_ctgs > 0; });

  _dynamic_min_depth = DYN_MIN_DEPTH;
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
