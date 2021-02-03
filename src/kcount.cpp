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

#include "kcount.hpp"

#define KCOUNT_GPUS

#if defined(ENABLE_GPUS) && defined(KCOUNT_GPUS)
#include "gpu-utils/utils.hpp"
#include "kcount-gpu/kcount_driver.hpp"
#endif

//#define DBG_ADD_KMER DBG
#define DBG_ADD_KMER(...)

using namespace std;

#if defined(ENABLE_GPUS) && defined(KCOUNT_GPUS)
template <int MAX_K>
static void process_read_block_gpu(kcount_gpu::KcountGPUDriver &gpu_driver, unsigned kmer_len, int qual_offset,
                                   const string &seq_block, const string &quals_block, dist_object<KmerDHT<MAX_K>> &kmer_dht,
                                   int64_t &num_Ns, int64_t &num_kmers, int64_t &num_gpu_waits) {
  int qual_cutoff = KCOUNT_QUAL_CUTOFF;
  gpu_driver.process_read_block(qual_offset, seq_block, num_Ns);
  while (!gpu_driver.kernel_is_done()) {
    num_gpu_waits++;
    progress();
  }
  auto &packed_kmers = gpu_driver.get_packed_kmers();
  auto &kmer_targets = gpu_driver.get_kmer_targets();
  auto &is_rcs = gpu_driver.get_is_rcs();
  int num_kmer_longs = Kmer<MAX_K>::get_N_LONGS();
  for (int i = 0; i < (int)kmer_targets.size(); i++) {
    // invalid kmer
    if (kmer_targets[i] == -1) continue;
    char left_base = '0';
    if (i > 0 && (quals_block[i - 1] >= qual_offset + qual_cutoff)) left_base = seq_block[i - 1];
    char right_base = '0';
    if (i < packed_kmers.size() - 1 && (quals_block[i + kmer_len] >= qual_offset + qual_cutoff))
      right_base = seq_block[i + kmer_len];
    if (is_rcs[i]) {
      swap(left_base, right_base);
      left_base = comp_nucleotide(left_base);
      right_base = comp_nucleotide(right_base);
    }
    Kmer<MAX_K> kmer(&(packed_kmers[i * num_kmer_longs]));
#ifdef DEBUG
    auto cpu_target = kmer_dht->get_kmer_target_rank(kmer);
    if (cpu_target != kmer_targets[i]) DIE("cpu target is ", cpu_target, " but gpu target is ", kmer_targets[i]);
#endif
    kmer_dht->add_kmer(kmer, left_base, right_base, 1, true, kmer_targets[i]);
    DBG_ADD_KMER("kcount add_kmer ", kmer.to_string(), " count ", 1, "\n");
    num_kmers++;
  }
}
#endif

template <int MAX_K>
static void process_read(unsigned kmer_len, int qual_offset, const string &seq, const string &quals,
                         dist_object<KmerDHT<MAX_K>> &kmer_dht, int64_t &num_Ns, int64_t &num_kmers, vector<Kmer<MAX_K>> &kmers) {
  int qual_cutoff = KCOUNT_QUAL_CUTOFF;
  // split into kmers
  Kmer<MAX_K>::get_kmers(kmer_len, seq, kmers);
  // skip kmers that contain an N
  size_t found_N_pos = seq.find_first_of('N');
  if (found_N_pos == string::npos)
    found_N_pos = seq.length();
  else
    num_Ns++;
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
    char left_base = '0';
    if (i > 0 && (quals[i - 1] >= qual_offset + qual_cutoff)) left_base = seq[i - 1];
    char right_base = '0';
    if (i < kmers.size() - 1 && (quals[i + kmer_len] >= qual_offset + qual_cutoff)) right_base = seq[i + kmer_len];
    kmer_dht->add_kmer(kmers[i], left_base, right_base, 1);
    DBG_ADD_KMER("kcount add_kmer ", kmers[i].to_string(), " count ", 1, "\n");
    num_kmers++;
  }
}

template <int MAX_K>
static void count_kmers(unsigned kmer_len, int qual_offset, vector<PackedReads *> &packed_reads_list,
                        dist_object<KmerDHT<MAX_K>> &kmer_dht, PASS_TYPE pass_type, int ranks_per_gpu) {
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
  switch (pass_type) {
    case BLOOM_SET_PASS: progbar_prefix = "Pass 1: Processing reads to setup bloom filter"; break;
    case BLOOM_COUNT_PASS: progbar_prefix = "Pass 2: Processing reads to count kmers"; break;
    case NO_BLOOM_PASS: progbar_prefix = "Processing reads to count kmers"; break;
    default: DIE("Should never get here");
  };
  IntermittentTimer t_pp(__FILENAME__ + string(":kmer parse and pack"));
  kmer_dht->set_pass(pass_type);
  barrier();

#if defined(ENABLE_GPUS) && defined(KCOUNT_GPUS)
  kcount_gpu::KcountGPUDriver gpu_driver;
  int gpu_devices = gpu_utils::get_num_node_gpus();
  if (gpu_devices <= 0) {
    // CPU only
    gpu_devices = 0;
    SWARN("GPUs are enabled but no GPU could be configured for kmer counting");
  } else {
    auto init_time = gpu_driver.init(rank_me(), rank_n(), kmer_len, Kmer<MAX_K>::get_N_LONGS());
    SLOG(KLGREEN, "Initialized kcount_gpu driver in ", fixed, setprecision(3), init_time, " s", KNORM, "\n");
  }
  int64_t num_gpu_waits = 0;
  int num_read_blocks = 0;
  string seq_block, quals_block;
  seq_block.reserve(KCOUNT_READ_BLOCK_SIZE);
  quals_block.reserve(KCOUNT_READ_BLOCK_SIZE);
#endif

  for (auto packed_reads : packed_reads_list) {
    packed_reads->reset();
    string id, seq, quals;
    ProgressBar progbar(packed_reads->get_local_num_reads(), progbar_prefix);
#if !defined(ENABLE_GPUS) || !defined(KCOUNT_GPUS)
    vector<Kmer<MAX_K>> kmers;
#endif
    while (true) {
      if (!packed_reads->get_next_read(id, seq, quals)) break;
      num_reads++;
      progbar.update();
      if (seq.length() < kmer_len) continue;
#if defined(ENABLE_GPUS) && defined(KCOUNT_GPUS)
      if (seq_block.length() + 1 + seq.length() >= KCOUNT_READ_BLOCK_SIZE) {
        process_read_block_gpu(gpu_driver, kmer_len, qual_offset, seq_block, quals_block, kmer_dht, num_Ns, num_kmers,
                               num_gpu_waits);
        seq_block.clear();
        quals_block.clear();
        num_read_blocks++;
      }
      seq_block += seq;
      seq_block += "_";
      quals_block += quals;
      quals_block += " ";
#else
      process_read(kmer_len, qual_offset, seq, quals, kmer_dht, num_Ns, num_kmers, kmers);
      progress();
#endif
    }
    progbar.done();
  }
#if defined(ENABLE_GPUS) && defined(KCOUNT_GPUS)
  if (!seq_block.empty()) {
    process_read_block_gpu(gpu_driver, kmer_len, qual_offset, seq_block, quals_block, kmer_dht, num_Ns, num_kmers, num_gpu_waits);
    num_read_blocks++;
  }
  SLOG(KLGREEN, "Number of calls to progress while gpu driver was running: ", num_gpu_waits, KNORM, "\n");
  auto [gpu_time_tot, gpu_time_malloc, gpu_time_cp, gpu_time_kernel] = gpu_driver.get_elapsed_times();
  SLOG(KLGREEN, "Called GPU ", num_read_blocks, " times", KNORM, "\n");
  SLOG(KLGREEN, "GPU times (secs): ", fixed, setprecision(3), " total ", gpu_time_tot, ", malloc ", gpu_time_malloc, ", cp ",
       gpu_time_cp, ", kernel ", gpu_time_kernel, KNORM, "\n");
#endif

  kmer_dht->flush_updates();
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  DBG("This rank processed ", num_reads, " reads\n");
  auto all_num_reads = reduce_one(num_reads, op_fast_add, 0).wait();
  auto all_num_bad_quals = reduce_one(num_bad_quals, op_fast_add, 0).wait();
  auto all_num_Ns = reduce_one(num_Ns, op_fast_add, 0).wait();
  auto all_distinct_kmers = kmer_dht->get_num_kmers();
  SLOG_VERBOSE("Processed a total of ", all_num_reads, " reads\n");
  if (pass_type != BLOOM_SET_PASS) {
    SLOG_VERBOSE("Found ", perc_str(all_distinct_kmers, all_num_kmers), " unique kmers\n");
    if (all_num_bad_quals) SLOG_VERBOSE("Found ", perc_str(all_num_bad_quals, all_num_kmers), " bad quality positions\n");
    if (all_num_Ns) SLOG_VERBOSE("Found ", perc_str(all_num_Ns, all_num_kmers), " kmers with Ns\n");
  }
};

// count ctg kmers if using bloom
template <int MAX_K>
static void count_ctg_kmers(unsigned kmer_len, Contigs &ctgs, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  BarrierTimer timer(__FILEFUNC__);
  ProgressBar progbar(ctgs.size(), "Counting kmers in contigs");
  int64_t num_kmers = 0;
  vector<Kmer<MAX_K>> kmers;
  kmer_dht->set_pass(CTG_BLOOM_SET_PASS);
  for (auto it = ctgs.begin(); it != ctgs.end(); ++it) {
    auto ctg = it;
    progbar.update();
    if (ctg->seq.length() >= kmer_len) {
      Kmer<MAX_K>::get_kmers(kmer_len, ctg->seq, kmers);
      if (kmers.size() != ctg->seq.length() - kmer_len + 1)
        DIE("kmers size mismatch ", kmers.size(), " != ", (ctg->seq.length() - kmer_len + 1), " '", ctg->seq, "'");
      for (int i = 1; i < (int)(ctg->seq.length() - kmer_len); i++) {
        kmer_dht->add_kmer(kmers[i], ctg->seq[i - 1], ctg->seq[i + kmer_len], 1);
      }
      num_kmers += kmers.size();
    }
    progress();
  }
  progbar.done();
  kmer_dht->flush_updates();
  DBG("This rank processed ", ctgs.size(), " contigs and ", num_kmers, " kmers\n");
  auto all_num_ctgs = reduce_one(ctgs.size(), op_fast_add, 0).wait();
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_ctgs, " contigs and ", all_num_kmers, " kmers\n");
  barrier();
};

template <int MAX_K>
static void add_ctg_kmers(unsigned kmer_len, unsigned prev_kmer_len, Contigs &ctgs, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_kmers = 0;
  int64_t num_prev_kmers = kmer_dht->get_num_kmers();
#ifdef USE_KMER_DEPTHS
  double tot_depth_diff = 0;
  double max_depth_diff = 0;
#endif
  ProgressBar progbar(ctgs.size(), "Adding extra contig kmers from kmer length " + to_string(prev_kmer_len));
  vector<Kmer<MAX_K>> kmers;
  kmer_dht->set_pass(CTG_KMERS_PASS);
  for (auto it = ctgs.begin(); it != ctgs.end(); ++it) {
    auto ctg = it;
    progbar.update();
    if (ctg->seq.length() >= kmer_len + 2) {
      Kmer<MAX_K>::get_kmers(kmer_len, ctg->seq, kmers);
      if (kmers.size() != ctg->seq.length() - kmer_len + 1)
        DIE("kmers size mismatch ", kmers.size(), " != ", (ctg->seq.length() - kmer_len + 1), " '", ctg->seq, "'");
      for (int i = 1; i < (int)(ctg->seq.length() - kmer_len); i++) {
        kmer_count_t depth = ctg->depth;
#ifdef USE_KMER_DEPTHS
        kmer_count_t kmer_depth = ctg->get_kmer_depth(i, kmer_len, prev_kmer_len);
        tot_depth_diff += (double)(kmer_depth - depth) / depth;
        max_depth_diff = max(max_depth_diff, (double)abs(kmer_depth - depth));
        depth = kmer_depth;
#endif
        kmer_dht->add_kmer(kmers[i], ctg->seq[i - 1], ctg->seq[i + kmer_len], depth);
        num_kmers++;
      }
    }
    progress();
  }
  progbar.done();
  kmer_dht->flush_updates();
  DBG("This rank processed ", ctgs.size(), " contigs and ", num_kmers, " kmers\n");
  auto all_num_ctgs = reduce_one(ctgs.size(), op_fast_add, 0).wait();
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_ctgs, " contigs and ", all_num_kmers, " kmers\n");
  SLOG_VERBOSE("Found ", perc_str(kmer_dht->get_num_kmers() - num_prev_kmers, all_num_kmers), " additional unique kmers\n");
#ifdef USE_KMER_DEPTH
  auto all_tot_depth_diff = reduce_one(tot_depth_diff, op_fast_add, 0).wait();
  SLOG_VERBOSE(KLRED, "Average depth diff ", all_tot_depth_diff / all_num_kmers, " max depth diff ",
               reduce_one(max_depth_diff, op_fast_max, 0).wait(), KNORM, "\n");
#endif
};

template <int MAX_K>
void analyze_kmers(unsigned kmer_len, unsigned prev_kmer_len, int qual_offset, vector<PackedReads *> &packed_reads_list,
                   int dmin_thres, Contigs &ctgs, dist_object<KmerDHT<MAX_K>> &kmer_dht, int ranks_per_gpu, bool dump_kmers) {
  BarrierTimer timer(__FILEFUNC__);
  auto fut_has_contigs = upcxx::reduce_all(ctgs.size(), upcxx::op_fast_max).then([](size_t max_ctgs) { return max_ctgs > 0; });

  _dynamic_min_depth = DYN_MIN_DEPTH;
  _dmin_thres = dmin_thres;

  if (kmer_dht->get_use_bloom()) {
    count_kmers(kmer_len, qual_offset, packed_reads_list, kmer_dht, BLOOM_SET_PASS, ranks_per_gpu);
    if (fut_has_contigs.wait()) count_ctg_kmers(kmer_len, ctgs, kmer_dht);
    kmer_dht->reserve_space_and_clear_bloom1();
    count_kmers(kmer_len, qual_offset, packed_reads_list, kmer_dht, BLOOM_COUNT_PASS, ranks_per_gpu);
  } else {
    count_kmers(kmer_len, qual_offset, packed_reads_list, kmer_dht, NO_BLOOM_PASS, ranks_per_gpu);
  }
  barrier();
  kmer_dht->print_load_factor();
  barrier();
  kmer_dht->purge_kmers(2);
  int64_t new_count = kmer_dht->get_num_kmers();
  SLOG_VERBOSE("After purge of kmers < 2, there are ", new_count, " unique kmers\n");
  barrier();
  if (fut_has_contigs.wait()) {
    add_ctg_kmers(kmer_len, prev_kmer_len, ctgs, kmer_dht);
    kmer_dht->purge_kmers(1);
  }
  barrier();
  kmer_dht->compute_kmer_exts();
  if (dump_kmers) kmer_dht->dump_kmers(kmer_len);
  barrier();
  kmer_dht->clear_stores();
  auto [bytes_sent, max_bytes_sent] = kmer_dht->get_bytes_sent();
  SLOG_VERBOSE("Total bytes sent ", get_size_str(bytes_sent), " balance ", (double)bytes_sent / (rank_n() * max_bytes_sent), "\n");
};
