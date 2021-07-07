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

#include <stdarg.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/mem_profile.hpp"
#include "upcxx_utils/progress_bar.hpp"
#include "upcxx_utils/timers.hpp"
#include "zstr.hpp"

#include "stage_timers.hpp"
#include "kmer_dht.hpp"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

//#define DBG_INS_CTG_KMER DBG
#define DBG_INS_CTG_KMER(...)
//#define DBG_INSERT_KMER DBG
#define DBG_INSERT_KMER(...)

// global variables to avoid passing dist objs to rpcs
static uint64_t _num_kmers_counted = 0;
static int num_inserts = 0;

template <int MAX_K>
void KmerDHT<MAX_K>::get_kmers_and_exts(Supermer &supermer, vector<KmerAndExt<MAX_K>> &kmers_and_exts) {
  vector<bool> quals;
  quals.resize(supermer.seq.length());
  for (int i = 0; i < supermer.seq.length(); i++) {
    quals[i] = isupper(supermer.seq[i]);
    supermer.seq[i] = toupper(supermer.seq[i]);
  }
  auto kmer_len = Kmer<MAX_K>::get_k();
  vector<Kmer<MAX_K>> kmers;
  Kmer<MAX_K>::get_kmers(kmer_len, supermer.seq, kmers);
  kmers_and_exts.clear();
  for (int i = 1; i < (int)(supermer.seq.length() - kmer_len); i++) {
    Kmer<MAX_K> kmer = kmers[i];
    char left_ext = supermer.seq[i - 1];
    if (!quals[i - 1]) left_ext = '0';
    char right_ext = supermer.seq[i + kmer_len];
    if (!quals[i + kmer_len]) right_ext = '0';
    // get the lexicographically smallest
    Kmer<MAX_K> kmer_rc = kmer.revcomp();
    if (kmer_rc < kmer) {
      kmer = kmer_rc;
      swap(left_ext, right_ext);
      left_ext = comp_nucleotide(left_ext);
      right_ext = comp_nucleotide(right_ext);
    };
    kmers_and_exts.push_back({.kmer = kmer, .count = supermer.count, .left = left_ext, .right = right_ext});
  }
}

template <int MAX_K>
void KmerDHT<MAX_K>::update_count(Supermer supermer, dist_object<KmerMap> &kmers,
                                  dist_object<HashTableGPUDriver<MAX_K>> &ht_gpu_driver) {
  num_inserts++;
  ht_gpu_driver->insert_supermer(supermer.seq, supermer.count);
}

template <int MAX_K>
KmerDHT<MAX_K>::KmerDHT(uint64_t my_num_kmers, int max_kmer_store_bytes, int max_rpcs_in_flight, bool useHHSS)
    : kmers({})
    , ht_gpu_driver({})
    , kmer_store()
    , max_kmer_store_bytes(max_kmer_store_bytes)
    , initial_kmer_dht_reservation(0)
    , my_num_kmers(my_num_kmers)
    , max_rpcs_in_flight(max_rpcs_in_flight)
    , estimated_error_rate(0.0) {
  // minimizer len depends on k
  minimizer_len = Kmer<MAX_K>::get_k() * 2 / 3 + 1;
  if (minimizer_len < 15) minimizer_len = 15;
  if (minimizer_len > 27) minimizer_len = 27;
  SLOG_VERBOSE("Using a minimizer length of ", minimizer_len, "\n");
  // main purpose of the timer here is to track memory usage
  BarrierTimer timer(__FILEFUNC__);
  auto node0_cores = upcxx::local_team().rank_n();
  // check if we have enough memory to run - conservative because we don't want to run out of memory
  double adjustment_factor = 0.2;
  auto my_adjusted_num_kmers = my_num_kmers * adjustment_factor;
  double required_space = estimate_hashtable_memory(my_adjusted_num_kmers, sizeof(Kmer<MAX_K>) + sizeof(KmerCounts)) * node0_cores;
  auto max_reqd_space = upcxx::reduce_all(required_space, upcxx::op_fast_max).wait();
  auto free_mem = get_free_mem();
  auto lowest_free_mem = upcxx::reduce_all(free_mem, upcxx::op_fast_min).wait();
  auto highest_free_mem = upcxx::reduce_all(free_mem, upcxx::op_fast_max).wait();
  SLOG_VERBOSE("With adjustment factor of ", adjustment_factor, " require ", get_size_str(max_reqd_space), " per node (",
               my_adjusted_num_kmers, " kmers per rank), and there is ", get_size_str(lowest_free_mem), " to ",
               get_size_str(highest_free_mem), " available on the nodes\n");
  if (lowest_free_mem * 0.80 < max_reqd_space) SWARN("Insufficient memory available: this could crash with OOM");

  kmer_store.set_size("kmers", max_kmer_store_bytes, max_rpcs_in_flight, useHHSS);

  barrier();
  // in this case we have to roughly estimate the hash table size because the space is reserved now
  // err on the side of excess because the whole point of doing this is speed and we don't want a
  // hash table resize
  // Unfortunately, this estimate depends on the depth of the sample - high depth means more wasted memory,
  // but low depth means potentially resizing the hash table, which is very expensive
  initial_kmer_dht_reservation = my_adjusted_num_kmers;
  double kmers_space_reserved = my_adjusted_num_kmers * (sizeof(Kmer<MAX_K>) + sizeof(KmerCounts));
  SLOG_VERBOSE("Reserving at least ", get_size_str(node0_cores * kmers_space_reserved), " for kmer hash tables with ",
               node0_cores * my_adjusted_num_kmers, " entries on node 0\n");
  double init_free_mem = get_free_mem();
  if (my_adjusted_num_kmers <= 0) DIE("no kmers to reserve space for");
  if (gpu_utils::get_num_node_gpus() <= 0) {
    DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  } else {
    // calculate total slots for hash table. Reserve space for parse and pack
    int bytes_for_pnp = KCOUNT_GPU_SEQ_BLOCK_SIZE * (2 + Kmer<MAX_K>::get_N_LONGS() * sizeof(uint64_t) + sizeof(int));
    int max_dev_id = reduce_one(gpu_utils::get_gpu_device_pci_id(), op_fast_max, 0).wait();
    auto gpu_avail_mem = (gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n() - bytes_for_pnp) * 0.95;
    auto gpu_tot_mem = gpu_utils::get_tot_gpu_mem() * max_dev_id / upcxx::local_team().rank_n() - bytes_for_pnp;
    SLOG_GPU("Available GPU memory per rank for kmers hash table is ", get_size_str(gpu_avail_mem), " out of a max of ",
             get_size_str(gpu_tot_mem), "\n");
    double init_time;
    size_t gpu_bytes_reqd;
    my_adjusted_num_kmers *= 4;
    ht_gpu_driver->init(rank_me(), rank_n(), Kmer<MAX_K>::get_k(), my_adjusted_num_kmers, gpu_avail_mem, init_time, gpu_bytes_reqd);
    auto capacity = ht_gpu_driver->get_capacity(kcount_gpu::READ_KMERS_PASS);
    SLOG_GPU("GPU read kmers hash table has capacity per rank of ", capacity, " for ", (int64_t)my_adjusted_num_kmers,
             " elements\n");
    if (capacity < my_adjusted_num_kmers * 0.8)
      SLOG_VERBOSE("GPU read kmers hash table has less than requested capacity: ", perc_str(capacity, my_adjusted_num_kmers),
                   "; full capacity requires ", get_size_str(gpu_bytes_reqd), " memory on GPU but only have ",
                   get_size_str(gpu_avail_mem));

    SLOG_GPU("Initialized hash table GPU driver in ", std::fixed, std::setprecision(3), init_time, " s\n");
    auto gpu_free_mem = gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n();
    SLOG_GPU("After initializing GPU hash table, there is ", get_size_str(gpu_free_mem), " memory available per rank, with ",
             get_size_str(bytes_for_pnp), " reserved for PnP and ", get_size_str(gpu_bytes_reqd),
             " reserved for the kmer hash table\n");
  }
  barrier();
}

template <int MAX_K>
void KmerDHT<MAX_K>::clear() {
  kmers->clear();
  KmerMap().swap(*kmers);
  clear_stores();
}

template <int MAX_K>
void KmerDHT<MAX_K>::clear_stores() {
  kmer_store.clear();
}

template <int MAX_K>
KmerDHT<MAX_K>::~KmerDHT() {
  clear();
}

template <int MAX_K>
pair<int64_t, int64_t> KmerDHT<MAX_K>::get_bytes_sent() {
  auto all_bytes_sent = reduce_one(bytes_sent, op_fast_add, 0).wait();
  auto max_bytes_sent = reduce_one(bytes_sent, op_fast_max, 0).wait();
  return {all_bytes_sent, max_bytes_sent};
}

template <int MAX_K>
void KmerDHT<MAX_K>::init_ctg_kmers(int64_t max_elems) {
  using_ctg_kmers = true;
  int max_dev_id = reduce_one(gpu_utils::get_gpu_device_pci_id(), op_fast_max, 0).wait();
  // we don't need to reserve space for either pnp or the read kmers because those have already reduced the gpu_avail_mem
  auto gpu_avail_mem = (gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n()) * 0.95;
  auto gpu_tot_mem = gpu_utils::get_tot_gpu_mem() * max_dev_id / upcxx::local_team().rank_n();
  SLOG_GPU("Available GPU memory per rank for ctg kmers hash table is ", get_size_str(gpu_avail_mem), " out of a max of ",
           get_size_str(gpu_tot_mem), "\n");
  ht_gpu_driver->init_ctg_kmers(max_elems, gpu_avail_mem);
  SLOG_GPU("GPU ctg kmers hash table has capacity per rank of ", ht_gpu_driver->get_capacity(kcount_gpu::CTG_KMERS_PASS), " for ",
           fixed, max_elems, " elements\n");
}

template <int MAX_K>
void KmerDHT<MAX_K>::set_pass(PASS_TYPE pass_type) {
  _num_kmers_counted = 0;
  this->pass_type = pass_type;
  ht_gpu_driver->set_pass(READ_KMERS_PASS ? kcount_gpu::READ_KMERS_PASS : kcount_gpu::CTG_KMERS_PASS);
  kmer_store.set_update_func([&kmers = this->kmers, &ht_gpu_driver = this->ht_gpu_driver](Supermer supermer) {
    update_count(supermer, kmers, ht_gpu_driver);
  });
}

template <int MAX_K>
int KmerDHT<MAX_K>::get_minimizer_len() {
  return minimizer_len;
}

template <int MAX_K>
uint64_t KmerDHT<MAX_K>::get_num_kmers(bool all) {
  if (!all)
    return reduce_one((uint64_t)kmers->size(), op_fast_add, 0).wait();
  else
    return reduce_all((uint64_t)kmers->size(), op_fast_add).wait();
}

template <int MAX_K>
float KmerDHT<MAX_K>::max_load_factor() {
  return reduce_one(kmers->max_load_factor(), op_fast_max, 0).wait();
}

template <int MAX_K>
void KmerDHT<MAX_K>::print_load_factor() {
  int64_t num_kmers_est = initial_kmer_dht_reservation * rank_n();
  int64_t num_kmers = get_num_kmers();
  SLOG_VERBOSE("Originally reserved ", num_kmers_est, " and now have ", num_kmers, " elements\n");
  auto avg_load_factor = reduce_one(kmers->load_factor(), op_fast_add, 0).wait() / upcxx::rank_n();
  SLOG_VERBOSE("kmer DHT load factor: ", avg_load_factor, "\n");
}

template <int MAX_K>
int64_t KmerDHT<MAX_K>::get_local_num_kmers(void) {
  return kmers->size();
}

template <int MAX_K>
double KmerDHT<MAX_K>::get_estimated_error_rate() {
  return estimated_error_rate;
}

template <int MAX_K>
upcxx::intrank_t KmerDHT<MAX_K>::get_kmer_target_rank(const Kmer<MAX_K> &kmer, const Kmer<MAX_K> *kmer_rc) const {
  assert(&kmer != kmer_rc && "Can be a palindrome, cannot be the same Kmer instance");
  return kmer.minimizer_hash_fast(minimizer_len, kmer_rc) % rank_n();
}

template <int MAX_K>
KmerCounts *KmerDHT<MAX_K>::get_local_kmer_counts(Kmer<MAX_K> &kmer) {
  const auto it = kmers->find(kmer);
  if (it == kmers->end()) return nullptr;
  return &it->second;
}

#ifdef DEBUG
template <int MAX_K>
bool KmerDHT<MAX_K>::kmer_exists(Kmer<MAX_K> kmer_fw) {
  const Kmer<MAX_K> kmer_rc = kmer_fw.revcomp();
  const Kmer<MAX_K> *kmer = (kmer_rc < kmer_fw) ? &kmer_rc : &kmer_fw;

  return rpc(
             get_kmer_target_rank(kmer_fw, &kmer_rc),
             [](Kmer<MAX_K> kmer, dist_object<KmerMap> &kmers) -> bool {
               const auto it = kmers->find(kmer);
               if (it == kmers->end()) return false;
               return true;
             },
             *kmer, kmers)
      .wait();
}
#endif

template <int MAX_K>
void KmerDHT<MAX_K>::add_supermer(Supermer &supermer, int target_rank) {
  kmer_store.update(target_rank, supermer);
}

template <int MAX_K>
void KmerDHT<MAX_K>::flush_updates() {
  BarrierTimer timer(__FILEFUNC__);
  kmer_store.flush_updates();

  barrier();
  ht_gpu_driver->flush_inserts();
  // a bunch of stats about the hash table on the GPU
  auto insert_stats = ht_gpu_driver->get_stats(static_cast<kcount_gpu::PASS_TYPE>(pass_type));
  auto avg_num_gpu_calls = reduce_one(ht_gpu_driver->get_num_gpu_calls(), op_fast_add, 0).wait() / rank_n();
  auto max_num_gpu_calls = reduce_one(ht_gpu_driver->get_num_gpu_calls(), op_fast_max, 0).wait();
  SLOG_GPU("Number of calls to ", (pass_type == READ_KMERS_PASS ? "read" : "ctg"), " hash table GPU driver: ", avg_num_gpu_calls,
           " avg, ", max_num_gpu_calls, " max\n");
  uint64_t num_dropped_elems = reduce_one((uint64_t)insert_stats.dropped, op_fast_add, 0).wait();
  uint64_t num_attempted_inserts = reduce_one((uint64_t)insert_stats.attempted, op_fast_add, 0).wait();
  uint64_t capacity = ht_gpu_driver->get_capacity(static_cast<kcount_gpu::PASS_TYPE>(pass_type));
  uint64_t all_capacity = reduce_one(capacity, op_fast_add, 0).wait();
  if (num_dropped_elems) {
    if (num_dropped_elems > num_attempted_inserts / 10000)
      SWARN("GPU ", (pass_type == READ_KMERS_PASS ? "read" : "ctg"), " hash table: failed to insert ",
            perc_str(num_dropped_elems, num_attempted_inserts), " elements; capacity ", all_capacity);
    else
      SLOG_VERBOSE("GPU ", (pass_type == READ_KMERS_PASS ? "read" : "ctg"), " hash table: failed to insert ",
                   perc_str(num_dropped_elems, num_attempted_inserts), " elements; capacity ", all_capacity, "\n");
  }
  uint64_t key_empty_overlaps = reduce_one((uint64_t)insert_stats.key_empty_overlaps, op_fast_add, 0).wait();
  if (key_empty_overlaps) {
    if (key_empty_overlaps > num_attempted_inserts / 10000)
      SWARN("GPU ", (pass_type == READ_KMERS_PASS ? "read" : "ctg"), " hash table: dropped ",
            perc_str(key_empty_overlaps, num_attempted_inserts), " kmers with longs equal to KEY_EMPTY");
    else
      SLOG_VERBOSE("GPU ", (pass_type == READ_KMERS_PASS ? "read" : "ctg"), " hash table: dropped ",
                   perc_str(key_empty_overlaps, num_attempted_inserts), " kmers with longs equal to KEY_EMPTY\n");
  }
  double load = (double)(insert_stats.new_inserts) / capacity;
  double avg_load_factor = reduce_one(load, op_fast_add, 0).wait() / rank_n();
  double max_load_factor = reduce_one(load, op_fast_max, 0).wait();
  SLOG_GPU("GPU ", (pass_type == READ_KMERS_PASS ? "read" : "ctg"), " kmer hash table load factor ", fixed, setprecision(3),
           avg_load_factor, " avg, ", max_load_factor, " max\n");
  SLOG_GPU("GPU ", (pass_type == READ_KMERS_PASS ? "reads" : "ctgs"), " kmers hash table final size per rank is ",
           insert_stats.new_inserts, " entries\n");
  auto avg_kmers_processed = reduce_one(_num_kmers_counted, op_fast_add, 0).wait() / rank_n();
  auto max_kmers_processed = reduce_one(_num_kmers_counted, op_fast_max, 0).wait();
  SLOG_VERBOSE("Avg kmers processed per rank ", avg_kmers_processed, " (balance ",
               (double)avg_kmers_processed / max_kmers_processed, ")\n");
}

template <int MAX_K>
void KmerDHT<MAX_K>::purge_kmers(int threshold) {
  BarrierTimer timer(__FILEFUNC__);
  auto num_prior_kmers = get_num_kmers();
  uint64_t num_purged = 0;
  for (auto it = kmers->begin(); it != kmers->end();) {
    auto kmer_counts = make_shared<KmerCounts>(it->second);
    if ((kmer_counts->count < threshold) || (kmer_counts->left_exts.is_zero() && kmer_counts->right_exts.is_zero())) {
      num_purged++;
      it = kmers->erase(it);
    } else {
      ++it;
    }
  }
  auto all_num_purged = reduce_one(num_purged, op_fast_add, 0).wait();
  SLOG_VERBOSE("Purged ", perc_str(all_num_purged, num_prior_kmers), " kmers below frequency threshold of ", threshold, "\n");
}

template <int MAX_K>
void KmerDHT<MAX_K>::insert_from_gpu_hashtable() {
  barrier();
  Timer insert_timer("gpu insert to cpu timer");
  insert_timer.start();
  int num_dropped = 0, num_entries = 0, num_purged = 0;
  ht_gpu_driver->done_all_inserts(num_dropped, num_entries, num_purged);
  if (num_dropped)
    WARN("GPU dropped ", num_dropped, " entries out of ", num_entries, " when compacting to output hash table" KNORM "\n");
  auto all_capacity = reduce_one((uint64_t)ht_gpu_driver->get_capacity(kcount_gpu::READ_KMERS_PASS), op_fast_add, 0).wait();
  auto all_num_purged = reduce_one((uint64_t)num_purged, op_fast_add, 0).wait();
  auto all_num_entries = reduce_one((uint64_t)num_entries, op_fast_add, 0).wait();
  auto prepurge_num_entries = all_num_entries + all_num_purged;
  SLOG_GPU("GPU read kmers hash table: purged ", perc_str(all_num_purged, prepurge_num_entries), " singleton kmers out of ",
           prepurge_num_entries, "\n");
  SLOG_GPU("GPU hash table final size is ", all_num_entries, " entries and final load factor is ",
           ((double)all_num_entries / all_capacity), "\n");
  barrier();

  // add some space for the ctg kmers
  kmers->reserve(num_entries * 1.5);
  uint64_t invalid = 0;
  while (true) {
    assert(HashTableGPUDriver<MAX_K>::get_N_LONGS() == Kmer<MAX_K>::get_N_LONGS());
    auto [kmer_array, count_exts] = ht_gpu_driver->get_next_entry();
    if (!kmer_array) break;
    // empty slot
    if (!count_exts->count) continue;
    if (count_exts->left == 'X' && count_exts->right == 'X') {
      // these are eliminated during purging in CPU version
      invalid++;
      continue;
    }
    KmerCounts kmer_counts = {.left_exts = {0},
                              .right_exts = {0},
                              .uutig_frag = nullptr,
                              .count = static_cast<kmer_count_t>(min(
                                  static_cast<kcount_gpu::count_t>(std::numeric_limits<kmer_count_t>::max()), count_exts->count)),
                              .left = (char)count_exts->left,
                              .right = (char)count_exts->right,
                              .from_ctg = false};
    if ((kmer_counts.count < 2)) WARN("Found a kmer that should have been purged, count is ", kmer_counts.count);
    Kmer<MAX_K> kmer(reinterpret_cast<const uint64_t *>(kmer_array->longs));

    // FIXME: should only be done in debug mode
    const auto it = kmers->find(kmer);
    if (it != kmers->end())
      WARN("Found a duplicate kmer ", kmer.to_string(), " - shouldn't happen: existing count ", it->second.count, " new count ",
           kmer_counts.count);
    kmers->insert({kmer, kmer_counts});
  }
  insert_timer.stop();

  auto all_avg_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_add, 0).wait() / rank_n();
  auto all_max_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_max, 0).wait();
  SLOG_GPU("Inserting kmers from GPU to cpu hash table took ", all_avg_elapsed_time, " avg, ", all_max_elapsed_time, " max\n");
  auto all_kmers_size = reduce_one((uint64_t)kmers->size(), op_fast_add, 0).wait();
  if (kmers->size() != (num_entries - invalid))
    WARN("kmers->size() is ", kmers->size(), " != ", (num_entries - invalid), " num_entries");
  auto all_invalid = reduce_one((uint64_t)invalid, op_fast_add, 0).wait();
  if (all_kmers_size != all_num_entries - all_invalid)
    SWARN("CPU kmer counts not equal to gpu kmer counts: ", all_kmers_size, " != ", (all_num_entries - all_invalid),
          " all_num_entries: ", all_num_entries, " all_invalid: ", all_invalid);
  double gpu_insert_time = 0, gpu_kernel_time = 0;
  ht_gpu_driver->get_elapsed_time(gpu_insert_time, gpu_kernel_time);
  auto avg_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_add, 0).wait() / rank_n();
  auto max_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_max, 0).wait();
  auto avg_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_add, 0).wait() / rank_n();
  auto max_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_max, 0).wait();
  stage_timers.kernel_kmer_analysis->inc_elapsed(max_gpu_kernel_time);
  SLOG_GPU("Elapsed GPU time for kmer hash tables:\n");
  SLOG_GPU("  insert: ", fixed, setprecision(3), avg_gpu_insert_time, " avg, ", max_gpu_insert_time, " max\n");
  SLOG_GPU("  kernel: ", fixed, setprecision(3), avg_gpu_kernel_time, " avg, ", max_gpu_kernel_time, " max\n");
  barrier();
}

template <int MAX_K>
void KmerDHT<MAX_K>::compute_kmer_exts() {
  if (using_ctg_kmers) {
    barrier();
    int attempted_inserts = 0, dropped_inserts = 0, new_inserts = 0;
    ht_gpu_driver->done_ctg_kmer_inserts(attempted_inserts, dropped_inserts, new_inserts);
    barrier();
    auto num_dropped_elems = reduce_one((uint64_t)dropped_inserts, op_fast_add, 0).wait();
    auto num_attempted_inserts = reduce_one((uint64_t)attempted_inserts, op_fast_add, 0).wait();
    auto num_new_inserts = reduce_one((uint64_t)new_inserts, op_fast_add, 0).wait();
    SLOG_GPU("GPU ctg kmers hash table: inserted ", num_new_inserts, " new elements into read kmers hash table\n");
    auto all_capacity = reduce_one((uint64_t)ht_gpu_driver->get_capacity(kcount_gpu::READ_KMERS_PASS), op_fast_add, 0).wait();
    if (num_dropped_elems) {
      if (num_dropped_elems > num_attempted_inserts / 10000)
        SWARN("GPU read kmers hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts),
              " ctg kmers; total capacity ", all_capacity);
      else
        SLOG_VERBOSE("GPU read kmers hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts),
                     " ctg kmers; total capacity ", all_capacity, "\n");
    }
    barrier();
  }
  insert_from_gpu_hashtable();
}

// one line per kmer, format:
// KMERCHARS LR N
// where L is left extension and R is right extension, one char, either X, F or A, C, G, T
// where N is the count of the kmer frequency
template <int MAX_K>
void KmerDHT<MAX_K>::dump_kmers() {
  BarrierTimer timer(__FILEFUNC__);
  int k = Kmer<MAX_K>::get_k();
  string dump_fname = "kmers-" + to_string(k) + ".txt.gz";
  get_rank_path(dump_fname, rank_me());
  zstr::ofstream dump_file(dump_fname);
  ostringstream out_buf;
  ProgressBar progbar(kmers->size(), "Dumping kmers to " + dump_fname);
  int64_t i = 0;
  for (auto &elem : *kmers) {
    out_buf << elem.first << " " << elem.second.count << " " << elem.second.left << " " << elem.second.right;
    out_buf << " " << elem.second.left_exts.to_string() << " " << elem.second.right_exts.to_string();
    out_buf << endl;
    i++;
    if (!(i % 1000)) {
      dump_file << out_buf.str();
      out_buf = ostringstream();
    }
    progbar.update();
  }
  if (!out_buf.str().empty()) dump_file << out_buf.str();
  dump_file.close();
  progbar.done();
  SLOG_VERBOSE("Dumped ", this->get_num_kmers(), " kmers\n");
}

template <int MAX_K>
typename KmerDHT<MAX_K>::KmerMap::iterator KmerDHT<MAX_K>::local_kmers_begin() {
  return kmers->begin();
}

template <int MAX_K>
typename KmerDHT<MAX_K>::KmerMap::iterator KmerDHT<MAX_K>::local_kmers_end() {
  return kmers->end();
}

template <int MAX_K>
int32_t KmerDHT<MAX_K>::get_time_offset_us() {
  std::chrono::duration<double> t_elapsed = CLOCK_NOW() - start_t;
  return std::chrono::duration_cast<std::chrono::microseconds>(t_elapsed).count();
}
