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
static int64_t _num_kmers_counted = 0;
static int num_inserts = 0;

template <int MAX_K>
void KmerDHT<MAX_K>::update_count(KmerAndExt kmer_and_ext, dist_object<KmerMap> &kmers,
                                  dist_object<HashTableGPUDriver<MAX_K>> &ht_gpu_driver) {
#ifdef ENABLE_KCOUNT_GPUS_HT
  num_inserts++;
  // FIXME: buffer hash table entries, and when full, copy across to
  ht_gpu_driver->insert_kmer(kmer_and_ext.kmer.get_longs(), kmer_and_ext.count, kmer_and_ext.left, kmer_and_ext.right);
#else
  // find it - if it isn't found then insert it, otherwise increment the counts
  const auto it = kmers->find(kmer_and_ext.kmer);
  if (it == kmers->end()) {
    KmerCounts kmer_counts = {.left_exts = {0},
                              .right_exts = {0},
                              .uutig_frag = nullptr,
                              .count = kmer_and_ext.count,
                              .left = 'X',
                              .right = 'X',
                              .from_ctg = false};
    kmer_counts.left_exts.inc(kmer_and_ext.left, kmer_and_ext.count);
    kmer_counts.right_exts.inc(kmer_and_ext.right, kmer_and_ext.count);
    auto prev_bucket_count = kmers->bucket_count();
    kmers->insert({kmer_and_ext.kmer, kmer_counts});
    // since sizes are an estimate this could happen, but it will impact performance
    if (prev_bucket_count < kmers->bucket_count())
      SWARN("Hash table on rank 0 was resized from ", prev_bucket_count, " to ", kmers->bucket_count());
    DBG_INSERT_KMER("inserted kmer ", kmer_and_ext.kmer.to_string(), " with count ", kmer_counts.count, "\n");
  } else {
    auto kmer_count = &it->second;
    int count = kmer_count->count + kmer_and_ext.count;
    if (count > numeric_limits<kmer_count_t>::max()) count = numeric_limits<kmer_count_t>::max();
    kmer_count->count = count;
    kmer_count->left_exts.inc(kmer_and_ext.left, kmer_and_ext.count);
    kmer_count->right_exts.inc(kmer_and_ext.right, kmer_and_ext.count);
  }
#endif
}

template <int MAX_K>
void KmerDHT<MAX_K>::update_ctg_kmers_count(KmerAndExt kmer_and_ext, dist_object<KmerMap> &kmers,
                                            dist_object<HashTableGPUDriver<MAX_K>> &ht_gpu_driver) {
#ifdef ENABLE_KCOUNT_GPUS_HT
  num_inserts++;
  // FIXME: buffer hash table entries, and when full, copy across to
  ht_gpu_driver->insert_kmer(kmer_and_ext.kmer.get_longs(), kmer_and_ext.count, kmer_and_ext.left, kmer_and_ext.right);
#else
  // insert a new kmer derived from the previous round's contigs
  const auto it = kmers->find(kmer_and_ext.kmer);
  bool insert = false;
  if (it == kmers->end()) {
    // if it isn't found then insert it
    insert = true;
  } else {
    auto kmer_counts = &it->second;
    if (!kmer_counts->from_ctg) {
      // existing kmer is from a read, only replace with new contig kmer if the existing kmer is not UU
      char left_ext = kmer_counts->get_left_ext();
      char right_ext = kmer_counts->get_right_ext();
      if (left_ext == 'X' || left_ext == 'F' || right_ext == 'X' || right_ext == 'F') {
        // non-UU, replace
        insert = true;
        DBG_INS_CTG_KMER("replace non-UU read kmer\n");
      }
    } else {
      // existing kmer from previous round's contigs
      // update kmer counts
      if (!kmer_counts->count) {
        // previously must have been a conflict and set to zero, so don't do anything
        DBG_INS_CTG_KMER("skip conflicted kmer, depth 0\n");
      } else {
        // will always insert, although it may get purged later for a conflict
        insert = true;
        char left_ext = kmer_counts->get_left_ext();
        char right_ext = kmer_counts->get_right_ext();
        if (left_ext != kmer_and_ext.left || right_ext != kmer_and_ext.right) {
          // if the two contig kmers disagree on extensions, set up to purge by setting the count to 0
          kmer_and_ext.count = 0;
          DBG_INS_CTG_KMER("set to purge conflict: prev ", left_ext, ", ", right_ext, " new ", kmer_and_ext.left, ", ",
                           kmer_and_ext.right, "\n");
        } else {
          // multiple occurrences of the same kmer derived from different contigs or parts of contigs
          // The only way this kmer could have been already found in the contigs only is if it came from a localassm
          // extension. In which case, all such kmers should not be counted again for each contig, because each
          // contig can use the same reads independently, and the depth will be oversampled.
          kmer_and_ext.count = min(kmer_and_ext.count, kmer_counts->count);
          DBG_INS_CTG_KMER("increase count of existing ctg kmer from ", kmer_counts->count, " to ", kmer_and_ext.count, "\n");
        }
      }
    }
  }
  if (insert) {
    kmer_count_t count = kmer_and_ext.count;
    KmerCounts kmer_counts = {
        .left_exts = {0}, .right_exts = {0}, .uutig_frag = nullptr, .count = count, .left = 'X', .right = 'X', .from_ctg = true};
    kmer_counts.left_exts.inc(kmer_and_ext.left, count);
    kmer_counts.right_exts.inc(kmer_and_ext.right, count);
    (*kmers)[kmer_and_ext.kmer] = kmer_counts;
  }
#endif
}

template <int MAX_K>
KmerDHT<MAX_K>::KmerDHT(uint64_t my_num_kmers, int max_kmer_store_bytes, int max_rpcs_in_flight, bool useHHSS)
    : kmers({})
    , ht_gpu_driver({})
    , kmer_store(kmers, ht_gpu_driver)
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
#ifdef ENABLE_KCOUNT_GPUS_HT
  if (gpu_utils::get_num_node_gpus() <= 0) {
    DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  } else {
    // calculate total slots for hash table. Reserve space for parse and pack
    int bytes_for_pnp = KCOUNT_GPU_SEQ_BLOCK_SIZE * (2 + Kmer<MAX_K>::get_N_LONGS() * sizeof(uint64_t) + sizeof(int));
    int max_dev_id = reduce_one(gpu_utils::get_gpu_device_pci_id(), op_fast_max, 0).wait();
    auto gpu_avail_mem = (gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n() - bytes_for_pnp) * 0.9;
    auto gpu_tot_mem = gpu_utils::get_tot_gpu_mem() * max_dev_id / upcxx::local_team().rank_n() - bytes_for_pnp;
    SLOG(KLMAGENTA, "Available GPU memory per rank for kmers hash table is ", get_size_str(gpu_avail_mem), " out of a max of ",
         get_size_str(gpu_tot_mem), KNORM, "\n");
    double init_time;
    size_t gpu_bytes_reqd;
    my_adjusted_num_kmers *= 4;
    ht_gpu_driver->init(rank_me(), rank_n(), Kmer<MAX_K>::get_k(), my_adjusted_num_kmers, gpu_avail_mem, init_time, gpu_bytes_reqd);
    auto capacity = ht_gpu_driver->get_capacity(kcount_gpu::READ_KMERS_PASS);
    SLOG(KLMAGENTA, "GPU ctg kmers hash table has capacity per rank of ", capacity, " for ", my_adjusted_num_kmers,
         " elements" KNORM "\n");
    if (capacity < my_adjusted_num_kmers)
      SWARN("Require ", get_size_str(gpu_bytes_reqd), " memory on GPU but only have ", get_size_str(gpu_avail_mem),
            ". Capacity is ", perc_str(capacity, my_adjusted_num_kmers), " of max expected kmers");

    SLOG(KLMAGENTA, "Initialized hash table GPU driver in ", std::fixed, std::setprecision(3), init_time, " s", KNORM, "\n");
    auto gpu_free_mem = gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n();
    SLOG(KLMAGENTA, "After initializing GPU hash table, there is ", get_size_str(gpu_free_mem), " memory available per rank, with ",
         get_size_str(bytes_for_pnp), " reserved for PnP and ", get_size_str(gpu_bytes_reqd),
         " reserved for the kmer hash table" KNORM, "\n");
  }
#else
  kmers->reserve(my_adjusted_num_kmers);
#endif
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
  // SLOG(KLMAGENTA, "There were ", perc_str(num_last_inserts, num_inserts), " last inserts out of ", num_inserts, "\n");
  clear();
}

template <int MAX_K>
pair<int64_t, int64_t> KmerDHT<MAX_K>::get_bytes_sent() {
  auto all_bytes_sent = reduce_one(bytes_sent, op_fast_add, 0).wait();
  auto max_bytes_sent = reduce_one(bytes_sent, op_fast_max, 0).wait();
  return {all_bytes_sent, max_bytes_sent};
}

template <int MAX_K>
void KmerDHT<MAX_K>::init_ctg_kmers(int max_elems) {
  using_ctg_kmers = true;
#ifdef ENABLE_KCOUNT_GPUS_HT
  int max_dev_id = reduce_one(gpu_utils::get_gpu_device_pci_id(), op_fast_max, 0).wait();
  // we don't need to reserve space for either pnp or the read kmers because those have already reduced the gpu_avail_mem
  auto gpu_avail_mem = (gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n()) * 0.9;
  auto gpu_tot_mem = gpu_utils::get_tot_gpu_mem() * max_dev_id / upcxx::local_team().rank_n();
  SLOG(KLMAGENTA, "Available GPU memory per rank for ctg kmers hash table is ", get_size_str(gpu_avail_mem), " out of a max of ",
       get_size_str(gpu_tot_mem), KNORM, "\n");
  ht_gpu_driver->init_ctg_kmers(max_elems, gpu_avail_mem);
  SLOG(KLMAGENTA, "GPU ctg kmers hash table has capacity per rank of ", ht_gpu_driver->get_capacity(kcount_gpu::CTG_KMERS_PASS),
       " for ", max_elems, " elements" KNORM "\n");
#endif
}

template <int MAX_K>
void KmerDHT<MAX_K>::set_pass(PASS_TYPE pass_type) {
  _num_kmers_counted = 0;
  this->pass_type = pass_type;
  switch (pass_type) {
    case READ_KMERS_PASS:
#ifdef ENABLE_KCOUNT_GPUS_HT
      ht_gpu_driver->set_pass(kcount_gpu::READ_KMERS_PASS);
#endif
      kmer_store.set_update_func(update_count);
      break;
    case CTG_KMERS_PASS:
#ifdef ENABLE_KCOUNT_GPUS_HT
      ht_gpu_driver->set_pass(kcount_gpu::CTG_KMERS_PASS);
#endif
      kmer_store.set_update_func(update_ctg_kmers_count);
      break;
  };
}

template <int MAX_K>
int KmerDHT<MAX_K>::get_minimizer_len() {
  return minimizer_len;
}

template <int MAX_K>
int64_t KmerDHT<MAX_K>::get_num_kmers(bool all) {
  if (!all)
    return reduce_one(kmers->size(), op_fast_add, 0).wait();
  else
    return reduce_all(kmers->size(), op_fast_add).wait();
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
bool KmerDHT<MAX_K>::kmer_exists(Kmer<MAX_K> kmer) {
  Kmer<MAX_K> kmer_rc = kmer.revcomp();
  if (kmer_rc < kmer) kmer.swap(kmer_rc);
  return rpc(
             get_kmer_target_rank(kmer, &kmer_rc),
             [](Kmer<MAX_K> kmer, dist_object<KmerMap> &kmers) -> bool {
               const auto it = kmers->find(kmer);
               if (it == kmers->end()) return false;
               return true;
             },
             kmer, kmers)
      .wait();
}
#endif

template <int MAX_K>
void KmerDHT<MAX_K>::add_kmer(Kmer<MAX_K> kmer, char left_ext, char right_ext, kmer_count_t count, int target_rank) {
  _num_kmers_counted++;
  if (!count) count = 1;
  if (target_rank == -1) {
    // get the lexicographically smallest
    Kmer<MAX_K> kmer_rc = kmer.revcomp();
    if (kmer_rc < kmer) {
      kmer.swap(kmer_rc);
      swap(left_ext, right_ext);
      left_ext = comp_nucleotide(left_ext);
      right_ext = comp_nucleotide(right_ext);
    }
    target_rank = get_kmer_target_rank(kmer, &kmer_rc);
  }
  KmerAndExt kmer_and_ext = {.kmer = kmer, .count = count, .left = left_ext, .right = right_ext};
  kmer_store.update(target_rank, kmer_and_ext);
  bytes_sent += sizeof(kmer_and_ext);
}

template <int MAX_K>
void KmerDHT<MAX_K>::flush_updates() {
  BarrierTimer timer(__FILEFUNC__);
  kmer_store.flush_updates();

#ifdef ENABLE_KCOUNT_GPUS_HT
  barrier();
  ht_gpu_driver->flush_inserts();
  // a bunch of stats about the hash table on the GPU
  auto insert_stats = ht_gpu_driver->get_stats(static_cast<kcount_gpu::PASS_TYPE>(pass_type));
  auto avg_num_gpu_calls = reduce_one(insert_stats.num_gpu_calls, op_fast_add, 0).wait() / rank_n();
  auto max_num_gpu_calls = reduce_one(insert_stats.num_gpu_calls, op_fast_max, 0).wait();
  SLOG(KLMAGENTA "Number of calls to hash table GPU driver: ", avg_num_gpu_calls, " avg, ", max_num_gpu_calls, " max" KNORM "\n");
  auto num_dropped_elems = reduce_one(insert_stats.dropped, op_fast_add, 0).wait();
  auto num_attempted_inserts = reduce_one(insert_stats.attempted, op_fast_add, 0).wait();
  auto capacity = ht_gpu_driver->get_capacity(static_cast<kcount_gpu::PASS_TYPE>(pass_type));
  if (num_dropped_elems)
    SWARN("GPU hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts), " elements; capacity ",
          reduce_one(capacity, op_fast_add, 0).wait(), " \n");
  double load = (double)(insert_stats.new_inserts) / capacity;
  double avg_load_factor = reduce_one(load, op_fast_add, 0).wait() / rank_n();
  double max_load_factor = reduce_one(load, op_fast_max, 0).wait();
  SLOG(KLMAGENTA "GPU kmer hash table load factor ", fixed, setprecision(3), avg_load_factor, " avg, ", max_load_factor,
       " max" KNORM "\n");
#else
  if (pass_type == READ_KMERS_PASS) {
    barrier();
    print_load_factor();
    barrier();
    purge_kmers(2);
    int64_t new_count = get_num_kmers();
    SLOG_VERBOSE("After purge of kmers < 2, there are ", new_count, " unique kmers\n");
  } else {
    purge_kmers(1);
  }
  barrier();
#endif
  auto avg_kmers_processed = reduce_one(_num_kmers_counted, op_fast_add, 0).wait() / rank_n();
  auto max_kmers_processed = reduce_one(_num_kmers_counted, op_fast_max, 0).wait();
  SLOG_VERBOSE("Avg kmers processed per rank ", avg_kmers_processed, " (balance ",
               (double)avg_kmers_processed / max_kmers_processed, ")\n");
}

template <int MAX_K>
void KmerDHT<MAX_K>::purge_kmers(int threshold) {
  BarrierTimer timer(__FILEFUNC__);
  auto num_prior_kmers = get_num_kmers();
  int64_t num_purged = 0;
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
#ifdef ENABLE_KCOUNT_GPUS_HT
  Timer insert_timer("gpu insert to cpu timer");
  insert_timer.start();
  ht_gpu_driver->done_all_inserts();
  auto num_entries = ht_gpu_driver->get_num_unique_elems();
  int64_t all_num_entries = reduce_one(ht_gpu_driver->get_num_unique_elems(), op_fast_add, 0).wait();
  auto num_purged = ht_gpu_driver->get_num_purged();
  auto all_num_purged = reduce_one(num_purged, op_fast_add, 0).wait();
  auto prepurge_num_entries = all_num_entries + all_num_purged;
  SLOG(KLMAGENTA "Purged ", perc_str(all_num_purged, prepurge_num_entries), " singleton kmers out of ", prepurge_num_entries,
       KNORM "\n");
  SLOG(KLMAGENTA, "GPU hash table contains ", all_num_entries, " valid entries" KNORM "\n");

  // add some space for the ctg kmers
  kmers->reserve(num_entries * 1.5);
  int num_empty_key_drops = 0;
  int sum_drop_depth = 0;
  while (true) {
    assert(HashTableGPUDriver<MAX_K>::get_N_LONGS() == Kmer<MAX_K>::get_N_LONGS());
    auto [kmer_array, kmer_counts_array] = ht_gpu_driver->get_next_entry();
    if (!kmer_array) break;
    // empty slot
    if (!kmer_counts_array->data[0]) continue;
    ExtCounts left_exts = {0}, right_exts = {0};
    left_exts.set(kmer_counts_array->data + 1);
    right_exts.set(kmer_counts_array->data + 5);
    KmerCounts kmer_counts = {
        .left_exts = left_exts,
        .right_exts = right_exts,
        .uutig_frag = nullptr,
        .count = static_cast<kmer_count_t>(
            min(static_cast<kcount_gpu::count_t>(std::numeric_limits<kmer_count_t>::max()), kmer_counts_array->data[0])),
        .left = 'X',
        .right = 'X',
        .from_ctg = false};
    // purge out low freq kmers
    if ((kmer_counts.count < 2) || (kmer_counts.left_exts.is_zero() && kmer_counts.right_exts.is_zero()))
      WARN("Found a kmer that should have been purged, count is ", kmer_counts.count);
    Kmer<MAX_K> kmer(reinterpret_cast<const uint64_t *>(kmer_array->longs));

    // FIXME: should only be done in debug mode
    const auto it = kmers->find(kmer);
    if (it != kmers->end())
      WARN("Found a duplicate kmer ", kmer.to_string(), " - shouldn't happen: existing count ", it->second.count, " new count ",
           kmer_counts.count);

    // drop all kmers that have values corresponding to the empty key as these are likely to be errors
    // we expect these to be very unlikely, requiring a full run of Ts for a full long, i.e. 32 Ts
    bool drop_entry = false;
    auto kmer_longs = kmer.get_longs();
    for (int j = 0; j < kmer.get_N_LONGS(); j++) {
      if (kmer_longs[j] == KEY_EMPTY) {
        num_empty_key_drops++;
        sum_drop_depth += kmer_counts.count;
        drop_entry = true;
        break;
      }
    }
    if (!drop_entry) kmers->insert({kmer, kmer_counts});
  }
  insert_timer.stop();
  auto all_num_empty_key_drops = reduce_one(num_empty_key_drops, op_fast_add, 0).wait();
  auto all_drop_depth = reduce_one(sum_drop_depth, op_fast_add, 0).wait();
  if (!rank_me() && all_num_empty_key_drops)
    SWARN("Dropped ", all_num_empty_key_drops, " kmer inserts because of empty key overlap, average depth ",
          (double)all_drop_depth / all_num_empty_key_drops);

  auto all_avg_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_add, 0).wait() / rank_n();
  auto all_max_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_max, 0).wait();
  SLOG(KLMAGENTA, "Inserting kmers from GPU to cpu hash table took ", all_avg_elapsed_time, " avg, ", all_max_elapsed_time,
       " max" KNORM "\n");
  auto all_kmers_size = reduce_one(kmers->size(), op_fast_add, 0).wait();
  if (kmers->size() != num_entries) WARN("kmers->size() is ", kmers->size(), " != ", num_entries, " num_entries");
  if (!rank_me() && all_kmers_size != all_num_entries)
    SWARN("CPU kmer counts not equal to gpu kmer counts: ", all_kmers_size, " != ", all_num_entries);

  double gpu_insert_time = 0, gpu_kernel_time = 0, gpu_memcpy_time = 0;
  ht_gpu_driver->get_elapsed_time(gpu_insert_time, gpu_kernel_time, gpu_memcpy_time);
  auto avg_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_add, 0).wait() / rank_n();
  auto max_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_max, 0).wait();
  auto avg_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_add, 0).wait() / rank_n();
  auto max_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_max, 0).wait();
  auto avg_gpu_memcpy_time = reduce_one(gpu_memcpy_time, op_fast_add, 0).wait() / rank_n();
  auto max_gpu_memcpy_time = reduce_one(gpu_memcpy_time, op_fast_max, 0).wait();
  stage_timers.kernel_kmer_analysis->inc_elapsed(max_gpu_kernel_time);
  SLOG(KLMAGENTA "Elapsed GPU time for kmer hash tables:" KNORM "\n");
  SLOG(KLMAGENTA "  insert: ", fixed, setprecision(3), avg_gpu_insert_time, " avg, ", max_gpu_insert_time, " max" KNORM "\n");
  SLOG(KLMAGENTA "  kernel: ", fixed, setprecision(3), avg_gpu_kernel_time, " avg, ", max_gpu_kernel_time, " max" KNORM "\n");
  SLOG(KLMAGENTA "  to cpu: ", fixed, setprecision(3), avg_gpu_memcpy_time, " avg, ", max_gpu_memcpy_time, " max" KNORM "\n");
#endif
}

template <int MAX_K>
void KmerDHT<MAX_K>::compute_kmer_exts() {
  // FIXME: if ctg kmers were processed, call gpu code that inserts ctg kmers into the read kmers hashtable
  if (using_ctg_kmers) ht_gpu_driver->done_ctg_kmer_inserts();
  insert_from_gpu_hashtable();
  // FIXME: this needs to be done in the GPU and then only the left and right selected exts need to be written back, further
  // reducing the copy back (i.e. instead of 9 uint32_t we have 2 bytes, e.g for k=33 we have 2*8+2=18 bytes instead of 2*8+9*4=52)
  BarrierTimer timer(__FILEFUNC__);
  for (auto &elem : *kmers) {
    auto kmer_counts = &elem.second;
    kmer_counts->left = kmer_counts->get_left_ext();
    kmer_counts->right = kmer_counts->get_right_ext();
  }
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
