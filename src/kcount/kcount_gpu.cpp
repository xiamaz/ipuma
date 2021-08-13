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
#include "kcount.hpp"
#include "kmer_dht.hpp"

#include "gpu-utils/gpu_utils.hpp"
#include "kcount-gpu/parse_and_pack.hpp"
#include "kcount-gpu/gpu_hash_table.hpp"

#define SLOG_GPU(...) SLOG(KLMAGENTA, __VA_ARGS__, KNORM)
//#define SLOG_GPU SLOG_VERBOSE

using namespace std;
using namespace upcxx_utils;
using namespace upcxx;
using namespace kcount_gpu;

static int64_t num_pnp_gpu_waits = 0;
static int64_t bytes_kmers_sent = 0;
static int64_t bytes_supermers_sent = 0;
static int64_t num_kmers = 0;
static int64_t num_Ns = 0;
static int num_block_calls = 0;

static ParseAndPackGPUDriver *pnp_gpu_driver;

template <int MAX_K>
BlockInserter<MAX_K>::BlockInserter(int qual_offset, int minimizer_len) {
  if (gpu_utils::get_num_node_gpus() <= 0) DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  double init_time;
  num_pnp_gpu_waits = 0;
  bytes_kmers_sent = 0;
  bytes_supermers_sent = 0;
  num_kmers = 0;
  num_Ns = 0;
  num_block_calls = 0;
  pnp_gpu_driver = new ParseAndPackGPUDriver(rank_me(), rank_n(), qual_offset, Kmer<MAX_K>::get_k(), Kmer<MAX_K>::get_N_LONGS(),
                                             minimizer_len, init_time);
  SLOG_GPU("Initialized PnP GPU driver in ", fixed, setprecision(3), init_time, " s\n");
}

template <int MAX_K>
BlockInserter<MAX_K>::~BlockInserter() {
  SLOG_GPU("Number of calls to progress while PnP GPU driver was running on rank 0: ", num_pnp_gpu_waits, "\n");
  SLOG_GPU("Number of calls to PnP GPU kernel for rank 0: ", num_block_calls, "\n");
  auto [gpu_time_tot, gpu_time_kernel] = pnp_gpu_driver->get_elapsed_times();
  SLOG_GPU("Elapsed times for PnP GPU: ", fixed, setprecision(3), " total ", gpu_time_tot, ", kernel ", gpu_time_kernel, "\n");
  delete pnp_gpu_driver;
  pnp_gpu_driver = nullptr;
  auto tot_supermers_bytes_sent = reduce_one(bytes_supermers_sent, op_fast_add, 0).wait();
  auto tot_kmers_bytes_sent = reduce_one(bytes_kmers_sent, op_fast_add, 0).wait();
  SLOG_VERBOSE("Total bytes sent in compressed supermers ", get_size_str(tot_supermers_bytes_sent),
               " and what would have been sent in kmers ", get_size_str(tot_kmers_bytes_sent), " compression is ", fixed,
               setprecision(3), (double)tot_kmers_bytes_sent / tot_supermers_bytes_sent, "\n");
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_kmers, " kmers\n");
  auto all_num_Ns = reduce_one(num_Ns, op_fast_add, 0).wait();
  if (all_num_Ns) SLOG_VERBOSE("Found ", perc_str(all_num_Ns, all_num_kmers), " kmers with Ns\n");
}

template <int MAX_K>
void BlockInserter<MAX_K>::process_block(unsigned kmer_len, string &seq_block, const vector<kmer_count_t> &depth_block,
                                         dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  num_block_calls++;
  bool from_ctgs = !depth_block.empty();
  unsigned int num_valid_kmers = 0;
  if (!pnp_gpu_driver->process_seq_block(seq_block, num_valid_kmers))
    DIE("seq length is too high, ", seq_block.length(), " >= ", KCOUNT_SEQ_BLOCK_SIZE);
  bytes_kmers_sent += sizeof(KmerAndExt<MAX_K>) * num_valid_kmers;
  while (!pnp_gpu_driver->kernel_is_done()) {
    num_pnp_gpu_waits++;
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
    num_kmers += supermer.seq.length() - Kmer<MAX_K>::get_k();
    progress();
  }
}

template <int MAX_K>
struct HashTableInserter<MAX_K>::HashTableInserterState {
  HashTableGPUDriver<MAX_K> ht_gpu_driver;

  HashTableInserterState()
      : ht_gpu_driver({}) {}
};

template <int MAX_K>
HashTableInserter<MAX_K>::HashTableInserter() {}

template <int MAX_K>
HashTableInserter<MAX_K>::~HashTableInserter() {
  if (state != nullptr) delete state;
}

template <int MAX_K>
void HashTableInserter<MAX_K>::init(int max_elems) {
  state = new HashTableInserterState();
  double init_time;
  // calculate total slots for hash table. Reserve space for parse and pack
  int bytes_for_pnp = KCOUNT_SEQ_BLOCK_SIZE * (2 + Kmer<MAX_K>::get_N_LONGS() * sizeof(uint64_t) + sizeof(int));
  size_t gpu_bytes_reqd;
  int max_dev_id = reduce_one(gpu_utils::get_gpu_device_pci_id(), op_fast_max, 0).wait();
  auto gpu_avail_mem = (gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n() - bytes_for_pnp) * 0.95;
  auto gpu_tot_mem = gpu_utils::get_tot_gpu_mem() * max_dev_id / upcxx::local_team().rank_n() - bytes_for_pnp;
  SLOG_GPU("Available GPU memory per rank for kmers hash table is ", get_size_str(gpu_avail_mem), " out of a max of ",
           get_size_str(gpu_tot_mem), "\n");
  assert(state != nullptr);
  state->ht_gpu_driver.init(rank_me(), rank_n(), Kmer<MAX_K>::get_k(), max_elems, gpu_avail_mem, init_time, gpu_bytes_reqd);
  auto capacity = state->ht_gpu_driver.get_capacity();
  SLOG_GPU("GPU read kmers hash table has capacity per rank of ", capacity, " for ", (int64_t)max_elems, " elements\n");
  if (capacity < max_elems * 0.8)
    SLOG_VERBOSE("GPU read kmers hash table has less than requested capacity: ", perc_str(capacity, max_elems),
                 "; full capacity requires ", get_size_str(gpu_bytes_reqd), " memory on GPU but only have ",
                 get_size_str(gpu_avail_mem));
  SLOG_GPU("Initialized hash table GPU driver in ", fixed, setprecision(3), init_time, " s\n");
  auto gpu_free_mem = gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n();
  SLOG_GPU("After initializing GPU hash table, there is ", get_size_str(gpu_free_mem), " memory available per rank, with ",
           get_size_str(bytes_for_pnp), " reserved for PnP and ", get_size_str(gpu_bytes_reqd),
           " reserved for the kmer hash table\n");
}

template <int MAX_K>
void HashTableInserter<MAX_K>::init_ctg_kmers(int max_elems) {
  assert(state != nullptr);
  int max_dev_id = reduce_one(gpu_utils::get_gpu_device_pci_id(), op_fast_max, 0).wait();
  // we don't need to reserve space for either pnp or the read kmers because those have already reduced the gpu_avail_mem
  auto gpu_avail_mem = (gpu_utils::get_free_gpu_mem() * max_dev_id / upcxx::local_team().rank_n()) * 0.95;
  auto gpu_tot_mem = gpu_utils::get_tot_gpu_mem() * max_dev_id / upcxx::local_team().rank_n();
  SLOG_GPU("Available GPU memory per rank for ctg kmers hash table is ", get_size_str(gpu_avail_mem), " out of a max of ",
           get_size_str(gpu_tot_mem), "\n");
  state->ht_gpu_driver.init_ctg_kmers(max_elems, gpu_avail_mem);
  SLOG_GPU("GPU ctg kmers hash table has capacity per rank of ", state->ht_gpu_driver.get_capacity(), " for ", fixed, max_elems,
           " elements\n");
}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_supermer(const std::string &supermer_seq, kmer_count_t supermer_count) {
  assert(state != nullptr);
  state->ht_gpu_driver.insert_supermer(supermer_seq, supermer_count);
}

template <int MAX_K>
void HashTableInserter<MAX_K>::flush_inserts() {
  state->ht_gpu_driver.flush_inserts();
  auto avg_num_gpu_calls = reduce_one(state->ht_gpu_driver.get_num_gpu_calls(), op_fast_add, 0).wait() / rank_n();
  auto max_num_gpu_calls = reduce_one(state->ht_gpu_driver.get_num_gpu_calls(), op_fast_max, 0).wait();
  if (state->ht_gpu_driver.pass_type == kcount_gpu::READ_KMERS_PASS)
    SLOG_GPU("GPU stats for read kmers pass:\n");
  else
    SLOG_GPU("GPU stats for ctg kmers pass:\n");
  SLOG_GPU("   Number of calls to hash table GPU driver: ", avg_num_gpu_calls, " avg, ", max_num_gpu_calls, " max\n");
  // a bunch of stats about the hash table on the GPU
  auto insert_stats = state->ht_gpu_driver.get_stats();
  uint64_t num_dropped_elems = reduce_one((uint64_t)insert_stats.dropped, op_fast_add, 0).wait();
  uint64_t num_attempted_inserts = reduce_one((uint64_t)insert_stats.attempted, op_fast_add, 0).wait();
  uint64_t capacity = state->ht_gpu_driver.get_capacity();
  uint64_t all_capacity = reduce_one(capacity, op_fast_add, 0).wait();
  if (num_dropped_elems) {
    if (num_dropped_elems > num_attempted_inserts / 10000)
      SWARN("GPU hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts), " elements; capacity ",
            all_capacity);
    else
      SLOG_GPU("   Failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts), " elements; capacity ", all_capacity,
               "\n");
  }
  uint64_t key_empty_overlaps = reduce_one((uint64_t)insert_stats.key_empty_overlaps, op_fast_add, 0).wait();
  if (key_empty_overlaps) {
    if (key_empty_overlaps > num_attempted_inserts / 10000)
      SWARN("GPU hash table: dropped ", perc_str(key_empty_overlaps, num_attempted_inserts),
            " kmers with longs equal to KEY_EMPTY");
    else
      SLOG_GPU("   Dropped ", perc_str(key_empty_overlaps, num_attempted_inserts), " kmers with longs equal to KEY_EMPTY\n");
  }
  double load = (double)(insert_stats.new_inserts) / capacity;
  double avg_load_factor = reduce_one(load, op_fast_add, 0).wait() / rank_n();
  double max_load_factor = reduce_one(load, op_fast_max, 0).wait();
  SLOG_GPU("   Load factor ", fixed, setprecision(3), avg_load_factor, " avg, ", max_load_factor, " max\n");
  SLOG_GPU("   Final size per rank is ", insert_stats.new_inserts, " entries\n");
}

template <int MAX_K>
void HashTableInserter<MAX_K>::done_ctg_kmer_inserts() {
  barrier();
  int attempted_inserts = 0, dropped_inserts = 0, new_inserts = 0;
  state->ht_gpu_driver.done_ctg_kmer_inserts(attempted_inserts, dropped_inserts, new_inserts);
  barrier();
  auto num_dropped_elems = reduce_one((uint64_t)dropped_inserts, op_fast_add, 0).wait();
  auto num_attempted_inserts = reduce_one((uint64_t)attempted_inserts, op_fast_add, 0).wait();
  auto num_new_inserts = reduce_one((uint64_t)new_inserts, op_fast_add, 0).wait();
  SLOG_GPU("GPU ctg kmers hash table: inserted ", num_new_inserts, " new elements into read kmers hash table\n");
  auto all_capacity = reduce_one((uint64_t)state->ht_gpu_driver.get_capacity(), op_fast_add, 0).wait();
  if (num_dropped_elems) {
    if (num_dropped_elems > num_attempted_inserts / 10000)
      SWARN("GPU read kmers hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts),
            " ctg kmers; total capacity ", all_capacity);
    else
      SLOG_VERBOSE("GPU read kmers hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts),
                   " ctg kmers; total capacity ", all_capacity, "\n");
  }
}

template <int MAX_K>
int HashTableInserter<MAX_K>::done_all_inserts() {
  int num_dropped = 0, num_entries = 0, num_purged = 0;
  barrier();
  state->ht_gpu_driver.done_all_inserts(num_dropped, num_entries, num_purged);
  barrier();
  if (num_dropped)
    WARN("GPU dropped ", num_dropped, " entries out of ", num_entries, " when compacting to output hash table" KNORM "\n");
  auto all_capacity = reduce_one((uint64_t)state->ht_gpu_driver.get_capacity(), op_fast_add, 0).wait();
  auto all_num_purged = reduce_one((uint64_t)num_purged, op_fast_add, 0).wait();
  auto all_num_entries = reduce_one((uint64_t)num_entries, op_fast_add, 0).wait();
  auto prepurge_num_entries = all_num_entries + all_num_purged;
  SLOG_GPU("GPU read kmers hash table: purged ", perc_str(all_num_purged, prepurge_num_entries), " singleton kmers out of ",
           prepurge_num_entries, "\n");
  SLOG_GPU("GPU hash table final size is ", all_num_entries, " entries and final load factor is ",
           ((double)all_num_entries / all_capacity), "\n");
  return num_entries;
}

template <int MAX_K>
std::tuple<bool, Kmer<MAX_K>, KmerCounts> HashTableInserter<MAX_K>::get_next_entry() {
  auto [kmer_array, count_exts] = state->ht_gpu_driver.get_next_entry();
  if (!kmer_array) return {true, {}, {}};
  KmerCounts kmer_counts = {.left_exts = {0},
                            .right_exts = {0},
                            .uutig_frag = nullptr,
                            .count = static_cast<kmer_count_t>(min(count_exts->count, static_cast<count_t>(UINT16_MAX))),
                            .left = (char)count_exts->left,
                            .right = (char)count_exts->right,
                            .from_ctg = false};
  Kmer<MAX_K> kmer(reinterpret_cast<const uint64_t *>(kmer_array->longs));
  return {false, kmer, kmer_counts};
}

template <int MAX_K>
void HashTableInserter<MAX_K>::get_elapsed_time(double &insert_time, double &kernel_time) {
  state->ht_gpu_driver.get_elapsed_time(insert_time, kernel_time);
}

template <int MAX_K>
int64_t HashTableInserter<MAX_K>::get_capacity() {
  return state->ht_gpu_driver.get_capacity();
}

#define BLOCK_INSERTER_K(KMER_LEN) template struct BlockInserter<KMER_LEN>;
#define HASH_TABLE_INSERTER_K(KMER_LEN) template struct HashTableInserter<KMER_LEN>;

BLOCK_INSERTER_K(32);
HASH_TABLE_INSERTER_K(32);
#if MAX_BUILD_KMER >= 64
BLOCK_INSERTER_K(64);
HASH_TABLE_INSERTER_K(64);
#endif
#if MAX_BUILD_KMER >= 96
BLOCK_INSERTER_K(96);
HASH_TABLE_INSERTER_K(96);
#endif
#if MAX_BUILD_KMER >= 128
BLOCK_INSERTER_K(128);
HASH_TABLE_INSERTER_K(128);
#endif
#if MAX_BUILD_KMER >= 160
BLOCK_INSERTER_K(160);
HASH_TABLE_INSERTER_K(160);
#endif
#undef BLOCK_INSERTER_K
#undef HASH_TABLE_INSERTER_K
