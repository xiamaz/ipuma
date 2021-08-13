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

template <int MAX_K>
struct SeqBlockInserter<MAX_K>::SeqBlockInserterState {
  ParseAndPackGPUDriver *pnp_gpu_driver;
  int64_t num_pnp_gpu_waits = 0;
  int num_block_calls = 0;
  int64_t num_kmers = 0;
  int64_t bytes_kmers_sent = 0;
  int64_t bytes_supermers_sent = 0;
  string seq_block;
  vector<kmer_count_t> depth_block;

  SeqBlockInserterState()
      : pnp_gpu_driver(nullptr) {
    seq_block.reserve(KCOUNT_SEQ_BLOCK_SIZE);
    depth_block.reserve(KCOUNT_SEQ_BLOCK_SIZE);
  }
};

template <int MAX_K>
SeqBlockInserter<MAX_K>::SeqBlockInserter(int qual_offset, int minimizer_len) {
  if (gpu_utils::get_num_node_gpus() <= 0) DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  double init_time;
  state = new SeqBlockInserterState();
  state->pnp_gpu_driver = new ParseAndPackGPUDriver(rank_me(), rank_n(), qual_offset, Kmer<MAX_K>::get_k(),
                                                    Kmer<MAX_K>::get_N_LONGS(), minimizer_len, init_time);
  SLOG_GPU("Initialized PnP GPU driver in ", fixed, setprecision(3), init_time, " s\n");
}

template <int MAX_K>
SeqBlockInserter<MAX_K>::~SeqBlockInserter() {
  if (state->pnp_gpu_driver) {
    delete state->pnp_gpu_driver;
    state->pnp_gpu_driver = nullptr;
  }
  if (state) delete state;
}

template <int MAX_K>
static void process_block(SeqBlockInserter<MAX_K> *seq_block_inserter, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  unsigned int num_valid_kmers = 0;
  auto state = seq_block_inserter->state;
  bool from_ctgs = !state->depth_block.empty();
  state->num_block_calls++;
  if (!state->pnp_gpu_driver->process_seq_block(state->seq_block, num_valid_kmers))
    DIE("seq length is too high, ", state->seq_block.length(), " >= ", KCOUNT_SEQ_BLOCK_SIZE);
  state->bytes_kmers_sent += sizeof(KmerAndExt<MAX_K>) * num_valid_kmers;
  while (!state->pnp_gpu_driver->kernel_is_done()) {
    state->num_pnp_gpu_waits++;
    progress();
  }
  state->pnp_gpu_driver->pack_seq_block(state->seq_block);
  int num_targets = (int)state->pnp_gpu_driver->supermers.size();
  for (int i = 0; i < num_targets; i++) {
    auto target = state->pnp_gpu_driver->supermers[i].target;
    auto offset = state->pnp_gpu_driver->supermers[i].offset;
    auto len = state->pnp_gpu_driver->supermers[i].len;
    Supermer supermer;
    int packed_len = len / 2;
    if (offset % 2 || len % 2) packed_len++;
    supermer.seq = state->pnp_gpu_driver->packed_seqs.substr(offset / 2, packed_len);
    if (offset % 2) supermer.seq[0] &= 15;
    if ((offset + len) % 2) supermer.seq[supermer.seq.length() - 1] &= 240;
    supermer.count = (from_ctgs ? state->depth_block[offset + 1] : (kmer_count_t)1);
    state->bytes_supermers_sent += supermer.get_bytes();
    kmer_dht->add_supermer(supermer, target);
    state->num_kmers += (2 * supermer.seq.length() - Kmer<MAX_K>::get_k());
    progress();
  }
  state->seq_block.clear();
  state->depth_block.clear();
}

template <int MAX_K>
void SeqBlockInserter<MAX_K>::process_seq(string &seq, kmer_count_t depth, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  if (seq.length() >= KCOUNT_SEQ_BLOCK_SIZE)
    DIE("Oh dear, my laziness is revealed: the ctg seq is too long ", seq.length(), " for this GPU implementation ",
        KCOUNT_SEQ_BLOCK_SIZE);
  if (state->seq_block.length() + 1 + seq.length() >= KCOUNT_SEQ_BLOCK_SIZE) {
    process_block(this, kmer_dht);
  } else {
    state->seq_block += seq;
    state->seq_block += "_";
    if (depth) state->depth_block.insert(state->depth_block.end(), seq.length() + 1, depth);
  }
}

template <int MAX_K>
void SeqBlockInserter<MAX_K>::done_processing(dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  if (!state->seq_block.empty()) process_block(this, kmer_dht);
  SLOG_GPU("GPU Parse-N-Pack stats:\n");
  SLOG_GPU("  number of calls to progress (on rank 0): ", state->num_pnp_gpu_waits, "\n");
  SLOG_GPU("  number of calls to PnP GPU kernel (on rank 0): ", state->num_block_calls, "\n");
  auto [gpu_time_tot, gpu_time_kernel] = state->pnp_gpu_driver->get_elapsed_times();
  SLOG_GPU("  elapsed times: ", fixed, setprecision(3), " total ", gpu_time_tot, ", kernel ", gpu_time_kernel, "\n");
  auto tot_supermers_bytes_sent = reduce_one(state->bytes_supermers_sent, op_fast_add, 0).wait();
  auto tot_kmers_bytes_sent = reduce_one(state->bytes_kmers_sent, op_fast_add, 0).wait();
  SLOG_VERBOSE("Total bytes sent in compressed supermers ", get_size_str(tot_supermers_bytes_sent), " (compression is ", fixed,
               setprecision(3), (double)tot_kmers_bytes_sent / tot_supermers_bytes_sent, " over kmers)\n");
  auto all_num_kmers = reduce_one(state->num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_kmers, " kmers\n");
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
    SLOG_GPU("GPU hash table stats for read kmers pass:\n");
  else
    SLOG_GPU("GPU hash table stats for ctg kmers pass:\n");
  SLOG_GPU("  number of calls to hash table GPU driver: ", avg_num_gpu_calls, " avg, ", max_num_gpu_calls, " max\n");
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
      SLOG_GPU("  failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts), " elements; capacity ", all_capacity,
               "\n");
  }
  uint64_t key_empty_overlaps = reduce_one((uint64_t)insert_stats.key_empty_overlaps, op_fast_add, 0).wait();
  if (key_empty_overlaps) {
    if (key_empty_overlaps > num_attempted_inserts / 10000)
      SWARN("GPU hash table: dropped ", perc_str(key_empty_overlaps, num_attempted_inserts),
            " kmers with longs equal to KEY_EMPTY");
    else
      SLOG_GPU("  dropped ", perc_str(key_empty_overlaps, num_attempted_inserts), " kmers with longs equal to KEY_EMPTY\n");
  }
  double load = (double)(insert_stats.new_inserts) / capacity;
  double avg_load_factor = reduce_one(load, op_fast_add, 0).wait() / rank_n();
  double max_load_factor = reduce_one(load, op_fast_max, 0).wait();
  SLOG_GPU("  load factor ", fixed, setprecision(3), avg_load_factor, " avg, ", max_load_factor, " max\n");
  SLOG_GPU("  final size per rank is ", insert_stats.new_inserts, " entries\n");
}

template <int MAX_K>
void HashTableInserter<MAX_K>::get_elapsed_time(double &insert_time, double &kernel_time) {
  state->ht_gpu_driver.get_elapsed_time(insert_time, kernel_time);
}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_into_local_hashtable(dist_object<KmerMap<MAX_K>> &local_kmers) {
  barrier();
  Timer insert_timer("gpu insert to cpu timer");
  insert_timer.start();
  if (state->ht_gpu_driver.pass_type == CTG_KMERS_PASS) {
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
  barrier();
  int num_dropped = 0, num_entries = 0, num_purged = 0;
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
  barrier();

  // add some space for the ctg kmers
  local_kmers->reserve(num_entries * 1.5);
  uint64_t invalid = 0;
  while (true) {
    auto [kmer_array, count_exts] = state->ht_gpu_driver.get_next_entry();
    if (!kmer_array) break;
    // empty slot
    if (!count_exts->count) continue;
    if ((char)count_exts->left == 'X' && (char)count_exts->right == 'X') {
      // these are eliminated during purging in CPU version
      invalid++;
      continue;
    }
    if ((count_exts->count < 2)) {
      WARN("Found a kmer that should have been purged, count is ", count_exts->count);
      invalid++;
      continue;
    }
    KmerCounts kmer_counts = {.uutig_frag = nullptr,
                              .count = static_cast<kmer_count_t>(min(count_exts->count, static_cast<count_t>(UINT16_MAX))),
                              .left = (char)count_exts->left,
                              .right = (char)count_exts->right};
    Kmer<MAX_K> kmer(reinterpret_cast<const uint64_t *>(kmer_array->longs));
    const auto it = local_kmers->find(kmer);
    if (it != local_kmers->end())
      WARN("Found a duplicate kmer ", kmer.to_string(), " - shouldn't happen: existing count ", it->second.count, " new count ",
           kmer_counts.count);
    local_kmers->insert({kmer, kmer_counts});
  }
  insert_timer.stop();

  auto all_avg_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_add, 0).wait() / rank_n();
  auto all_max_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_max, 0).wait();
  SLOG_GPU("Inserting kmers from GPU to cpu hash table took ", all_avg_elapsed_time, " avg, ", all_max_elapsed_time, " max\n");
  auto all_kmers_size = reduce_one((uint64_t)local_kmers->size(), op_fast_add, 0).wait();
  if (local_kmers->size() != (num_entries - invalid))
    WARN("kmers->size() is ", local_kmers->size(), " != ", (num_entries - invalid), " num_entries");
  auto all_invalid = reduce_one((uint64_t)invalid, op_fast_add, 0).wait();
  if (all_kmers_size != all_num_entries - all_invalid)
    SWARN("CPU kmer counts not equal to gpu kmer counts: ", all_kmers_size, " != ", (all_num_entries - all_invalid),
          " all_num_entries: ", all_num_entries, " all_invalid: ", all_invalid);
  double gpu_insert_time = 0, gpu_kernel_time = 0;
  state->ht_gpu_driver.get_elapsed_time(gpu_insert_time, gpu_kernel_time);
  auto avg_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_add, 0).wait() / rank_n();
  auto max_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_max, 0).wait();
  auto avg_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_add, 0).wait() / rank_n();
  auto max_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_max, 0).wait();
  SLOG_GPU("Elapsed GPU time for kmer hash tables:\n");
  SLOG_GPU("  insert: ", fixed, setprecision(3), avg_gpu_insert_time, " avg, ", max_gpu_insert_time, " max\n");
  SLOG_GPU("  kernel: ", fixed, setprecision(3), avg_gpu_kernel_time, " avg, ", max_gpu_kernel_time, " max\n");
  barrier();
}

#define seq_block_inserter_K(KMER_LEN) template struct SeqBlockInserter<KMER_LEN>;
#define HASH_TABLE_INSERTER_K(KMER_LEN) template struct HashTableInserter<KMER_LEN>;

seq_block_inserter_K(32);
HASH_TABLE_INSERTER_K(32);
#if MAX_BUILD_KMER >= 64
seq_block_inserter_K(64);
HASH_TABLE_INSERTER_K(64);
#endif
#if MAX_BUILD_KMER >= 96
seq_block_inserter_K(96);
HASH_TABLE_INSERTER_K(96);
#endif
#if MAX_BUILD_KMER >= 128
seq_block_inserter_K(128);
HASH_TABLE_INSERTER_K(128);
#endif
#if MAX_BUILD_KMER >= 160
seq_block_inserter_K(160);
HASH_TABLE_INSERTER_K(160);
#endif
#undef seq_block_inserter_K
#undef HASH_TABLE_INSERTER_K
