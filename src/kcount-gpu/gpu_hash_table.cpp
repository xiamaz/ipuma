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

#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "upcxx_utils/colors.h"
#include "gpu_common.hpp"
#include "gpu_hash_table.hpp"

#include "gpu_hash_funcs.cpp"

using namespace std;
using namespace gpu_common;
using namespace kcount_gpu;

template <int MAX_K>
__device__ bool kmers_equal(const KmerArray<MAX_K> &kmer1, const KmerArray<MAX_K> &kmer2) {
  int n_longs = kmer1.N_LONGS;
  for (int i = 0; i < n_longs; i++) {
    if (kmer1.longs[i] != kmer2.longs[i]) return false;
  }
  return true;
}

template <int MAX_K>
__device__ size_t kmer_hash(const KmerArray<MAX_K> &kmer) {
  return gpu_murmurhash3_64(reinterpret_cast<const void *>(kmer.longs), kmer.N_LONGS * sizeof(uint64_t));
}

__device__ int8_t get_ext(count_t *ext_counts, int pos, int8_t *ext_map) {
  count_t top_count = 0, runner_up_count = 0;
  int top_ext_pos = 0;
  count_t kmer_count = ext_counts[0];
  for (int i = pos; i < pos + 4; i++) {
    if (ext_counts[i] >= top_count) {
      runner_up_count = top_count;
      top_count = ext_counts[i];
      top_ext_pos = i;
    } else if (ext_counts[i] > runner_up_count) {
      runner_up_count = ext_counts[i];
    }
  }
  // set dynamic_min_depth to 1.0 for single depth data (non-metagenomes)
  int dmin_dyn = (1.0 - DYN_MIN_DEPTH) * kmer_count;
  // if (dmin_dyn < _dmin_thres) dmin_dyn = _dmin_thres;
  if (dmin_dyn < 2.0) dmin_dyn = 2.0;
  if (top_count < dmin_dyn) return 'X';
  if (runner_up_count >= dmin_dyn) return 'F';
  return ext_map[top_ext_pos - pos];
}

__device__ bool ext_conflict(count_t *ext_counts, int start_idx) {
  int idx = -1;
  for (int i = start_idx; i < start_idx + 4; i++) {
    if (ext_counts[i]) {
      // conflict
      if (idx != -1) return true;
      idx = i;
    }
  }
  return false;
}

template <int MAX_K>
__global__ void gpu_merge_ctg_kmers(KmerCountsMap<MAX_K> read_kmers, const KmerCountsMap<MAX_K> ctg_kmers,
                                    unsigned int *insert_counts) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  int8_t ext_map[4] = {'A', 'C', 'G', 'T'};
  int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  int attempted_inserts = 0;
  int dropped_inserts = 0;
  int new_inserts = 0;
  if (threadid < ctg_kmers.capacity) {
    count_t *counts = ctg_kmers.vals[threadid].data;
    if (counts[0] && !ext_conflict(counts, 1) && !ext_conflict(counts, 5)) {
      KmerArray<MAX_K> kmer = ctg_kmers.keys[threadid];
      uint64_t slot = kmer_hash(kmer) % read_kmers.capacity;
      auto start_slot = slot;
      attempted_inserts++;
      const int MAX_PROBE = (read_kmers.capacity < 200 ? read_kmers.capacity : 200);
      int j;
      for (j = 0; j < MAX_PROBE; j++) {
        uint64_t old_key =
            atomicCAS(reinterpret_cast<unsigned long long *>(&(read_kmers.keys[slot].longs[0])), KEY_EMPTY, kmer.longs[0]);
        if (old_key == KEY_EMPTY) {
          // now all others should be empty
          for (int long_i = 1; long_i < N_LONGS; long_i++) {
            old_key = atomicCAS(reinterpret_cast<unsigned long long *>(&(read_kmers.keys[slot].longs[long_i])), KEY_EMPTY,
                                kmer.longs[long_i]);
            if (old_key != KEY_EMPTY) {
              printf("ERROR: old key is not KEY_EMPTY!! Why?\n");
              break;
            }
          }
          new_inserts++;
          // always add it when there is no existing kmer from the reads
          memcpy(read_kmers.vals[slot].data, counts, sizeof(CountsArray));
          break;
        } else if (old_key == kmer.longs[0]) {
          // now check to see if this is the same key, i.e. same in every position as this key
          bool found_slot = true;
          for (int long_i = 1; long_i < N_LONGS; long_i++) {
            // check the value atomically by adding nothing
            old_key = atomicAdd(reinterpret_cast<unsigned long long *>(&(read_kmers.keys[slot].longs[long_i])), 0);
            if (old_key != kmer.longs[long_i]) {
              found_slot = false;
              break;
            }
          }
          if (found_slot) {
            // existing kmer from reads - only replace if the kmer is non-UU
            int8_t left_ext = get_ext(read_kmers.vals[slot].data, 1, ext_map);
            int8_t right_ext = get_ext(read_kmers.vals[slot].data, 5, ext_map);
            if (left_ext == 'X' || left_ext == 'F' || right_ext == 'X' || right_ext == 'F')
              memcpy(read_kmers.vals[slot].data, counts, sizeof(CountsArray));
            break;
          }
        }
        // quadratic probing - worse cache but reduced clustering
        slot = (start_slot + (j + 1) * (j + 1)) % read_kmers.capacity;
      }
      // this entry didn't get inserted because we ran out of probing time (and probably space)
      if (j == MAX_PROBE) dropped_inserts++;
    }
  }
  reduce(attempted_inserts, ctg_kmers.capacity, &(insert_counts[0]));
  reduce(dropped_inserts, ctg_kmers.capacity, &(insert_counts[1]));
  reduce(new_inserts, ctg_kmers.capacity, &(insert_counts[2]));
}

template <int MAX_K>
__global__ void gpu_compact_ht(KmerCountsMap<MAX_K> elems, KmerExtsMap<MAX_K> compact_elems, unsigned int *elem_counts) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  const int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  int dropped_inserts = 0;
  int unique_inserts = 0;
  int8_t ext_map[4] = {'A', 'C', 'G', 'T'};
  if (threadid < elems.capacity) {
    if (elems.vals[threadid].data[0]) {
      KmerArray<MAX_K> kmer = elems.keys[threadid];
      count_t *counts = elems.vals[threadid].data;
      uint64_t slot = kmer_hash(kmer) % compact_elems.capacity;
      auto start_slot = slot;
      // we set a constraint on the max probe to track whether we are getting excessive collisions and need a bigger default
      // compact table
      const int MAX_PROBE = (compact_elems.capacity < 200 ? compact_elems.capacity : 200);
      // look for empty slot in compact hash table
      int j;
      for (j = 0; j < MAX_PROBE; j++) {
        uint64_t old_key =
            atomicCAS(reinterpret_cast<unsigned long long *>(&(compact_elems.keys[slot].longs[0])), KEY_EMPTY, kmer.longs[0]);
        if (old_key == KEY_EMPTY) {
          // found empty slot - there will be no duplicate keys since we're copying across from another hash table
          unique_inserts++;
          for (int long_i = 1; long_i < N_LONGS; long_i++) {
#ifdef DEBUG
            // Use these for debugging
            // old_key = atomicCAS(&(compact_elems.keys[slot].longs[long_i]), KEY_EMPTY, kmer.longs[long_i]);
            // if (old_key != KEY_EMPTY) printf("ERROR: key is not empty when setting in compact ht!\n");
#endif
            compact_elems.keys[slot].longs[long_i] = kmer.longs[long_i];
          }
          // compute exts
          int8_t left_ext = get_ext(counts, 1, ext_map);
          int8_t right_ext = get_ext(counts, 5, ext_map);
          compact_elems.vals[slot].count = counts[0];
          compact_elems.vals[slot].left = left_ext;
          compact_elems.vals[slot].right = right_ext;
          break;
        }
        // quadratic probing - worse cache but reduced clustering
        slot = (start_slot + (j + 1) * (j + 1)) % compact_elems.capacity;
      }
      if (j == MAX_PROBE) dropped_inserts++;
    }
  }
  reduce(dropped_inserts, compact_elems.capacity, &(elem_counts[0]));
  reduce(unique_inserts, compact_elems.capacity, &(elem_counts[1]));
}

template <int MAX_K>
__global__ void gpu_purge_invalid(KmerCountsMap<MAX_K> elems, unsigned int *elem_counts) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  int num_purged = 0;
  int num_elems = 0;
  if (threadid < elems.capacity) {
    if (elems.vals[threadid].data[0]) {
      int ext_sum = 0;
      for (int j = 1; j < 9; j++) ext_sum += elems.vals[threadid].data[j];
      if (elems.vals[threadid].data[0] < 2 || !ext_sum) {
        memset(elems.vals[threadid].data, 0, sizeof(CountsArray));
        memset(elems.keys[threadid].longs, KEY_EMPTY_BYTE, N_LONGS * sizeof(uint64_t));
        num_purged++;
      } else {
        num_elems++;
      }
    }
  }
  reduce(num_purged, elems.capacity, &(elem_counts[0]));
  reduce(num_elems, elems.capacity, &(elem_counts[1]));
}

static __constant__ char to_base[] = {'0', 'a', 'c', 'g', 't', 'A', 'C', 'G', 'T', 'N'};

inline __device__ char to_base_func(int index, int pp) {
  if (index > 9) {
    printf("ERROR: index out of range for to_base: %d, packed seq pos %d\n", index, pp);
    return 0;
  }
  if (index == 0) return '_';
  return to_base[index];
}

__global__ void gpu_unpack_supermer_block(SupermerBuff unpacked_supermer_buff, SupermerBuff packed_supermer_buff, int buff_len) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadid >= buff_len) return;
  uint8_t packed = packed_supermer_buff.seqs[threadid];
  if (packed == '_') return;
  uint8_t left_side = (packed & 240) >> 4;
  unpacked_supermer_buff.seqs[threadid * 2] = to_base_func(left_side, packed);
  if (packed_supermer_buff.counts) unpacked_supermer_buff.counts[threadid * 2] = packed_supermer_buff.counts[threadid];
  uint8_t right_side = packed & 15;
  unpacked_supermer_buff.seqs[threadid * 2 + 1] = to_base_func(right_side, packed);
  if (packed_supermer_buff.counts) unpacked_supermer_buff.counts[threadid * 2 + 1] = packed_supermer_buff.counts[threadid];
}

inline __device__ bool is_valid_base(char base) {
  return (base == 'A' || base == 'C' || base == 'G' || base == 'T' || base == '0' || base == 'N');
}

inline __device__ bool bad_qual(char base) { return (base == 'a' || base == 'c' || base == 'g' || base == 't'); }

template <int MAX_K>
__device__ bool get_kmer_from_supermer(SupermerBuff supermer_buff, uint32_t buff_len, int kmer_len, uint64_t *kmer, char &left_ext,
                                       char &right_ext, count_t &count) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_kmers = buff_len - kmer_len + 1;
  if (threadid >= num_kmers) return false;
  const int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  if (!pack_seq_to_kmer(&(supermer_buff.seqs[threadid]), kmer_len, N_LONGS, kmer)) return false;
  if (threadid + kmer_len >= buff_len) return false;  // printf("out of bounds %d >= %d\n", threadid + kmer_len, buff_len);
  left_ext = supermer_buff.seqs[threadid - 1];
  right_ext = supermer_buff.seqs[threadid + kmer_len];
  if (left_ext == '_' || right_ext == '_') return false;
  if (!left_ext || !right_ext) return false;
  if (supermer_buff.counts) {
    count = supermer_buff.counts[threadid];
  } else {
    count = 1;
    if (bad_qual(left_ext)) left_ext = '0';
    if (bad_qual(right_ext)) right_ext = '0';
  }
  if (!is_valid_base(left_ext)) {
    printf("ERROR: threadid %d, invalid char for left nucleotide %d\n", threadid, (uint8_t)left_ext);
    return false;
  }
  if (!is_valid_base(right_ext)) {
    printf("ERROR: threadid %d, invalid char for right nucleotide %d\n", threadid, (uint8_t)right_ext);
    return false;
  }
  uint64_t kmer_rc[N_LONGS];
  revcomp(kmer, kmer_rc, kmer_len, N_LONGS);
  for (int l = 0; l < N_LONGS; l++) {
    if (kmer_rc[l] == kmer[l]) continue;
    if (kmer_rc[l] < kmer[l]) {
      // swap
      char tmp = left_ext;
      left_ext = comp_nucleotide(right_ext);
      right_ext = comp_nucleotide(tmp);
      if (!left_ext || !right_ext) return false;
      memcpy(kmer, kmer_rc, N_LONGS * sizeof(uint64_t));
    }
    break;
  }
  return true;
}

template <int MAX_K>
__global__ void gpu_insert_supermer_block(KmerCountsMap<MAX_K> elems, SupermerBuff supermer_buff, uint32_t buff_len, int kmer_len,
                                          bool ctg_kmers, InsertStats *insert_stats) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  const int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  int attempted_inserts = 0, dropped_inserts = 0, new_inserts = 0, key_empty_overlaps = 0;
  while (threadid > 0 && threadid < buff_len) {
    attempted_inserts++;
    KmerArray<MAX_K> kmer;
    char left_ext, right_ext;
    count_t kmer_count;
    if (!get_kmer_from_supermer<MAX_K>(supermer_buff, buff_len, kmer_len, kmer.longs, left_ext, right_ext, kmer_count)) break;
    bool skip_key_empty_overlap = false;
    for (int long_i = 0; long_i < N_LONGS; long_i++) {
      if (kmer.longs[long_i] == KEY_EMPTY) {
        skip_key_empty_overlap = true;
        key_empty_overlaps++;
        break;
      }
    }
    if (!skip_key_empty_overlap) {
      uint64_t slot = kmer_hash(kmer) % elems.capacity;
      auto start_slot = slot;
      int j;
      const int MAX_PROBE = (elems.capacity < 200 ? elems.capacity : 200);
      for (j = 0; j < MAX_PROBE; j++) {
        uint64_t old_key =
            atomicCAS(reinterpret_cast<unsigned long long *>(&(elems.keys[slot].longs[0])), KEY_EMPTY, kmer.longs[0]);
        // only insert new kmers; drop duplicates
        if (old_key == KEY_EMPTY || old_key == kmer.longs[0]) {
          bool found_slot = true;
          for (int long_i = 1; long_i < N_LONGS; long_i++) {
            uint64_t old_key =
                atomicCAS(reinterpret_cast<unsigned long long *>(&(elems.keys[slot].longs[long_i])), KEY_EMPTY, kmer.longs[long_i]);
            if (old_key != KEY_EMPTY && old_key != kmer.longs[long_i]) {
              found_slot = false;
              break;
            }
          }
          if (found_slot) {
            count_t *counts = elems.vals[slot].data;
            if (ctg_kmers) {
              // the count is the min of all counts. Use CAS to deal with the initial zero value
              int prev_count = atomicCAS(&(counts[0]), 0, kmer_count);
              if (prev_count)
                atomicMin(&(counts[0]), kmer_count);
              else
                new_inserts++;
            } else {
              int prev_count = atomicAdd(&(counts[0]), kmer_count);
              if (!prev_count) new_inserts++;
            }
            switch (left_ext) {
              case 'A': atomicAdd(&(counts[1]), kmer_count); break;
              case 'C': atomicAdd(&(counts[2]), kmer_count); break;
              case 'G': atomicAdd(&(counts[3]), kmer_count); break;
              case 'T': atomicAdd(&(counts[4]), kmer_count); break;
            }
            switch (right_ext) {
              case 'A': atomicAdd(&(counts[5]), kmer_count); break;
              case 'C': atomicAdd(&(counts[6]), kmer_count); break;
              case 'G': atomicAdd(&(counts[7]), kmer_count); break;
              case 'T': atomicAdd(&(counts[8]), kmer_count); break;
            }
            break;
          }
        }
        // quadratic probing - worse cache but reduced clustering
        slot = (start_slot + (j + 1) * (j + 1)) % elems.capacity;
      }
      // this entry didn't get inserted because we ran out of probing time (and probably space)
      if (j == MAX_PROBE) dropped_inserts++;
    }
    break;
  }
  reduce(attempted_inserts, buff_len, &insert_stats->attempted);
  reduce(dropped_inserts, buff_len, &insert_stats->dropped);
  reduce(new_inserts, buff_len, &insert_stats->new_inserts);
  reduce(key_empty_overlaps, buff_len, &insert_stats->key_empty_overlaps);
}

template <int MAX_K>
struct HashTableGPUDriver<MAX_K>::HashTableDriverState {
  cudaEvent_t event;
  QuickTimer insert_timer, kernel_timer;
};

template <int MAX_K>
void KmerArray<MAX_K>::set(const uint64_t *kmer) {
  memcpy(longs, kmer, N_LONGS * sizeof(uint64_t));
}

template <int MAX_K>
void KmerCountsMap<MAX_K>::init(int64_t ht_capacity) {
  capacity = ht_capacity;
  cudaErrchk(cudaMalloc(&keys, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMemset(keys, KEY_EMPTY_BYTE, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMalloc(&vals, capacity * sizeof(CountsArray)));
  cudaErrchk(cudaMemset(vals, 0, capacity * sizeof(CountsArray)));
}

template <int MAX_K>
void KmerCountsMap<MAX_K>::clear() {
  cudaFree(keys);
  cudaFree(vals);
}

template <int MAX_K>
void KmerExtsMap<MAX_K>::init(int64_t ht_capacity) {
  capacity = ht_capacity;
  cudaErrchk(cudaMalloc(&keys, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMemset(keys, KEY_EMPTY_BYTE, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMalloc(&vals, capacity * sizeof(CountExts)));
  cudaErrchk(cudaMemset(vals, 0, capacity * sizeof(CountExts)));
}

template <int MAX_K>
void KmerExtsMap<MAX_K>::clear() {
  cudaFree(keys);
  cudaFree(vals);
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::HashTableGPUDriver() {}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int max_elems, size_t gpu_avail_mem,
                                     double &init_time, size_t &gpu_bytes_reqd) {
  QuickTimer init_timer;
  init_timer.start();
  this->upcxx_rank_me = upcxx_rank_me;
  this->upcxx_rank_n = upcxx_rank_n;
  this->kmer_len = kmer_len;
  int device_count = 0;
  cudaErrchk(cudaGetDeviceCount(&device_count));
  int my_gpu_id = upcxx_rank_me % device_count;
  cudaErrchk(cudaSetDevice(my_gpu_id));
  // now check that we have sufficient memory for the required capacity
  size_t elem_buff_size = KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * (1 + sizeof(count_t)) * 1.5;
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(CountsArray);
  gpu_bytes_reqd = (max_elems * elem_size) / 0.85 + elem_buff_size;
  // save 1/10 of avail gpu memory for possible ctg kmers and compact hash table
  // set capacity to max avail remaining from gpu memory - more slots means lower load
  auto max_slots = 0.85 * (gpu_avail_mem - elem_buff_size) / elem_size;
  // find the first prime number lower than this value
  prime.set(min((size_t)max_slots, (size_t)(max_elems * 3)), false);
  auto ht_capacity = prime.get();
  read_kmers_dev.init(ht_capacity);
  // for transferring packed elements from host to gpu
  elem_buff_host.seqs = new char[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE];
  // these are not used for kmers from reads
  elem_buff_host.counts = nullptr;
  // buffer on the device
  cudaErrchk(cudaMalloc(&packed_elem_buff_dev.seqs, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE));
  cudaErrchk(cudaMalloc(&unpacked_elem_buff_dev.seqs, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * 2));
  packed_elem_buff_dev.counts = nullptr;
  unpacked_elem_buff_dev.counts = nullptr;

  cudaErrchk(cudaMalloc(&gpu_insert_stats, sizeof(InsertStats)));
  cudaErrchk(cudaMemset(gpu_insert_stats, 0, sizeof(InsertStats)));

  dstate = new HashTableDriverState();
  init_timer.stop();
  init_time = init_timer.get_elapsed();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::init_ctg_kmers(int max_elems, size_t gpu_avail_mem) {
  size_t elem_buff_size = KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * (1 + sizeof(count_t)) * 1.5;
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(CountsArray);
  size_t max_slots = 0.9 * (gpu_avail_mem - elem_buff_size) / elem_size;
  prime.set(min(max_slots, (size_t)(max_elems * 3)), false);
  auto ht_capacity = prime.get();
  ctg_kmers_dev.init(ht_capacity);
  elem_buff_host.counts = new count_t[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE];
  cudaErrchk(cudaMalloc(&packed_elem_buff_dev.counts, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(count_t)));
  cudaErrchk(cudaMalloc(&unpacked_elem_buff_dev.counts, 2 * KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(count_t)));
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::~HashTableGPUDriver() {
  if (dstate) delete dstate;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_supermer_block() {
  dstate->insert_timer.start();
  bool is_ctg_kmers = (pass_type == CTG_KMERS_PASS);
  cudaErrchk(cudaMemcpy(packed_elem_buff_dev.seqs, elem_buff_host.seqs, buff_len, cudaMemcpyHostToDevice));
  cudaErrchk(cudaMemset(unpacked_elem_buff_dev.seqs, 0, buff_len * 2));
  if (is_ctg_kmers)
    cudaErrchk(cudaMemcpy(packed_elem_buff_dev.counts, elem_buff_host.counts, buff_len * sizeof(count_t), cudaMemcpyHostToDevice));

  int gridsize, threadblocksize;
  get_kernel_config(buff_len, gpu_unpack_supermer_block, gridsize, threadblocksize);
  GPUTimer t;
  t.start();
  gpu_unpack_supermer_block<<<gridsize, threadblocksize>>>(unpacked_elem_buff_dev, packed_elem_buff_dev, buff_len);
  get_kernel_config(buff_len * 2, gpu_insert_supermer_block<MAX_K>, gridsize, threadblocksize);
  gpu_insert_supermer_block<<<gridsize, threadblocksize>>>(is_ctg_kmers ? ctg_kmers_dev : read_kmers_dev, unpacked_elem_buff_dev,
                                                           buff_len * 2, kmer_len, is_ctg_kmers, gpu_insert_stats);
  t.stop();
  dstate->kernel_timer.inc(t.get_elapsed());
  num_gpu_calls++;
  dstate->insert_timer.stop();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_supermer(const string &supermer_seq, count_t supermer_count) {
  if (buff_len + supermer_seq.length() + 1 >= KCOUNT_GPU_HASHTABLE_BLOCK_SIZE) {
    insert_supermer_block();
    buff_len = 0;
  }
  memcpy(&(elem_buff_host.seqs[buff_len]), supermer_seq.c_str(), supermer_seq.length());
  if (pass_type == CTG_KMERS_PASS) {
    for (int i = 0; i < (int)supermer_seq.length(); i++) elem_buff_host.counts[buff_len + i] = supermer_count;
  }
  buff_len += supermer_seq.length();
  elem_buff_host.seqs[buff_len] = '_';
  if (pass_type == CTG_KMERS_PASS) elem_buff_host.counts[buff_len] = 0;
  buff_len++;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::purge_invalid(int &num_purged, int &num_entries) {
  num_purged = num_entries = 0;
  unsigned int *counts_gpu;
  int NUM_COUNTS = 2;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  GPUTimer t;
  int gridsize, threadblocksize;
  get_kernel_config(read_kmers_dev.capacity, gpu_purge_invalid<MAX_K>, gridsize, threadblocksize);
  t.start();
  // now purge all invalid kmers (do it on the gpu)
  gpu_purge_invalid<<<gridsize, threadblocksize>>>(read_kmers_dev, counts_gpu);
  t.stop();
  dstate->kernel_timer.inc(t.get_elapsed());

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  num_purged = counts_host[0];
  num_entries = counts_host[1];
  auto expected_num_entries = read_kmers_stats.new_inserts - num_purged;
  if (num_entries != (int)expected_num_entries)
    cout << KLRED << "[" << upcxx_rank_me << "] WARNING mismatch " << num_entries << " != " << expected_num_entries << " diff "
         << (num_entries - expected_num_entries) << " new inserts " << read_kmers_stats.new_inserts << " num purged " << num_purged
         << KNORM << endl;
  read_kmers_dev.num = num_entries;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::flush_inserts() {
  if (buff_len) {
    insert_supermer_block();
    buff_len = 0;
  }
  cudaErrchk(cudaMemcpy(pass_type == READ_KMERS_PASS ? &read_kmers_stats : &ctg_kmers_stats, gpu_insert_stats, sizeof(InsertStats),
                        cudaMemcpyDeviceToHost));
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_all_inserts(int &num_dropped, int &num_unique, int &num_purged) {
  int num_entries = 0;
  purge_invalid(num_purged, num_entries);
  read_kmers_dev.num = num_entries;
  if (elem_buff_host.seqs) delete[] elem_buff_host.seqs;
  if (elem_buff_host.counts) delete[] elem_buff_host.counts;
  cudaFree(packed_elem_buff_dev.seqs);
  cudaFree(unpacked_elem_buff_dev.seqs);
  if (packed_elem_buff_dev.counts) cudaFree(packed_elem_buff_dev.counts);
  if (unpacked_elem_buff_dev.counts) cudaFree(unpacked_elem_buff_dev.counts);
  cudaFree(gpu_insert_stats);
  // overallocate to reduce collisions
  num_entries *= 1.3;
  // now compact the hash table entries
  unsigned int *counts_gpu;
  int NUM_COUNTS = 2;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  KmerExtsMap<MAX_K> compact_read_kmers_dev;
  compact_read_kmers_dev.init(num_entries);
  GPUTimer t;
  int gridsize, threadblocksize;
  get_kernel_config(read_kmers_dev.capacity, gpu_compact_ht<MAX_K>, gridsize, threadblocksize);
  t.start();
  gpu_compact_ht<<<gridsize, threadblocksize>>>(read_kmers_dev, compact_read_kmers_dev, counts_gpu);
  t.stop();
  dstate->kernel_timer.inc(t.get_elapsed());

  read_kmers_dev.clear();

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  num_dropped = counts_host[0];
  num_unique = counts_host[1];
  if (num_unique != read_kmers_dev.num)
    cerr << KLRED << "[" << upcxx_rank_me << "] <gpu_hash_table.cpp:" << __LINE__ << "> WARNING: " << KNORM
         << "mismatch in expected entries " << num_unique << " != " << read_kmers_dev.num << "\n";

  // now copy the gpu hash table values across to the host
  // We only do this once, which requires enough memory on the host to store the full GPU hash table, but since the GPU memory
  // is generally a lot less than the host memory, it should be fine.
  output_keys.resize(num_entries);
  output_vals.resize(num_entries);
  output_index = 0;
  // FIXME: can do this async - also
  cudaErrchk(cudaMemcpy(output_keys.data(), compact_read_kmers_dev.keys, compact_read_kmers_dev.capacity * sizeof(KmerArray<MAX_K>),
                        cudaMemcpyDeviceToHost));
  cudaErrchk(cudaMemcpy(output_vals.data(), compact_read_kmers_dev.vals, compact_read_kmers_dev.capacity * sizeof(CountExts),
                        cudaMemcpyDeviceToHost));
  compact_read_kmers_dev.clear();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_ctg_kmer_inserts(int &attempted_inserts, int &dropped_inserts, int &new_inserts) {
  unsigned int *counts_gpu;
  int NUM_COUNTS = 3;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  GPUTimer t;
  int gridsize, threadblocksize;
  get_kernel_config(ctg_kmers_dev.capacity, gpu_merge_ctg_kmers<MAX_K>, gridsize, threadblocksize);
  t.start();
  gpu_merge_ctg_kmers<<<gridsize, threadblocksize>>>(read_kmers_dev, ctg_kmers_dev, counts_gpu);
  t.stop();
  dstate->kernel_timer.inc(t.get_elapsed());
  ctg_kmers_dev.clear();
  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  attempted_inserts = counts_host[0];
  dropped_inserts = counts_host[1];
  new_inserts = counts_host[2];
  read_kmers_dev.num += new_inserts;
  read_kmers_stats.new_inserts += new_inserts;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::set_pass(PASS_TYPE p) {
  pass_type = p;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::get_elapsed_time(double &insert_time, double &kernel_time) {
  insert_time = dstate->insert_timer.get_elapsed();
  kernel_time = dstate->kernel_timer.get_elapsed();
}

template <int MAX_K>
pair<KmerArray<MAX_K> *, CountExts *> HashTableGPUDriver<MAX_K>::get_next_entry() {
  if (output_keys.empty() || output_index == output_keys.size()) return {nullptr, nullptr};
  output_index++;
  return {&(output_keys[output_index - 1]), &(output_vals[output_index - 1])};
}

template <int MAX_K>
int64_t HashTableGPUDriver<MAX_K>::get_capacity(PASS_TYPE p) {
  if (p == READ_KMERS_PASS)
    return read_kmers_dev.capacity;
  else
    return ctg_kmers_dev.capacity;
}

template <int MAX_K>
InsertStats &HashTableGPUDriver<MAX_K>::get_stats(PASS_TYPE p) {
  if (p == READ_KMERS_PASS)
    return read_kmers_stats;
  else
    return ctg_kmers_stats;
}

template <int MAX_K>
int HashTableGPUDriver<MAX_K>::get_num_gpu_calls() {
  return num_gpu_calls;
}

template class kcount_gpu::HashTableGPUDriver<32>;
#if MAX_BUILD_KMER >= 64
template class kcount_gpu::HashTableGPUDriver<64>;
#endif
#if MAX_BUILD_KMER >= 96
template class kcount_gpu::HashTableGPUDriver<96>;
#endif
#if MAX_BUILD_KMER >= 128
template class kcount_gpu::HashTableGPUDriver<128>;
#endif
#if MAX_BUILD_KMER >= 160
template class kcount_gpu::HashTableGPUDriver<160>;
#endif
