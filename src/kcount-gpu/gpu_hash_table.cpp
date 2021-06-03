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

template <int MAX_K>
__global__ void gpu_insert_kmer_block(KmerCountsMap<MAX_K> elems, const KmerAndExts<MAX_K> *elem_buff, uint32_t num_buff_entries,
                                      bool ctg_kmers, unsigned int *insert_counts) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  int attempted_inserts = 0, dropped_inserts = 0, new_inserts = 0, key_empty_overlaps = 0;
  if (threadid < num_buff_entries) {
    attempted_inserts++;

    // FIXME: the input buf should be an array of chars representing supermers. Need to take the threadid position in that supermer
    // and extract the kmer at that point - this is what we will use below. Note: we'll also need the quality scores to determine if
    // the kmer is valid. The computation is like the inner calculation in KmerDHT<MAX_K>::get_kmers_and_exts()

    KmerArray<MAX_K> kmer = elem_buff[threadid].kmer;
    bool skip_key_empty_overlap = false;
    for (int long_i = 0; long_i < N_LONGS; long_i++) {
      if (kmer.longs[long_i] == KEY_EMPTY) {
        skip_key_empty_overlap = true;
        key_empty_overlaps++;
        break;
      }
    }
    if (!skip_key_empty_overlap) {
      count_t kmer_count = elem_buff[threadid].count;
      char left_ext = elem_buff[threadid].left;
      char right_ext = elem_buff[threadid].right;
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
  }
  reduce(attempted_inserts, num_buff_entries, &(insert_counts[0]));
  reduce(dropped_inserts, num_buff_entries, &(insert_counts[1]));
  reduce(new_inserts, num_buff_entries, &(insert_counts[2]));
  reduce(key_empty_overlaps, num_buff_entries, &(insert_counts[3]));
}

template <int MAX_K>
struct HashTableGPUDriver<MAX_K>::HashTableDriverState {
  cudaEvent_t event;
  QuickTimer insert_timer, kernel_timer, memcpy_timer;
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
  size_t elem_buff_size = KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(KmerAndExts<MAX_K>);
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(CountsArray);
  gpu_bytes_reqd = (max_elems * elem_size) / 0.85 + elem_buff_size;
  // save 1/10 of avail gpu memory for possible ctg kmers and compact hash table
  // set capacity to max avail remaining from gpu memory - more slots means lower load
  auto max_slots = 0.85 * (gpu_avail_mem - elem_buff_size) / elem_size;
  // find the first prime number lower than this value
  prime.set(min((size_t)max_slots, (size_t)(max_elems * 3)), false);
  auto ht_capacity = prime.get();
  read_kmers_dev.init(ht_capacity);
  // for transferring elements from host to gpu
  elem_buff_host = new KmerAndExts<MAX_K>[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE];
  // buffer on the device
  cudaErrchk(cudaMalloc(&elem_buff_dev, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(KmerAndExts<MAX_K>)));

  dstate = new HashTableDriverState();
  init_timer.stop();
  init_time = init_timer.get_elapsed();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::init_ctg_kmers(int max_elems, size_t gpu_avail_mem) {
  size_t elem_buff_size = KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(KmerAndExts<MAX_K>);
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(CountsArray);
  size_t max_slots = 0.9 * (gpu_avail_mem - elem_buff_size) / elem_size;
  prime.set(min(max_slots, (size_t)(max_elems * 3)), false);
  auto ht_capacity = prime.get();
  ctg_kmers_dev.init(ht_capacity);
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::~HashTableGPUDriver() {
  if (dstate) delete dstate;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_kmer_block(KmerCountsMap<MAX_K> &kmer_counts_map, InsertStats &stats, bool ctg_kmers) {
  dstate->insert_timer.start();
  // copy across outside of thread so that we can reuse the elem_buff_host to carry on with inserts while the gpu is running
  cudaErrchk(cudaMemcpy(elem_buff_dev, elem_buff_host, num_buff_entries * sizeof(KmerAndExts<MAX_K>), cudaMemcpyHostToDevice));
  unsigned int *counts_gpu;
  int const NUM_COUNTS = 4;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));

  int gridsize, threadblocksize;
  get_kernel_config(num_buff_entries, gpu_insert_kmer_block<MAX_K>, gridsize, threadblocksize);
  GPUTimer t;
  t.start();
  gpu_insert_kmer_block<<<gridsize, threadblocksize>>>(kmer_counts_map, elem_buff_dev, num_buff_entries, ctg_kmers, counts_gpu);
  t.stop();
  dstate->kernel_timer.inc(t.get_elapsed());

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  stats.attempted += counts_host[0];
  stats.dropped += counts_host[1];
  stats.new_inserts += counts_host[2];
  stats.key_empty_overlaps += counts_host[3];
  if (static_cast<unsigned int>(num_buff_entries) != counts_host[0])
    cerr << KLRED << "[" << upcxx_rank_me << "] WARNING: " << KNORM
         << "mismatch in GPU entries processed vs input: " << counts_host[0] << " != " << num_buff_entries << endl;
  stats.num_gpu_calls++;
  dstate->insert_timer.stop();
}

// FIXME: needs to be in the gpu (actually same as function in parse_and_pack.cpp - should be moved to common funcs)
const uint64_t _GPU_TWINS[256] = {
    0xFF, 0xBF, 0x7F, 0x3F, 0xEF, 0xAF, 0x6F, 0x2F, 0xDF, 0x9F, 0x5F, 0x1F, 0xCF, 0x8F, 0x4F, 0x0F, 0xFB, 0xBB, 0x7B, 0x3B,
    0xEB, 0xAB, 0x6B, 0x2B, 0xDB, 0x9B, 0x5B, 0x1B, 0xCB, 0x8B, 0x4B, 0x0B, 0xF7, 0xB7, 0x77, 0x37, 0xE7, 0xA7, 0x67, 0x27,
    0xD7, 0x97, 0x57, 0x17, 0xC7, 0x87, 0x47, 0x07, 0xF3, 0xB3, 0x73, 0x33, 0xE3, 0xA3, 0x63, 0x23, 0xD3, 0x93, 0x53, 0x13,
    0xC3, 0x83, 0x43, 0x03, 0xFE, 0xBE, 0x7E, 0x3E, 0xEE, 0xAE, 0x6E, 0x2E, 0xDE, 0x9E, 0x5E, 0x1E, 0xCE, 0x8E, 0x4E, 0x0E,
    0xFA, 0xBA, 0x7A, 0x3A, 0xEA, 0xAA, 0x6A, 0x2A, 0xDA, 0x9A, 0x5A, 0x1A, 0xCA, 0x8A, 0x4A, 0x0A, 0xF6, 0xB6, 0x76, 0x36,
    0xE6, 0xA6, 0x66, 0x26, 0xD6, 0x96, 0x56, 0x16, 0xC6, 0x86, 0x46, 0x06, 0xF2, 0xB2, 0x72, 0x32, 0xE2, 0xA2, 0x62, 0x22,
    0xD2, 0x92, 0x52, 0x12, 0xC2, 0x82, 0x42, 0x02, 0xFD, 0xBD, 0x7D, 0x3D, 0xED, 0xAD, 0x6D, 0x2D, 0xDD, 0x9D, 0x5D, 0x1D,
    0xCD, 0x8D, 0x4D, 0x0D, 0xF9, 0xB9, 0x79, 0x39, 0xE9, 0xA9, 0x69, 0x29, 0xD9, 0x99, 0x59, 0x19, 0xC9, 0x89, 0x49, 0x09,
    0xF5, 0xB5, 0x75, 0x35, 0xE5, 0xA5, 0x65, 0x25, 0xD5, 0x95, 0x55, 0x15, 0xC5, 0x85, 0x45, 0x05, 0xF1, 0xB1, 0x71, 0x31,
    0xE1, 0xA1, 0x61, 0x21, 0xD1, 0x91, 0x51, 0x11, 0xC1, 0x81, 0x41, 0x01, 0xFC, 0xBC, 0x7C, 0x3C, 0xEC, 0xAC, 0x6C, 0x2C,
    0xDC, 0x9C, 0x5C, 0x1C, 0xCC, 0x8C, 0x4C, 0x0C, 0xF8, 0xB8, 0x78, 0x38, 0xE8, 0xA8, 0x68, 0x28, 0xD8, 0x98, 0x58, 0x18,
    0xC8, 0x88, 0x48, 0x08, 0xF4, 0xB4, 0x74, 0x34, 0xE4, 0xA4, 0x64, 0x24, 0xD4, 0x94, 0x54, 0x14, 0xC4, 0x84, 0x44, 0x04,
    0xF0, 0xB0, 0x70, 0x30, 0xE0, 0xA0, 0x60, 0x20, 0xD0, 0x90, 0x50, 0x10, 0xC0, 0x80, 0x40, 0x00};

static void _revcomp(uint64_t *longs, uint64_t *rc_longs, int kmer_len, int num_longs) {
  int last_long = (kmer_len + 31) / 32;
  for (int i = 0; i < last_long; i++) {
    uint64_t v = longs[i];
    rc_longs[last_long - 1 - i] = (_GPU_TWINS[v & 0xFF] << 56) | (_GPU_TWINS[(v >> 8) & 0xFF] << 48) |
                                  (_GPU_TWINS[(v >> 16) & 0xFF] << 40) | (_GPU_TWINS[(v >> 24) & 0xFF] << 32) |
                                  (_GPU_TWINS[(v >> 32) & 0xFF] << 24) | (_GPU_TWINS[(v >> 40) & 0xFF] << 16) |
                                  (_GPU_TWINS[(v >> 48) & 0xFF] << 8) | (_GPU_TWINS[(v >> 56)]);
  }
  uint64_t shift = (kmer_len % 32) ? 2 * (32 - (kmer_len % 32)) : 0;
  uint64_t shiftmask = (kmer_len % 32) ? (((((uint64_t)1) << shift) - 1) << (64 - shift)) : ((uint64_t)0);
  rc_longs[0] = rc_longs[0] << shift;
  for (int i = 1; i < last_long; i++) {
    rc_longs[i - 1] |= (rc_longs[i] & shiftmask) >> (64 - shift);
    rc_longs[i] = rc_longs[i] << shift;
  }
}

static char _comp_nucleotide(char ch) {
  switch (ch) {
    case 'A': return 'T';
    case 'C': return 'G';
    case 'G': return 'C';
    case 'T': return 'A';
    case 'N': return 'N';
    case '0': return '0';
    case 'U':
    case 'R':
    case 'Y':
    case 'K':
    case 'M':
    case 'S':
    case 'W':
    case 'B':
    case 'D':
    case 'H':
    case 'V': return 'N';
    default:
      cerr << KLRED << "Invalid char for nucleotide '" << ch << "' int " << (int)ch << KNORM << "\n";
      abort();
      break;
  }
  return 0;
}

// FIXME: this should be done on the GPU (and this code is taken from parse_and_pack.cpp)
static void get_kmer_from_supermer(const char *seqs, int kmer_len, int num_longs, int seqs_len, int threadid, uint64_t *kmer,
                                   bool *is_rc, bool *is_valid) {
  // unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  *is_valid = false;
  int num_kmers = seqs_len - kmer_len + 1;
  const int MAX_LONGS = (MAX_BUILD_KMER + 31) / 32;
  uint64_t kmer_rc[MAX_LONGS];
  if (threadid < num_kmers) {
    int l = 0, prev_l = 0;
    bool valid_kmer = true;
    uint64_t longs = 0;
    memset(kmer, 0, sizeof(uint64_t) * num_longs);
    // each thread extracts one kmer
    for (int k = 0; k < kmer_len; k++) {
      char s = seqs[threadid + k];
      if (s == '_' || s == 'N') {
        valid_kmer = false;
        break;
      }
      int j = k % 32;
      prev_l = l;
      l = k / 32;
      // we do it this way so we can operate on the variable longs in a register, rather than local memory in the array
      if (l > prev_l) {
        kmer[prev_l] = longs;
        longs = 0;
        prev_l = l;
      }
      uint64_t x = (s & 4) >> 1;
      longs |= ((x + ((x ^ (s & 2)) >> 1)) << (2 * (31 - j)));
    }
    kmer[l] = longs;
    if (valid_kmer) {
      _revcomp(kmer, kmer_rc, kmer_len, num_longs);
      *is_rc = false;
      for (l = 0; l < num_longs; l++) {
        if (kmer_rc[l] == kmer[l]) continue;
        if (kmer_rc[l] < kmer[l]) {
          *is_rc = true;
          memcpy(kmer, kmer_rc, num_longs * sizeof(uint64_t));
        }
        break;
      }
      *is_valid = true;
    }
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_supermer(int kmer_len, const string &supermer_seq, const string &supermer_quals,
                                                count_t supermer_count) {
  for (int i = 1; i < (int)(supermer_seq.length() - kmer_len); i++) {
    bool is_rc = false, is_valid = false;
    KmerArray<MAX_K> kmer;
    get_kmer_from_supermer(supermer_seq.c_str(), kmer_len, N_LONGS, supermer_seq.length(), i, kmer.longs, &is_rc, &is_valid);
    if (!is_valid) continue;
    char left_ext = supermer_seq[i - 1];
    if (!supermer_quals[i - 1]) left_ext = '0';
    char right_ext = supermer_seq[i + kmer_len];
    if (!supermer_quals[i + kmer_len]) right_ext = '0';
    if (is_rc) {
      swap(left_ext, right_ext);
      left_ext = _comp_nucleotide(left_ext);
      right_ext = _comp_nucleotide(right_ext);
    };
    elem_buff_host[num_buff_entries].kmer.set(kmer.longs);
    elem_buff_host[num_buff_entries].count = supermer_count;
    elem_buff_host[num_buff_entries].left = left_ext;
    elem_buff_host[num_buff_entries].right = right_ext;
    num_buff_entries++;
    if (num_buff_entries == KCOUNT_GPU_HASHTABLE_BLOCK_SIZE) {
      if (pass_type == READ_KMERS_PASS)
        insert_kmer_block(read_kmers_dev, read_kmers_stats, false);
      else
        insert_kmer_block(ctg_kmers_dev, ctg_kmers_stats, true);
      num_buff_entries = 0;
    }
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::purge_invalid(int &num_purged, int &num_entries) {
  num_purged = num_entries = 0;
  unsigned int *counts_gpu;
  int NUM_COUNTS = 2;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  GPUTimer purge_timer;
  purge_timer.start();
  int gridsize, threadblocksize;
  get_kernel_config(read_kmers_dev.capacity, gpu_purge_invalid<MAX_K>, gridsize, threadblocksize);
  // now purge all invalid kmers (do it on the gpu)
  gpu_purge_invalid<<<gridsize, threadblocksize>>>(read_kmers_dev, counts_gpu);
  purge_timer.stop();

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  num_purged = counts_host[0];
  num_entries = counts_host[1];
  auto expected_num_entries = read_kmers_stats.new_inserts - num_purged;
  if (num_entries != expected_num_entries)
    cout << KLRED << "[" << upcxx_rank_me << "] WARNING mismatch " << num_entries << " != " << expected_num_entries << " diff "
         << (num_entries - expected_num_entries) << " new inserts " << read_kmers_stats.new_inserts << " num purged " << num_purged
         << KNORM << endl;
  read_kmers_dev.num = num_entries;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::flush_inserts() {
  if (num_buff_entries) {
    if (pass_type == READ_KMERS_PASS)
      insert_kmer_block(read_kmers_dev, read_kmers_stats, false);
    else
      insert_kmer_block(ctg_kmers_dev, ctg_kmers_stats, true);
    num_buff_entries = 0;
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_all_inserts(int &num_dropped, int &num_unique, int &num_purged) {
  int num_entries = 0;
  purge_invalid(num_purged, num_entries);
  read_kmers_dev.num = num_entries;
  if (elem_buff_host) delete[] elem_buff_host;
  cudaFree(elem_buff_dev);
  // overallocate to reduce collisions
  num_entries *= 1.3;
  // now compact the hash table entries
  unsigned int *counts_gpu;
  int NUM_COUNTS = 2;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  KmerExtsMap<MAX_K> compact_read_kmers_dev;
  compact_read_kmers_dev.init(num_entries);
  GPUTimer compact_timer;
  compact_timer.start();
  int gridsize, threadblocksize;
  get_kernel_config(read_kmers_dev.capacity, gpu_compact_ht<MAX_K>, gridsize, threadblocksize);
  gpu_compact_ht<<<gridsize, threadblocksize>>>(read_kmers_dev, compact_read_kmers_dev, counts_gpu);
  compact_timer.stop();
  read_kmers_dev.clear();

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  num_dropped = counts_host[0];
  num_unique = counts_host[1];
  if (num_unique != read_kmers_dev.num)
    cerr << KLRED << "[" << upcxx_rank_me << "] <gpu_hash_table.cpp:" << __LINE__ << "> WARNING: " << KNORM
         << "mismatch in expected entries " << num_unique << " != " << read_kmers_dev.num << "\n";

  dstate->memcpy_timer.start();
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
  dstate->memcpy_timer.stop();
  compact_read_kmers_dev.clear();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_ctg_kmer_inserts(int &attempted_inserts, int &dropped_inserts, int &new_inserts) {
  unsigned int *counts_gpu;
  int NUM_COUNTS = 3;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  GPUTimer insert_timer;
  int gridsize, threadblocksize;
  get_kernel_config(ctg_kmers_dev.capacity, gpu_merge_ctg_kmers<MAX_K>, gridsize, threadblocksize);
  insert_timer.start();
  gpu_merge_ctg_kmers<<<gridsize, threadblocksize>>>(read_kmers_dev, ctg_kmers_dev, counts_gpu);
  insert_timer.stop();
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
void HashTableGPUDriver<MAX_K>::get_elapsed_time(double &insert_time, double &kernel_time, double &memcpy_time) {
  insert_time = dstate->insert_timer.get_elapsed();
  kernel_time = dstate->kernel_timer.get_elapsed();
  memcpy_time = dstate->memcpy_timer.get_elapsed();
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
