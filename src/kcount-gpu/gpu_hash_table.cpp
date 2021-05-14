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

__device__ int warpReduceSum(int val, int n) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned mask = __ballot_sync(0xffffffff, threadid < n);
  for (int offset = warpSize / 2; offset > 0; offset /= 2) val += __shfl_down_sync(mask, val, offset);
  return val;
}

__device__ int blockReduceSum(int val, int n) {
  static __shared__ int shared[32];  // Shared mem for 32 partial sums
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  val = warpReduceSum(val, n);  // Each warp performs partial reduction

  if (lane_id == 0) shared[warp_id] = val;  // Write reduced value to shared memory

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane_id] : 0;

  if (warp_id == 0) val = warpReduceSum(val, n);  // Final reduce within first warp
  __syncthreads();
  return val;
}

__device__ void reduce(int count, int num, unsigned int *result) {
  int block_num = blockReduceSum(count, num);
  if (threadIdx.x == 0) atomicAdd(result, block_num);
}

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
  return gpu_murmurhash3_64(reinterpret_cast<const void *>(kmer.longs), kmer.N_LONGS * sizeof(cu_uint64_t));
}

template <int MAX_K>
__global__ void gpu_compact_ht(ElemsArray<MAX_K> elems, ElemsArray<MAX_K> compact_elems, unsigned int *elem_counts) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  const int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  int dropped_inserts = 0;
  int unique_inserts = 0;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = threadid; i < elems.capacity; i += num_threads) {
    if (elems.vals[i].data[0]) {
      KmerArray<MAX_K> kmer = elems.keys[i];
      cu_uint64_t slot = kmer_hash(kmer) % compact_elems.capacity;
      auto start_slot = slot;
      // we set a constraint on the max probe to track whether we are getting excessive collisions and need a bigger default compact
      // table
      const int MAX_PROBE = (compact_elems.capacity < 1000 ? compact_elems.capacity : 1000);
      // look for empty slot in compact hash table
      int j;
      for (j = 0; j < MAX_PROBE; j++) {
        cu_uint64_t old_key = atomicCAS(&(compact_elems.keys[slot].longs[0]), KEY_EMPTY, kmer.longs[0]);
        if (old_key == KEY_EMPTY) {
          unique_inserts++;
          // found empty slot, copy across
          for (int k = 1; k < N_LONGS; k++) compact_elems.keys[slot].longs[k] = kmer.longs[k];
          for (int k = 0; k < 9; k++) compact_elems.vals[slot].data[k] = elems.vals[threadid].data[k];
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
__global__ void gpu_purge_invalid(ElemsArray<MAX_K> elems, unsigned int *elem_counts) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_purged = 0;
  int num_elems = 0;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = threadid; i < elems.capacity; i += num_threads) {
    if (elems.vals[i].data[0]) {
      int ext_sum = 0;
      for (int j = 1; j < 9; j++) ext_sum += elems.vals[i].data[j];
      if (elems.vals[i].data[0] < 2 || !ext_sum) {
        elems.vals[i].data[0] = 0;
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
__global__ void gpu_insert_kmer_block(ElemsArray<MAX_K> elems, const KmerAndExts<MAX_K> *elem_buff, uint32_t num_buff_entries,
                                      unsigned int *insert_counts) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

  int N_LONGS = KmerArray<MAX_K>::N_LONGS;
  int attempted_inserts = 0, dropped_inserts = 0, new_inserts = 0;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = threadid; i < num_buff_entries; i += num_threads) {
    KmerArray<MAX_K> kmer = elem_buff[i].kmer;
    count_t kmer_count = elem_buff[i].count;
    char left_ext = elem_buff[i].left;
    char right_ext = elem_buff[i].right;
    cu_uint64_t slot = kmer_hash(kmer) % elems.capacity;
    auto start_slot = slot;
    attempted_inserts++;
    int j;
    const int MAX_PROBE = (elems.capacity < 200 ? elems.capacity : 200);
    for (j = 0; j < MAX_PROBE; j++) {
      cu_uint64_t old_key = atomicCAS(&(elems.keys[slot].longs[0]), KEY_EMPTY, kmer.longs[0]);
      // only insert new kmers; drop duplicates
      if (old_key == KEY_EMPTY || old_key == kmer.longs[0]) {
        bool found_slot = true;
        bool is_empty = (old_key == KEY_EMPTY);
        for (int long_i = 1; long_i < N_LONGS; long_i++) {
          cu_uint64_t old_key = atomicCAS(&(elems.keys[slot].longs[long_i]), KEY_EMPTY, kmer.longs[long_i]);
          if (old_key != KEY_EMPTY && old_key != kmer.longs[long_i]) {
            found_slot = false;
            break;
          }
        }
        if (found_slot) {
          count_t old_count = atomicAdd(&(elems.vals[slot].data[0]), kmer_count);
          if (is_empty) new_inserts++;
          switch (left_ext) {
            case 'A': atomicAdd(&(elems.vals[slot].data[1]), kmer_count); break;
            case 'C': atomicAdd(&(elems.vals[slot].data[2]), kmer_count); break;
            case 'G': atomicAdd(&(elems.vals[slot].data[3]), kmer_count); break;
            case 'T': atomicAdd(&(elems.vals[slot].data[4]), kmer_count); break;
          }
          switch (right_ext) {
            case 'A': atomicAdd(&(elems.vals[slot].data[5]), kmer_count); break;
            case 'C': atomicAdd(&(elems.vals[slot].data[6]), kmer_count); break;
            case 'G': atomicAdd(&(elems.vals[slot].data[7]), kmer_count); break;
            case 'T': atomicAdd(&(elems.vals[slot].data[8]), kmer_count); break;
          }
          break;
        }
      }
      // linear probing
      // slot = (start_slot + j) % ht_capacity;
      // quadratic probing - worse cache but reduced clustering
      slot = (start_slot + (j + 1) * (j + 1)) % elems.capacity;
    }
    // this entry didn't get inserted because we ran out of probing time (and probably space)
    if (j == MAX_PROBE) dropped_inserts++;
  }
  reduce(attempted_inserts, num_buff_entries, &(insert_counts[0]));
  reduce(dropped_inserts, num_buff_entries, &(insert_counts[1]));
  reduce(new_inserts, num_buff_entries, &(insert_counts[2]));
}

template <int MAX_K>
struct HashTableGPUDriver<MAX_K>::HashTableDriverState {
  cudaEvent_t event;
  QuickTimer insert_timer, kernel_timer, memcpy_timer;
};

template <int MAX_K>
KmerArray<MAX_K>::KmerArray(const uint64_t *kmer) {
  memcpy(longs, kmer, N_LONGS * sizeof(cu_uint64_t));
}

template <int MAX_K>
void ElemsArray<MAX_K>::init(int64_t ht_capacity) {
  capacity = ht_capacity;
  cudaErrchk(cudaMalloc(&keys, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMemset(keys, 0xff, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMalloc(&vals, capacity * sizeof(KmerCountsArray)));
  cudaErrchk(cudaMemset(vals, 0, capacity * sizeof(KmerCountsArray)));
}

template <int MAX_K>
void ElemsArray<MAX_K>::clear() {
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
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(KmerCountsArray);
  // save 1/5 of avail gpu memory for possible ctg kmers - probably excessive
  // set capacity to max avail remaining from gpu memory - more slots means lower load
  prime.set(0.8 * (gpu_avail_mem - elem_buff_size) / elem_size, false);
  auto ht_capacity = prime.get();
  gpu_bytes_reqd = max_elems * elem_size + elem_buff_size;
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
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(KmerCountsArray);
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
void HashTableGPUDriver<MAX_K>::insert_kmer_block(ElemsArray<MAX_K> &elems_array, InsertStats &stats) {
  dstate->insert_timer.start();
  // copy across outside of thread so that we can reuse the elem_buff_host to carry on with inserts while the gpu is running
  cudaErrchk(cudaMemcpy(elem_buff_dev, elem_buff_host, num_buff_entries * sizeof(KmerAndExts<MAX_K>), cudaMemcpyHostToDevice));
  unsigned int *counts_gpu;
  int const NUM_COUNTS = 3;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));

  // int mingridsize = 0;
  // int threadblocksize = 0;
  // cudaErrchk(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_insert_kmer_block<MAX_K>, 0, 0));
  int threadblocksize = 512;
  int gridsize = ((uint32_t)num_buff_entries + threadblocksize - 1) / threadblocksize;

  GPUTimer t;
  t.start();
  gpu_insert_kmer_block<<<gridsize, threadblocksize>>>(elems_array, elem_buff_dev, num_buff_entries, counts_gpu);
  t.stop();
  dstate->kernel_timer.inc(t.get_elapsed());

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  stats.attempted += counts_host[0];
  stats.dropped += counts_host[1];
  stats.new_inserts += counts_host[2];
  if (static_cast<unsigned int>(num_buff_entries) != counts_host[0])
    cerr << KLRED << "[" << upcxx_rank_me << "] WARNING: " << KNORM
         << "mismatch in GPU entries processed vs input: " << counts_host[0] << " != " << num_buff_entries << endl;
  stats.num_gpu_calls++;
  dstate->insert_timer.stop();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_kmer(const uint64_t *kmer, count_t kmer_count, char left, char right) {
  elem_buff_host[num_buff_entries].kmer = kmer;
  elem_buff_host[num_buff_entries].count = kmer_count;
  elem_buff_host[num_buff_entries].left = left;
  elem_buff_host[num_buff_entries].right = right;
  num_buff_entries++;
  if (num_buff_entries == KCOUNT_GPU_HASHTABLE_BLOCK_SIZE) {
    // cp to dev and run kernel
    if (pass_type == READ_KMERS_PASS)
      insert_kmer_block(read_kmers_dev, read_kmers_stats);
    else
      insert_kmer_block(ctg_kmers_dev, ctg_kmers_stats);
    num_buff_entries = 0;
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::flush_inserts() {
  if (num_buff_entries) {
    if (pass_type == READ_KMERS_PASS)
      insert_kmer_block(read_kmers_dev, read_kmers_stats);
    else
      insert_kmer_block(ctg_kmers_dev, ctg_kmers_stats);
    num_buff_entries = 0;
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_all_inserts() {
  if (elem_buff_host) delete[] elem_buff_host;
  cudaFree(elem_buff_dev);
  // int mingridsize = 0;
  // int threadblocksize = 0;
  // cudaErrchk(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_purge_invalid<MAX_K>, 0, 0));
  int threadblocksize = 512;
  int gridsize = ((uint32_t)read_kmers_dev.capacity + threadblocksize - 1) / threadblocksize;

  unsigned int *counts_gpu;
  int NUM_COUNTS = 2;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  GPUTimer purge_timer;
  purge_timer.start();
  // now purge all invalid kmers (do it on the gpu)
  gpu_purge_invalid<<<gridsize, threadblocksize>>>(read_kmers_dev, counts_gpu);
  purge_timer.stop();

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  num_purged += counts_host[0];
  auto num_entries = counts_host[1];
  auto expected_num_entries = read_kmers_stats.new_inserts - num_purged;
  if (num_entries != expected_num_entries)
    cout << KLRED << "WARNING mismatch " << num_entries << " != " << expected_num_entries << " diff "
         << (num_entries - expected_num_entries) << KNORM << endl;
  auto orig_num_entries = num_entries;
  // overallocate to reduce collisions
  num_entries *= 1.3;

  // now compact the hash table entries
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  ElemsArray<MAX_K> compact_read_kmers_dev;
  compact_read_kmers_dev.init(num_entries);
  GPUTimer compact_timer;
  compact_timer.start();
  gpu_compact_ht<<<gridsize, threadblocksize>>>(read_kmers_dev, compact_read_kmers_dev, counts_gpu);
  compact_timer.stop();
  read_kmers_dev.clear();

  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  num_unique_elems = counts_host[1];
  if (num_unique_elems != orig_num_entries)
    cerr << KLRED << "Rank " << upcxx_rank_me << " WARNING: mismatch in expected entries " << num_unique_elems
         << " != " << orig_num_entries << KNORM "\n";
  if (counts_host[0])
    cerr << KLRED << "Rank " << upcxx_rank_me << " WARNING: dropped " << counts_host[0] << " when compacting output" KNORM "\n";

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
  cudaErrchk(cudaMemcpy(output_vals.data(), compact_read_kmers_dev.vals, compact_read_kmers_dev.capacity * sizeof(KmerCountsArray),
                        cudaMemcpyDeviceToHost));
  dstate->memcpy_timer.stop();
  compact_read_kmers_dev.clear();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_ctg_kmer_inserts() {
  ctg_kmers_dev.clear();
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
pair<KmerArray<MAX_K> *, KmerCountsArray *> HashTableGPUDriver<MAX_K>::get_next_entry() {
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
int64_t HashTableGPUDriver<MAX_K>::get_num_unique_elems() {
  return num_unique_elems;
}

template <int MAX_K>
int64_t HashTableGPUDriver<MAX_K>::get_num_purged() {
  return num_purged;
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
