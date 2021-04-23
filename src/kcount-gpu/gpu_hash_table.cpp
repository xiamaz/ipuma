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

using namespace std;
using namespace gpu_utils;
using namespace kcount_gpu;

// From DEDUKT
/*
#define BIG_CONSTANT(x) (x)

inline __device__ keyType cuda_murmur3_64(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

KeyValue *create_hashtable_GPU(int rank) {
  int count, devId;
  cudaGetDeviceCount(&count);
  int gpuID = rank % count;
  cudaSetDevice(gpuID);
  // Allocate memory
  KeyValue *hashtable;
  cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);
  cudaMemset(hashtable, 0, sizeof(KeyValue) * kHashTableCapacity);

  return hashtable;
}

template <int MAX_K>
__global__ void gpu_hashtable_insert(KeyValue *hashtable, const KmerArray<MAX_K> *kmers, const char *ext, unsigned int num_kmers) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadid < num_kmers) {
    KmerArray<MAX_K> new_key = kmers[threadid];
    uint64_t slot = cuda_murmur3_64(new_key) & (kHashTableCapacity - 1);
    // loop in linear probe to max 100 (?) steps
    int i;
    for (i = 0; i < 100; i++) {
      // try to get lock on slot
      while (atomicCAS(&hashtable[slot].lock, 0, 1) == 1)
        ;
      bool done = false;
      // check key
      KmerArray<MAX_K> old_key = hashtable[slot].key;
      if (old_key == empty()) {
        // insert our kv into this slot
        done = true;
      } else if (old_key == new_key) {
        // update the existing value
        done = true;
      }
      // release the lock
      if (atomicCAS(&hashtable[slot].lock, 1, 0) != 1) die("some sort of error here");
      // success, we're done with the insert
      if (done) return;
      // slot was occupied, linear probe along to next available slot
      slot = (slot + 1) & (kHashTableCapacity - 1);
    }
    if (i == 100) count_dropped_insert();
  }
}
// end from DEDUKT
*/

template <int MAX_K>
struct HashTableGPUDriver<MAX_K>::HashTableDriverState {
  cudaEvent_t event;
  QuickTimer ht_timer;
};

static size_t get_nearest_pow2(size_t val) {
  for (size_t i = val; i >= 1; i--) {
    // If i is a power of 2
    if ((i & (i - 1)) == 0) return i;
  }
  return 0;
}

template <int MAX_K>
KmerArray<MAX_K>::KmerArray(const uint64_t *kmer) {
  memcpy(longs.data(), kmer, N_LONGS * sizeof(uint64_t));
}

template <int MAX_K>
void KmerArray<MAX_K>::operator=(const uint64_t *kmer) {
  memcpy(longs.data(), kmer, N_LONGS * sizeof(uint64_t));
}

template <int MAX_K>
bool KmerArray<MAX_K>::operator==(const KmerArray<MAX_K> &o) const {
  return longs == o.longs;
}

template <int MAX_K>
size_t KmerArray<MAX_K>::hash() const {
  return MurmurHash3_x64_64(reinterpret_cast<const void *>(longs.data()), N_LONGS * sizeof(uint64_t));
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::HashTableGPUDriver()
    : t_func(0)
    , t_malloc(0)
    , t_cp(0)
    , t_kernel(0) {}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int max_elems, int gpu_avail_mem,
                                     double &init_time) {
  this->upcxx_rank_me = upcxx_rank_me;
  this->upcxx_rank_n = upcxx_rank_n;
  this->kmer_len = kmer_len;
  QuickTimer init_timer, malloc_timer;
  init_timer.start();
  int device_count = 0;
  cudaErrchk(cudaGetDeviceCount(&device_count));
  int my_gpu_id = upcxx_rank_me % device_count;
  cudaErrchk(cudaSetDevice(my_gpu_id));
  malloc_timer.start();

  ht_capacity = get_nearest_pow2(max_elems);
  // FIXME: to go on device
  elems_dev.resize(ht_capacity);
  memset((void *)elems_dev.data(), 0, ht_capacity * sizeof(KeyValue<MAX_K>));
  // cudaErrchk(cudaMalloc(&elems_dev, max_elems * sizeof(KeyValue<MAX_K>)));
  // cudaErrchk(cudaMemset(elems_dev, 0, max_elems * sizeof(KeyValue<MAX_K>)));

  cudaErrchk(cudaMalloc(&locks_dev, ht_capacity * sizeof(uint8_t)));
  cudaErrchk(cudaMemset(locks_dev, 0, ht_capacity * sizeof(uint8_t)));
  malloc_timer.stop();
  t_malloc += malloc_timer.get_elapsed();

  // for transferring elements from host to gpu
  elem_buff_host = new KmerAndExts<MAX_K>[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE];

  dstate = new HashTableDriverState();

  init_timer.stop();
  init_time = init_timer.get_elapsed();
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::~HashTableGPUDriver() {
  if (dstate) delete dstate;
}

static uint16_t inc_ext_count_with_limit(int count1, int count2) {
  count1 += count2;
  return std::min(count1, (int)numeric_limits<uint16_t>::max());
}

static void inc_ext_count(KmerCountsArray &kmer_counts, char ext, int count, bool is_left) {
  int start_pos = (is_left ? 1 : 5);
  switch (ext) {
    case 'A': kmer_counts[start_pos] = inc_ext_count_with_limit(kmer_counts[start_pos], count); break;
    case 'C': kmer_counts[start_pos + 1] = inc_ext_count_with_limit(kmer_counts[start_pos + 1], count); break;
    case 'G': kmer_counts[start_pos + 2] = inc_ext_count_with_limit(kmer_counts[start_pos + 2], count); break;
    case 'T': kmer_counts[start_pos + 3] = inc_ext_count_with_limit(kmer_counts[start_pos + 3], count); break;
  }
}

// FIXME: this needs to be on the device
template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_kmer_block(int64_t &num_new_elems, int64_t &num_inserts, int64_t &num_dropped) {
  for (int i = 0; i < num_buff_entries; i++) {
    KmerArray<MAX_K> kmer = elem_buff_host[i].kmer;
    uint16_t kmer_count = elem_buff_host[i].count;
    char left_ext = elem_buff_host[i].left;
    char right_ext = elem_buff_host[i].right;
    uint64_t slot = kmer.hash() & (ht_capacity - 1);
    num_inserts++;
    // loop in linear probe to max 100 (?) steps
    int j;
    const int MAX_PROBE = 100;
    for (j = 0; j < MAX_PROBE; j++) {
      bool found = false;
      // empty if the count is zero
      if (!elems_dev[slot].val[0]) {
        elems_dev[slot].key = kmer;
        num_new_elems++;
        found = true;
      } else if (elems_dev[slot].key == kmer) {
        found = true;
      }
      if (found) {
        KmerCountsArray &kmer_counts = elems_dev[slot].val;
        kmer_counts[0] = inc_ext_count_with_limit(kmer_counts[0], kmer_count);
        inc_ext_count(kmer_counts, left_ext, kmer_count, true);
        inc_ext_count(kmer_counts, right_ext, kmer_count, false);
        break;
      }
      // slot was occupied, linear probe along to next available slot
      slot = (slot + 1) & (ht_capacity - 1);
    }
    if (j == MAX_PROBE) {
      // this entry didn't get inserted because we ran out of probing time (and probably space)
      num_dropped++;
    }
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_kmer(const uint64_t *kmer, uint16_t kmer_count, char left, char right) {
  dstate->ht_timer.start();
  elem_buff_host[num_buff_entries].kmer = kmer;
  elem_buff_host[num_buff_entries].count = kmer_count;
  elem_buff_host[num_buff_entries].left = left;
  elem_buff_host[num_buff_entries].right = right;
  num_buff_entries++;
  if (num_buff_entries == KCOUNT_GPU_HASHTABLE_BLOCK_SIZE) {
    // cp to dev and run kernel
    insert_kmer_block(num_elems, num_attempted_inserts, num_dropped_entries);
    num_buff_entries = 0;
  }
  dstate->ht_timer.stop();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_inserts() {
  dstate->ht_timer.start();
  if (num_buff_entries) {
    insert_kmer_block(num_elems, num_attempted_inserts, num_dropped_entries);
    num_buff_entries = 0;
  }
  // delete to make space before returning the hash table entries
  if (elem_buff_host) delete[] elem_buff_host;
  // now copy the gpu hash table values across to the host
  // We only do this once, which requires enough memory on the host to store the full GPU hash table, but since the GPU memory
  // is generally a lot less than the host memory, it should be fine.
  output_elems.resize(ht_capacity);
  output_index = 0;

  // FIXME: should be copying from dev to host memory here
  memcpy(output_elems.data(), elems_dev.data(), ht_capacity * sizeof(KeyValue<MAX_K>));
  // FIXME: this should be gpu only
  elems_dev.clear();
  elems_dev.shrink_to_fit();
  // free up gpu memory
  // cudaFree(elems_dev);
  cudaFree(locks_dev);
  dstate->ht_timer.stop();
}

template <int MAX_K>
double HashTableGPUDriver<MAX_K>::get_kernel_elapsed_time() {
  return dstate->ht_timer.get_elapsed();
}

template <int MAX_K>
KeyValue<MAX_K> *HashTableGPUDriver<MAX_K>::get_next_entry() {
  if (output_elems.empty() || output_index == output_elems.size()) return nullptr;
  output_index++;
  return &(output_elems[output_index]);
}

template <int MAX_K>
int HashTableGPUDriver<MAX_K>::get_N_LONGS() {
  return N_LONGS;
}

template <int MAX_K>
int64_t HashTableGPUDriver<MAX_K>::get_num_elems() {
  return num_elems;
}

template <int MAX_K>
int64_t HashTableGPUDriver<MAX_K>::get_num_dropped() {
  return num_dropped_entries;
}

template <int MAX_K>
int64_t HashTableGPUDriver<MAX_K>::get_num_inserts() {
  return num_attempted_inserts;
}

template <int MAX_K>
double HashTableGPUDriver<MAX_K>::get_load_factor() {
  return (double)num_elems / ht_capacity;
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
