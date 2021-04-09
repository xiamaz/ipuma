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
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "upcxx_utils/colors.h"
#include "gpu_common.hpp"
#include "gpu_hash_table.hpp"

using namespace std;
using namespace gpu_utils;
using namespace kcount_gpu;

template <int MAX_K>
struct HashTableGPUDriver<MAX_K>::HashTableDriverState {
  cudaEvent_t event;
};

/*
static size_t get_nearest_pow2(size_t val) {
  for (size_t i = val; i >= 1; i--) {
    // If i is a power of 2
    if ((i & (i - 1)) == 0) return i;
  }
  return 0;
}
*/

template <int MAX_K>
KmerArray<MAX_K>::KmerArray(const uint64_t *x) {
  memcpy(data.data(), x, N_LONGS);
}

template <int MAX_K>
bool KmerArray<MAX_K>::operator==(const KmerArray<MAX_K> &o) const {
  return data == o.data;
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::HashTableGPUDriver()
    : num_entries(0)
    , t_func(0)
    , t_malloc(0)
    , t_cp(0)
    , t_kernel(0) {}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int gpu_avail_mem, double &init_time) {
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
  cudaErrchk(cudaMalloc(&dev_kmers, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * N_LONGS * sizeof(uint64_t)));
  cudaErrchk(cudaMalloc(&dev_counts, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(uint32_t)));
  malloc_timer.stop();
  t_malloc += malloc_timer.get_elapsed();

  host_kmers = new uint64_t[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * N_LONGS];
  host_counts = new uint8_t[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * 4];

  dstate = new HashTableDriverState();

  tmp_ht.reserve(KCOUNT_GPU_HASHTABLE_BLOCK_SIZE);

  init_timer.stop();
  init_time = init_timer.get_elapsed();
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::~HashTableGPUDriver() {
  cudaFree(dev_kmers);
  cudaFree(dev_counts);
  if (host_kmers) delete[] host_kmers;
  if (host_counts) delete[] host_counts;
  if (dstate) delete dstate;
}

template <int MAX_K>
int HashTableGPUDriver<MAX_K>::get_num_entries() {
  return num_entries;
}

static uint16_t inc_ext_count_with_limit(int count1, int count2) {
  count1 += count2;
  return std::min(count1, (int)numeric_limits<uint16_t>::max());
}

static void inc_ext_count(array<uint16_t, 9> &kmer_counts, char ext, int count, bool is_left) {
  int start_pos = (is_left ? 1 : 5);
  switch (ext) {
    case 'A': kmer_counts[start_pos] = inc_ext_count_with_limit(kmer_counts[start_pos], count); break;
    case 'C': kmer_counts[start_pos + 1] = inc_ext_count_with_limit(kmer_counts[start_pos + 1], count); break;
    case 'G': kmer_counts[start_pos + 2] = inc_ext_count_with_limit(kmer_counts[start_pos + 2], count); break;
    case 'T': kmer_counts[start_pos + 3] = inc_ext_count_with_limit(kmer_counts[start_pos + 3], count); break;
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_kmer(const uint64_t *kmer, uint16_t kmer_count, char left, char right) {
  memcpy(host_kmers + num_entries * N_LONGS, kmer, N_LONGS * sizeof(uint64_t));
  memcpy(host_counts + num_entries * 4, (void *)&kmer_count, 2);
  host_counts[num_entries * 4 + 2] = left;
  host_counts[num_entries * 4 + 3] = right;
  num_entries++;
  if (num_entries == KCOUNT_GPU_HASHTABLE_BLOCK_SIZE) {
    if (!upcxx_rank_me) cout << "inserting " << num_entries << " kmers into the GPU ht\n";
    for (int i = 0; i < num_entries; i++) {
      KmerArray<MAX_K> kmer(host_kmers + i * N_LONGS);
      uint16_t kmer_count;
      memcpy(&kmer_count, host_counts + i * 4, 2);
      char left_ext = host_counts[i * 4 + 2];
      char right_ext = host_counts[i * 4 + 3];
      // std::unordered_map<KmerArray<N_LONGS>, std::array<uint16_t, 9>> tmp_ht;
      // find it - if it isn't found then insert it, otherwise increment the counts
      auto it = tmp_ht.find(kmer);
      if (it == tmp_ht.end()) it = tmp_ht.insert({kmer, {0}}).first;
      array<uint16_t, 9> &kmer_counts = it->second;
      kmer_counts[0] = inc_ext_count_with_limit(kmer_counts[0], kmer_count);
      inc_ext_count(kmer_counts, left_ext, kmer_count, true);
      inc_ext_count(kmer_counts, right_ext, kmer_count, false);
    }
    // cp to dev and run kernel
    num_entries = 0;
  }
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_inserts() {
  if (num_entries) {
    if (!upcxx_rank_me) cout << "inserting " << num_entries << " kmers into the GPU ht\n";
    for (int i = 0; i < num_entries; i++) {
      KmerArray<MAX_K> kmer(host_kmers + i * N_LONGS);
      uint16_t kmer_count;
      memcpy(&kmer_count, host_counts + i * 4, 2);
      char left_ext = host_counts[i * 4 + 2];
      char right_ext = host_counts[i * 4 + 3];
      // std::unordered_map<KmerArray<N_LONGS>, std::array<uint16_t, 9>> tmp_ht;
      // find it - if it isn't found then insert it, otherwise increment the counts
      auto it = tmp_ht.find(kmer);
      if (it == tmp_ht.end()) it = tmp_ht.insert({kmer, {0}}).first;
      array<uint16_t, 9> &kmer_counts = it->second;
      kmer_counts[0] = inc_ext_count_with_limit(kmer_counts[0], kmer_count);
      inc_ext_count(kmer_counts, left_ext, kmer_count, true);
      inc_ext_count(kmer_counts, right_ext, kmer_count, false);
    }
    // cp to dev and run kernel
    num_entries = 0;
  }
  ostringstream os;
  os << "(GPU) Rank " << upcxx_rank_me << " found " << tmp_ht.size() << " unique kmers\n";
  cout << os.str() << flush;
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
