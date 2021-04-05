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

struct kcount_gpu::HashTableDriverState {
  cudaEvent_t event;
};

static size_t get_nearest_pow2(size_t val) {
  for (size_t i = val; i >= 1; i--) {
    // If i is a power of 2
    if ((i & (i - 1)) == 0) return i;
  }
  return 0;
}

kcount_gpu::HashTableGPUDriver::HashTableGPUDriver(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int num_kmer_longs,
                                                   int gpu_avail_mem, double &init_time)
    : upcxx_rank_me(upcxx_rank_me)
    , upcxx_rank_n(upcxx_rank_n)
    , kmer_len(kmer_len)
    , num_kmer_longs(num_kmer_longs)
    , t_func(0)
    , t_malloc(0)
    , t_cp(0)
    , t_kernel(0) {
  QuickTimer init_timer, malloc_timer;
  init_timer.start();
  int device_count = 0;
  cudaErrchk(cudaGetDeviceCount(&device_count));
  int my_gpu_id = upcxx_rank_me % device_count;
  cudaErrchk(cudaSetDevice(my_gpu_id));
  int bytes_per_slot = num_kmer_longs * sizeof(uint64_t) + sizeof(uint16_t) + 1;
  // ensure the size is a power of 2 in order to use optimized binary & instead of % for index calculation
  num_ht_slots = get_nearest_pow2(gpu_avail_mem / bytes_per_slot);
  malloc_timer.start();
  cudaErrchk(cudaMalloc(&dev_kmers, num_ht_slots * num_kmer_longs * sizeof(uint64_t)));
  cudaErrchk(cudaMalloc(&dev_counts, num_ht_slots * sizeof(uint16_t)));
  cudaErrchk(cudaMalloc(&dev_mutexes, num_ht_slots * sizeof(char)));
  malloc_timer.stop();
  t_malloc += malloc_timer.get_elapsed();

  dstate = new HashTableDriverState();
  init_timer.stop();
  init_time = init_timer.get_elapsed();
}

kcount_gpu::HashTableGPUDriver::~HashTableGPUDriver() {
  cudaFree(dev_kmers);
  cudaFree(dev_counts);
  cudaFree(dev_mutexes);
  delete dstate;
}

int kcount_gpu::HashTableGPUDriver::get_num_ht_slots() { return num_ht_slots; }