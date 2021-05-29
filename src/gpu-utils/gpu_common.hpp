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

#pragma once

#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>

// Functions that are common to all cuda code; not to be used by upcxx code

#define cudaErrchk(ans) \
  { gpu_common::gpu_die((ans), __FILE__, __LINE__); }

namespace gpu_common {

void gpu_die(cudaError_t code, const char *file, int line, bool abort = true);

using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

class QuickTimer {
  timepoint_t t;
  double secs = 0;

 public:
  QuickTimer();
  void start();
  void stop();
  void inc(double s);
  double get_elapsed();
};

class GPUTimer {
  cudaEvent_t start_event, stop_event;
  float elapsed_t_ms = 0;

 public:
  GPUTimer();
  ~GPUTimer();
  void start();
  void stop();
  double get_elapsed();
};

inline __device__ int warpReduceSum(int val, int n) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned mask = __ballot_sync(0xffffffff, threadid < n);
  for (int offset = warpSize / 2; offset > 0; offset /= 2) val += __shfl_down_sync(mask, val, offset);
  return val;
}

inline __device__ int blockReduceSum(int val, int n) {
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

inline __device__ void reduce(int count, int num, unsigned int *result) {
  int block_num = blockReduceSum(count, num);
  if (threadIdx.x == 0) atomicAdd(result, block_num);
}

template <class T>
inline void get_kernel_config(unsigned max_val, T func, int &gridsize, int &threadblocksize) {
  int mingridsize = 0;
  threadblocksize = 0;
  cudaErrchk(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, func, 0, 0));
  gridsize = (max_val + threadblocksize - 1) / threadblocksize;
}

}  // namespace gpu_common
