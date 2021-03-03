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

#include "gpu_common.hpp"
#include "parse_and_pack.hpp"

using namespace std;
using namespace gpu_utils;

__constant__ uint64_t GPU_TWINS[256] = {
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

__constant__ uint64_t GPU_0_MASK[32] = {
    0x0000000000000000, 0xC000000000000000, 0xF000000000000000, 0xFC00000000000000, 0xFF00000000000000, 0xFFC0000000000000,
    0xFFF0000000000000, 0xFFFC000000000000, 0xFFFF000000000000, 0xFFFFC00000000000, 0xFFFFF00000000000, 0xFFFFFC0000000000,
    0xFFFFFF0000000000, 0xFFFFFFC000000000, 0xFFFFFFF000000000, 0xFFFFFFFC00000000, 0xFFFFFFFF00000000, 0xFFFFFFFFC0000000,
    0xFFFFFFFFF0000000, 0xFFFFFFFFFC000000, 0xFFFFFFFFFF000000, 0xFFFFFFFFFFC00000, 0xFFFFFFFFFFF00000, 0xFFFFFFFFFFFC0000,
    0xFFFFFFFFFFFF0000, 0xFFFFFFFFFFFFC000, 0xFFFFFFFFFFFFF000, 0xFFFFFFFFFFFFFC00, 0xFFFFFFFFFFFFFF00, 0xFFFFFFFFFFFFFFC0,
    0xFFFFFFFFFFFFFFF0, 0xFFFFFFFFFFFFFFFC};

struct kcount_gpu::ParseAndPackDriverState {
  cudaEvent_t event;
};

kcount_gpu::ParseAndPackGPUDriver::ParseAndPackGPUDriver(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int num_kmer_longs,
                                                         int minimizer_len, double &init_time)
    : upcxx_rank_me(upcxx_rank_me)
    , upcxx_rank_n(upcxx_rank_n)
    , kmer_len(kmer_len)
    , num_kmer_longs(num_kmer_longs)
    , minimizer_len(minimizer_len)
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
  max_kmers = KCOUNT_GPU_SEQ_BLOCK_SIZE - kmer_len + 1;

  malloc_timer.start();
  cudaErrchk(cudaMalloc((void **)&dev_seqs, KCOUNT_GPU_SEQ_BLOCK_SIZE));
  cudaErrchk(cudaMalloc((void **)&dev_kmers, max_kmers * num_kmer_longs * sizeof(uint64_t)));
  cudaErrchk(cudaMalloc((void **)&dev_kmer_targets, max_kmers * sizeof(int)));
  cudaErrchk(cudaMalloc((void **)&dev_is_rcs, max_kmers));
  // total storage required is approx KCOUNT_GPU_SEQ_BLOCK_SIZE * (1 + num_kmers_longs * sizeof(uint64_t) + sizeof(int) + 1)
  malloc_timer.stop();
  t_malloc += malloc_timer.get_elapsed();

  dstate = new ParseAndPackDriverState();
  init_timer.stop();
  init_time = init_timer.get_elapsed();
}

kcount_gpu::ParseAndPackGPUDriver::~ParseAndPackGPUDriver() {
  cudaFree(dev_seqs);
  cudaFree(dev_kmers);
  cudaFree(dev_kmer_targets);
  cudaFree(dev_is_rcs);
  cudaDeviceSynchronize();
  delete dstate;
}

__device__ void revcomp(uint64_t *longs, uint64_t *rc_longs, int kmer_len, int num_longs) {
  int last_long = (kmer_len + 31) / 32;
  for (size_t i = 0; i < last_long; i++) {
    uint64_t v = longs[i];
    rc_longs[last_long - 1 - i] = (GPU_TWINS[v & 0xFF] << 56) | (GPU_TWINS[(v >> 8) & 0xFF] << 48) |
                                  (GPU_TWINS[(v >> 16) & 0xFF] << 40) | (GPU_TWINS[(v >> 24) & 0xFF] << 32) |
                                  (GPU_TWINS[(v >> 32) & 0xFF] << 24) | (GPU_TWINS[(v >> 40) & 0xFF] << 16) |
                                  (GPU_TWINS[(v >> 48) & 0xFF] << 8) | (GPU_TWINS[(v >> 56)]);
  }
  uint64_t shift = (kmer_len % 32) ? 2 * (32 - (kmer_len % 32)) : 0;
  uint64_t shiftmask = (kmer_len % 32) ? (((((uint64_t)1) << shift) - 1) << (64 - shift)) : ((uint64_t)0);
  rc_longs[0] = rc_longs[0] << shift;
  for (size_t i = 1; i < last_long; i++) {
    rc_longs[i - 1] |= (rc_longs[i] & shiftmask) >> (64 - shift);
    rc_longs[i] = rc_longs[i] << shift;
  }
}

__device__ uint64_t quick_hash(uint64_t v) {
  v = v * 3935559000370003845 + 2691343689449507681;
  v ^= v >> 21;
  v ^= v << 37;
  v ^= v >> 4;
  v *= 4768777513237032717;
  v ^= v << 20;
  v ^= v >> 41;
  v ^= v << 5;
  return v;
}

__device__ uint64_t gpu_minimizer_hash_fast(int m, int kmer_len, int num_longs, uint64_t *longs, uint64_t *rc_longs) {
  const int chunk_step = 32 - ((m + 3) / 4) * 4;  // chunk_step is a multiple of 4

  int base;
  int num_candidates = kmer_len - m + 1;
  const int max_candidates = MAX_BUILD_KMER;
  uint64_t rc_candidates[max_candidates];

  // calculate and temporarily store all revcomp minimizer candidates on the stack
  for (base = 0; base <= kmer_len - m; base += chunk_step) {
    int shift = base % 32;
    int l = base / 32;
    uint64_t tmp = rc_longs[l];
    if (shift) {
      tmp = (tmp << (shift * 2));
      if (l < num_longs - 1) tmp |= rc_longs[l + 1] >> (64 - shift * 2);
    }
    for (int j = 0; j < chunk_step; j++) {
      if (base + j + m > kmer_len) break;
      rc_candidates[base + j] = ((tmp << (j * 2)) & GPU_0_MASK[m]);
    }
  }

  uint64_t minimizer = 0;
  // calculate and compare minimizers from revcomp
  for (base = 0; base <= kmer_len - m; base += chunk_step) {
    int shift = base % 32;
    int l = base / 32;
    uint64_t tmp = longs[l];
    if (shift) {
      tmp = (tmp << (shift * 2));
      if (l < num_longs - 1) tmp |= longs[l + 1] >> (64 - shift * 2);
    }
    for (int j = 0; j < chunk_step; j++) {
      if (base + j + m > kmer_len) break;
      uint64_t fwd_candidate = ((tmp << (j * 2)) & GPU_0_MASK[m]);
      auto &rc_candidate = rc_candidates[num_candidates - base - j - 1];
      uint64_t &least_candidate = (fwd_candidate < rc_candidate) ? fwd_candidate : rc_candidate;
      if (least_candidate > minimizer) minimizer = least_candidate;
    }
  }
  return quick_hash(minimizer);
}

__global__ void parse_and_pack(char *seqs, int minimizer_len, int kmer_len, int num_longs, int seqs_len, uint64_t *kmers,
                               int *kmer_targets, char *is_rcs, int num_ranks) {
  unsigned int tid = threadIdx.x;
  unsigned int lane_id = tid & (blockDim.x - 1);
  int per_block_seq_len = blockDim.x;
  int st_char_block = blockIdx.x * per_block_seq_len;
  int num_kmers = seqs_len - kmer_len + 1;
  const int MAX_LONGS = (MAX_BUILD_KMER + 31) / 32;
  uint64_t rc_longs[MAX_LONGS];
  for (int i = st_char_block + lane_id; i < (st_char_block + per_block_seq_len) && i < num_kmers; i += blockDim.x) {
    int l = 0, prev_l = 0;
    bool valid_kmer = true;
    uint64_t longs = 0;
    uint64_t *kmer = &(kmers[i * num_longs]);
    // each thread extracts one kmer
    for (int k = 0; k < kmer_len; k++) {
      char s = seqs[i + k];
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
      revcomp(kmer, rc_longs, kmer_len, num_longs);
      bool must_rc = false;
      for (l = 0; l < num_longs; l++) {
        if (rc_longs[l] == kmer[l]) continue;
        if (rc_longs[l] < kmer[l]) must_rc = true;
        break;
      }
      kmer_targets[i] = gpu_minimizer_hash_fast(minimizer_len, kmer_len, num_longs, kmer, rc_longs) % num_ranks;
      if (must_rc) {
        memcpy(kmer, rc_longs, num_longs * sizeof(uint64_t));
        is_rcs[i] = 1;
      } else {
        is_rcs[i] = 0;
      }
    } else {
      // indicate invalid with -1
      kmer_targets[i] = -1;
    }
  }
}

bool kcount_gpu::ParseAndPackGPUDriver::process_seq_block(const string &seqs, int64_t &num_Ns) {
  QuickTimer func_timer, cp_timer, kernel_timer;

  if (seqs.length() >= KCOUNT_GPU_SEQ_BLOCK_SIZE) return false;

  func_timer.start();
  cudaErrchk(cudaEventCreateWithFlags(&dstate->event, cudaEventDisableTiming | cudaEventBlockingSync));

  int num_kmers = seqs.length() - kmer_len + 1;
  kernel_timer.start();
  cp_timer.start();
  size_t kmer_bytes = num_kmers * num_kmer_longs * sizeof(uint64_t);
  cudaErrchk(cudaMemcpy(dev_seqs, &seqs[0], seqs.length(), cudaMemcpyHostToDevice));
  cudaErrchk(cudaMemset(dev_kmers, 0, kmer_bytes));
  cudaErrchk(cudaMemset(dev_is_rcs, 0, num_kmers));
  cp_timer.stop();

  int block_size = 128;
  int num_blocks = (seqs.length() + (block_size - 1)) / block_size;

  parse_and_pack<<<num_blocks, block_size>>>(dev_seqs, minimizer_len, kmer_len, num_kmer_longs, seqs.length(), dev_kmers,
                                             dev_kmer_targets, dev_is_rcs, upcxx_rank_n);
  host_kmers.resize(num_kmers * num_kmer_longs);
  host_kmer_targets.resize(num_kmers);
  host_is_rcs.resize(num_kmers);
  cp_timer.start();
  cudaErrchk(cudaMemcpy(&(host_kmers[0]), dev_kmers, num_kmers * num_kmer_longs * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  cudaErrchk(cudaMemcpy(&(host_kmer_targets[0]), dev_kmer_targets, num_kmers * sizeof(int), cudaMemcpyDeviceToHost));
  cudaErrchk(cudaMemcpy(&(host_is_rcs[0]), dev_is_rcs, num_kmers, cudaMemcpyDeviceToHost));
  cp_timer.stop();
  t_cp += cp_timer.get_elapsed();
  kernel_timer.stop();
  // subtract the time taken by the copy from the kernel time
  t_kernel += (kernel_timer.get_elapsed() - cp_timer.get_elapsed());
  // this is used to signal completion
  cudaErrchk(cudaEventRecord(dstate->event));
  cudaErrchk(cudaEventSynchronize(dstate->event));
  func_timer.stop();
  t_func += func_timer.get_elapsed();
  return true;
}

tuple<double, double, double, double> kcount_gpu::ParseAndPackGPUDriver::get_elapsed_times() {
  return {t_func, t_malloc, t_cp, t_kernel};
}

bool kcount_gpu::ParseAndPackGPUDriver::kernel_is_done() {
  if (cudaEventQuery(dstate->event) != cudaSuccess) return false;
  cudaErrchk(cudaEventDestroy(dstate->event));
  return true;
}
