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
#include "gpu-utils/utils.hpp"
#include "kcount_driver.hpp"

#define KNORM "\x1B[0m"
#define KLGREEN "\x1B[92m"

using namespace std;
using namespace gpu_utils;

using timepoint_t = chrono::time_point<std::chrono::high_resolution_clock>;

#define cudaErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

static void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    ostringstream os;
    os << "GPU assert " << cudaGetErrorString(code) << " " << file << ":" << line << "\n";
    throw runtime_error(os.str());
  }
}

struct kcount_gpu::DriverState {
  int device_count;
  int my_gpu_id;
  int upcxx_rank_me;
  int upcxx_rank_n;
  double t_func = 0, t_malloc = 0, t_cp = 0, t_kernel = 0;
  char *seqs, *quals;
  // FIXME: only allowing kmers up to 32 in length
  uint64_t *kmers;
  int *kmer_targets;
  int max_kmers;
};

double kcount_gpu::KcountGPUDriver::init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len) {
  timepoint_t t = chrono::high_resolution_clock::now();
  driver_state = new DriverState();
  cudaErrchk(cudaGetDeviceCount(&driver_state->device_count));
  driver_state->my_gpu_id = upcxx_rank_me % driver_state->device_count;
  cudaErrchk(cudaSetDevice(driver_state->my_gpu_id));
  driver_state->upcxx_rank_me = upcxx_rank_me;
  driver_state->upcxx_rank_n = upcxx_rank_n;
  driver_state->t_func = 0;
  driver_state->t_malloc = 0;
  driver_state->t_cp = 0;
  driver_state->t_kernel = 0;
  driver_state->max_kmers = KCOUNT_READ_BLOCK_SIZE - kmer_len;
  // FIXME: this needs to support more than k=32

  timepoint_t t_malloc = chrono::high_resolution_clock::now();
  cudaErrchk(cudaMalloc(&driver_state->seqs, KCOUNT_READ_BLOCK_SIZE));
  cudaErrchk(cudaMalloc(&driver_state->quals, KCOUNT_READ_BLOCK_SIZE));
  // this is an upper limit on the kmers because there are many reads so there will be fewer kmers, but its ok to pad it
  cudaErrchk(cudaMalloc(&driver_state->kmers, driver_state->max_kmers * sizeof(uint64_t)));
  cudaErrchk(cudaMalloc(&driver_state->kmer_targets, driver_state->max_kmers * sizeof(int)));
  chrono::duration<double> t_elapsed = chrono::high_resolution_clock::now() - t_malloc;
  driver_state->t_malloc += t_elapsed.count();

  t_elapsed = chrono::high_resolution_clock::now() - t;
  return t_elapsed.count();
}

kcount_gpu::KcountGPUDriver::~KcountGPUDriver() {
  cudaFree(driver_state->seqs);
  cudaFree(driver_state->quals);
  cudaFree(driver_state->kmers);
  cudaFree(driver_state->kmer_targets);
  cudaDeviceSynchronize();
  delete driver_state;
}

__global__ void parse_and_pack(int upcxx_rank_me) {}

void kcount_gpu::KcountGPUDriver::process_read_block(unsigned kmer_len, int qual_offset, int num_kmers_in_block, string &read_seqs,
                                                     string &read_quals, vector<uint64_t> &host_kmers,
                                                     vector<int> &host_kmer_targets, int64_t &num_bad_quals, int64_t &num_Ns,
                                                     int64_t &num_kmers) {
  timepoint_t t_func = chrono::high_resolution_clock::now();
  timepoint_t t_cp = chrono::high_resolution_clock::now();
  cudaErrchk(cudaMemcpy(driver_state->seqs, &read_seqs[0], read_seqs.length() * sizeof(char), cudaMemcpyHostToDevice));
  cudaErrchk(cudaMemcpy(driver_state->quals, &read_quals[0], read_quals.length() * sizeof(char), cudaMemcpyHostToDevice));
  cudaMemset(driver_state->kmers, 0, driver_state->max_kmers * sizeof(uint64_t));
  cudaMemset(driver_state->kmer_targets, 0, driver_state->max_kmers * sizeof(int));

  chrono::duration<double> t_elapsed = chrono::high_resolution_clock::now() - t_cp;
  driver_state->t_cp += t_elapsed.count();
  /*
    int p_buff_len = ((n_kmers * BUFF_SCALE) + nproc - 1) / nproc;
    int b = 128;
    int g = (seq_len + (b - 1)) / b;
    int per_block_seq_len = (seq_len + (g - 1)) / g;

    timepoint_t t_kernel = chrono::high_resolution_clock::now();
    // CUDA call
    gpu_parseKmerNFillupBuff<<<g, b>>>(d_seq, d_kmers, klen, seq_len, d_outgoing, d_owner_counter, nproc, p_buff_len);
    t_elapsed = chrono::high_resolution_clock::now() - t_kernel;
    driver_state->t_kernel += t_elapsed.count();

    // h_outgoing = (uint64_t *) malloc ( n_kmers * BUFF_SCALE * sizeof(uint64_t));
    cudaErrchk(cudaMemcpy(&(h_outgoing[0]), d_outgoing, n_kmers * BUFF_SCALE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cudaErrchk(cudaMemcpy(owner_counter.data(), d_owner_counter, nproc * sizeof(int), cudaMemcpyDeviceToHost));

    uint64_t total_counter = 0;
    // printf("GPU ParseNPack: outgoing buufers: ");
    for (int i = 0; i < nproc; ++i) {
      total_counter += owner_counter[i];
    }
  */

  cudaDeviceSynchronize();

  t_elapsed = chrono::high_resolution_clock::now() - t_func;
  driver_state->t_func += t_elapsed.count();
}

tuple<double, double, double, double> kcount_gpu::KcountGPUDriver::get_elapsed_times() {
  return {driver_state->t_func, driver_state->t_malloc, driver_state->t_cp, driver_state->t_kernel};
}