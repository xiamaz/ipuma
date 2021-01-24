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
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "kcount_driver.hpp"

#define KNORM "\x1B[0m"
#define KLGREEN "\x1B[92m"

using namespace std;

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
  bool first_msg;
  int upcxx_rank_me;
};

double kcount_gpu::KcountGPUDriver::init(int upcxx_rank_me, int upcxx_rank_n) {
  using timepoint_t = chrono::time_point<std::chrono::high_resolution_clock>;
  timepoint_t t = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed;
  driver_state = new DriverState();
  cudaErrchk(cudaGetDeviceCount(&driver_state->device_count));
  driver_state->my_gpu_id = upcxx_rank_me % driver_state->device_count;
  cudaErrchk(cudaSetDevice(driver_state->my_gpu_id));
  driver_state->first_msg = true;
  driver_state->upcxx_rank_me = upcxx_rank_me;
  elapsed = chrono::high_resolution_clock::now() - t;
  return elapsed.count();
}

kcount_gpu::KcountGPUDriver::~KcountGPUDriver() { delete driver_state; }

__global__ void parse_and_pack(int upcxx_rank_me) { printf("GPU says hello from rank %d\n", upcxx_rank_me); }

void kcount_gpu::KcountGPUDriver::process_read_block(unsigned kmer_len, int qual_offset,
                                                     vector<pair<uint16_t, unsigned char *>> &read_block, int64_t &num_bad_quals,
                                                     int64_t &num_Ns, int64_t &num_kmers) {
  if (driver_state->first_msg) {
    cout << KLGREEN << "GPU called from rank " << driver_state->upcxx_rank_me << ": about to process block on gpu for kmer length "
         << kmer_len << KNORM << endl;
    driver_state->first_msg = false;
    int block_size = 1;
    int num_blocks = 1;
    parse_and_pack<<<num_blocks, block_size>>>(driver_state->upcxx_rank_me);
    cudaDeviceSynchronize();
  }
}
