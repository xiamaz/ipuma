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
#include <array>

#include "gpu_utils.hpp"
#include "upcxx_utils/colors.h"
#include "gpu_common.hpp"

using namespace std;

size_t gpu_utils::get_avail_gpu_mem_per_rank(int totRanks, int num_devices) {
  if (num_devices == 0) num_devices = get_num_node_gpus();
  if (!num_devices) return 0;
  int ranksPerDevice = totRanks / num_devices;
  return (get_tot_gpu_mem() * 0.8) / ranksPerDevice;
}

string gpu_utils::get_gpu_device_name() {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, 0));
  return prop.name;
}

size_t gpu_utils::get_tot_gpu_mem() {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, 0));
  return prop.totalGlobalMem;
}

size_t gpu_utils::get_free_gpu_mem() {
  size_t free_mem, tot_mem;
  cudaErrchk(cudaMemGetInfo(&free_mem, &tot_mem));
  return free_mem;
}

int gpu_utils::get_num_node_gpus() {
  int deviceCount = 0;
  auto res = cudaGetDeviceCount(&deviceCount);
  if (res != cudaSuccess) return 0;
  return deviceCount;
}

bool gpu_utils::initialize_gpu(double& time_to_initialize, int& device_count, size_t& total_mem) {
  using timepoint_t = chrono::time_point<chrono::high_resolution_clock>;
  double* first_touch;

  timepoint_t t = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed;

  device_count = get_num_node_gpus();
  if (device_count > 0) {
    total_mem = get_tot_gpu_mem();
    cudaErrchk(cudaMallocHost((void**)&first_touch, sizeof(double)));
    cudaErrchk(cudaFreeHost(first_touch));
  }
  elapsed = chrono::high_resolution_clock::now() - t;
  time_to_initialize = elapsed.count();
  return device_count > 0;
}

bool gpu_utils::initialize_gpu() {
  double t;
  int c;
  size_t m;
  return initialize_gpu(t, c, m);
}

string gpu_utils::get_gpu_device_description() {
  cudaDeviceProp prop;
  int num_devs = get_num_node_gpus();
  ostringstream os;
  for (int i = 0; i < num_devs; ++i) {
    cudaErrchk(cudaGetDeviceProperties(&prop, i));

    os << KLMAGENTA << "GPU Device number: " << i << "\n";
    os << "  Device name: " << prop.name << "\n";
    os << "  PCI device ID: " << prop.pciDeviceID << "\n";
    os << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    os << "  Clock Rate: " << prop.clockRate << "kHz\n";
    os << "  Total SMs: " << prop.multiProcessorCount << "\n";
    os << "  Shared Memory Per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
    os << "  Registers Per SM: " << prop.regsPerMultiprocessor << " 32-bit\n";
    os << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    os << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
    os << "  Total Global Memory: " << prop.totalGlobalMem << " bytes\n";
    os << "  Memory Clock Rate: " << prop.memoryClockRate << " kHz\n\n";

    os << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    os << "  Max threads in X-dimension of block: " << prop.maxThreadsDim[0] << "\n";
    os << "  Max threads in Y-dimension of block: " << prop.maxThreadsDim[1] << "\n";
    os << "  Max threads in Z-dimension of block: " << prop.maxThreadsDim[2] << "\n\n";

    os << "  Max blocks in X-dimension of grid: " << prop.maxGridSize[0] << "\n";
    os << "  Max blocks in Y-dimension of grid: " << prop.maxGridSize[1] << "\n";
    os << "  Max blocks in Z-dimension of grid: " << prop.maxGridSize[2] << "\n\n";

    os << "  Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes\n";
    os << "  Registers Per Block: " << prop.regsPerBlock << " 32-bit\n";
    os << "  Warp size: " << prop.warpSize << KNORM << "\n\n";
  }
  return os.str();
}

int gpu_utils::get_gpu_device_pci_id() {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, 0));
  return prop.pciBusID;
}
