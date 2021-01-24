#include <iostream>
#include <sstream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "utils.hpp"

#define cudaErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

static void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::ostringstream os;
    os << "GPU assert " << cudaGetErrorString(code) << " " << file << ":" << line << "\n";
    throw std::runtime_error(os.str());
  }
}

size_t gpu_utils::get_avail_gpu_mem_per_rank(int totRanks, int num_devices) {
  if (num_devices == 0) num_devices = get_num_node_gpus();
  if (!num_devices) return 0;
  int ranksPerDevice = totRanks / num_devices;
  return (get_tot_gpu_mem() * 0.8) / ranksPerDevice;
}

std::string gpu_utils::get_gpu_device_name() {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, 0));
  return prop.name;
}

size_t gpu_utils::get_tot_gpu_mem() {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, 0));
  return prop.totalGlobalMem;
}

int gpu_utils::get_num_node_gpus() {
  int deviceCount = 0;
  auto res = cudaGetDeviceCount(&deviceCount);
  if (res != cudaSuccess) return 0;
  return deviceCount;
}

bool gpu_utils::initialize_gpu(double& time_to_initialize, int& device_count, size_t& total_mem) {
  using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
  double* first_touch;

  timepoint_t t = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed;

  device_count = get_num_node_gpus();
  if (device_count > 0) {
    total_mem = get_tot_gpu_mem();
    cudaErrchk(cudaMallocHost((void**)&first_touch, sizeof(double)));
    cudaErrchk(cudaFreeHost(first_touch));
  }
  elapsed = std::chrono::high_resolution_clock::now() - t;
  time_to_initialize = elapsed.count();
  return device_count > 0;
}

bool gpu_utils::initialize_gpu() {
  double t;
  int c;
  size_t m;
  return initialize_gpu(t, c, m);
}
