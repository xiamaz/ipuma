#include <iostream>
#include <sstream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <array>

#include "gpu_utils.hpp"
#include "upcxx_utils/colors.h"
#include "gpu_common.hpp"

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

std::string gpu_utils::get_gpu_device_description() {
  cudaDeviceProp prop;
  int num_devs = get_num_node_gpus();
  std::ostringstream os;
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

struct gpu_utils::GPUTimerState {
  cudaEvent_t start_event, stop_event;
  float elapsed_t_ms = 0;
};

gpu_utils::GPUTimer::GPUTimer() {
  timer_state = new GPUTimerState();
  cudaEventCreate(&timer_state->start_event);
  cudaEventCreate(&timer_state->stop_event);
  timer_state->elapsed_t_ms = 0;
}

gpu_utils::GPUTimer::~GPUTimer() { delete timer_state; }

void gpu_utils::GPUTimer::start() { cudaEventRecord(timer_state->start_event); }

void gpu_utils::GPUTimer::stop() {
  cudaEventRecord(timer_state->stop_event);
  cudaEventSynchronize(timer_state->stop_event);
  float ms;
  cudaEventElapsedTime(&ms, timer_state->start_event, timer_state->stop_event);
  timer_state->elapsed_t_ms += ms;
}

double gpu_utils::GPUTimer::get_elapsed() { return timer_state->elapsed_t_ms / 1000.0; }
