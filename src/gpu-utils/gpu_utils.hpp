#pragma once

namespace gpu_utils {

size_t get_tot_gpu_mem();
size_t get_avail_gpu_mem_per_rank(int totRanks, int numDevices = 0);
size_t get_free_gpu_mem();
std::string get_gpu_device_name();
int get_num_node_gpus();
std::string get_gpu_device_description();
int get_gpu_device_pci_id();

// The first call to cudaMallocHost can take several seconds of real time but no cpu time
// so start it asap (call this in a new thread)
bool initialize_gpu();
bool initialize_gpu(double &time_to_initialize, int &device_count, size_t &total_mem);

struct GPUTimerState;

class GPUTimer {
  GPUTimerState *timer_state = nullptr;

 public:
  GPUTimer();
  ~GPUTimer();
  void start();
  void stop();
  double get_elapsed();
};

}  // namespace gpu_utils
