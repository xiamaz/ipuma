#pragma once

namespace gpu_utils {

size_t get_tot_gpu_mem();
size_t get_avail_gpu_mem_per_rank(int totRanks, int numDevices = 0);
std::string get_gpu_device_name();
int get_num_node_gpus();

// The first call to cudaMallocHost can take several seconds of real time but no cpu time
// so start it asap (call this in a new thread)
bool initialize_gpu();
bool initialize_gpu(double &time_to_initialize, int &device_count, size_t &total_mem);

}  // namespace gpu_utils
