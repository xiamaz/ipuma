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

#include <upcxx/upcxx.hpp>

#include "upcxx_utils/thread_pool.hpp"
#include "gpu-utils/gpu_utils.hpp"

using namespace upcxx;
using namespace upcxx_utils;

static bool init_gpu_thread = true;
static future<> detect_gpu_fut;
static int num_gpus = -1;

void init_devices() {
  // initialize the GPU and first-touch memory and functions in a new thread as this can take many seconds to complete
  double gpu_startup_duration = 0;
  size_t gpu_mem = 0;
  detect_gpu_fut = execute_in_thread_pool(
      [&gpu_startup_duration, &gpu_mem]() { gpu_utils::initialize_gpu(gpu_startup_duration, num_gpus, gpu_mem); });
  detect_gpu_fut = detect_gpu_fut.then([&gpu_startup_duration, &gpu_mem]() {
    if (num_gpus > 0) {
      SLOG_VERBOSE(KLMAGENTA, "Rank 0 is using ", num_gpus, " GPU/s (", gpu_utils::get_gpu_device_name(), ") on node 0, with ",
                   get_size_str(gpu_mem), " available memory. Detected in ", gpu_startup_duration, " s", KNORM, "\n");
      SLOG_VERBOSE(gpu_utils::get_gpu_device_description());
    } else {
      SDIE("No GPUs available - this build requires GPUs");
    }
  });
}

void done_init_devices() {
  if (init_gpu_thread) {
    Timer t("Waiting for GPU to be initialized (should be noop)");
    init_gpu_thread = false;
    detect_gpu_fut.wait();
  }
  int max_dev_id = reduce_one(num_gpus > 0 ? gpu_utils::get_gpu_device_pci_id() : 0, op_fast_max, 0).wait();
  SLOG_VERBOSE(KLMAGENTA, "Available number of GPUs on this node ", max_dev_id, KNORM, "\n");
}