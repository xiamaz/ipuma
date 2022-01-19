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
#include "devices_ipu.hpp"

#include "ipuma-sw/driver.hpp"
#include "ipuma-sw/ipu_base.h"
#include "ipuma-sw/ipu_batch_affine.h"

#include "aligner_cpu.hpp"
#include "klign.hpp"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

//#define SLOG_GPU(...) SLOG(KLMAGENTA, __VA_ARGS__, KNORM)
#define SLOG_GPU SLOG_VERBOSE

static future<> detect_ipu_fut;
static double gpu_startup_duration = 0;
static int num_gpus_on_node = 0;

void init_devices() {
  // initialize the GPU and first-touch memory and functions in a new thread as this can take many seconds to complete
  detect_ipu_fut = execute_in_thread_pool([]() {
    CPUAligner cpu_aln(false);
    auto& aln_scoring = cpu_aln.aln_scoring;
    ipu::SWConfig config = {aln_scoring.gap_opening, aln_scoring.gap_extending,        aln_scoring.match,
                            -aln_scoring.mismatch,   swatlib::Similarity::nucleicAcid, swatlib::DataType::nucleicAcid};
    ipu::batchaffine::IPUAlgoConfig algoconfig = {
      KLIGN_IPU_TILES,
      KLIGN_IPU_MAXAB_SIZE,
      KLIGN_IPU_MAX_BATCHES,
      KLIGN_IPU_BUFSIZE,
      ipu::batchaffine::VertexType::cpp,
    };
    init_single_ipu(config, algoconfig);
  });
}

void done_init_devices() {
    Timer t("Waiting for IPU to be initialized (should be noop)");
    detect_ipu_fut.wait();
    SWARN("IPU init is DONE");
    // if (gpu_utils::gpus_present()) {
    //   barrier(local_team());
    //   int num_uuids = 0;
    //   unordered_set<string> unique_ids;
    //   dist_object<vector<string>> gpu_uuids(gpu_utils::get_gpu_uuids(), local_team());
    //   for (auto uuid : *gpu_uuids) unique_ids.insert(uuid);
    //   if (!local_team().rank_me()) {
    //     for (int i = 1; i < local_team().rank_n(); i++) {
    //       auto gpu_uuids_i = gpu_uuids.fetch(i).wait();
    //       num_uuids += gpu_uuids_i.size();
    //       for (auto uuid : gpu_uuids_i) {
    //         unique_ids.insert(uuid);
    //       }
    //     }
    //     num_gpus_on_node = unique_ids.size();
    //     SLOG_GPU("Found UUIDs:\n");
    //     for (auto uuid : unique_ids) {
    //       SLOG_GPU(uuid, "\n");
    //     }
    //   }
    //   num_gpus_on_node = broadcast(num_gpus_on_node, 0, local_team()).wait();
    //   // gpu_utils::set_gpu_device(rank_me());
    //   // WARN("Num GPUs on node ", num_gpus_on_node, " gpu avail mem per rank is ", get_size_str(get_avail_gpu_mem_per_rank()),
    //   //     " memory for gpu ", gpu_utils::get_gpu_uuid(), " is ", gpu_utils::get_gpu_avail_mem());
    //   SLOG_GPU("Available number of GPUs on this node ", num_gpus_on_node, "\n");
    //   SLOG_GPU("Rank 0 is using GPU ", gpu_utils::get_gpu_device_name(), " on node 0, with ",
    //            get_size_str(gpu_utils::get_gpu_avail_mem()), " available memory (", get_size_str(get_avail_gpu_mem_per_rank()),
    //            " per rank). Detected in ", gpu_startup_duration, " s\n");
    //   SLOG_GPU(gpu_utils::get_gpu_device_descriptions());
    //   barrier(local_team());
    // } else {
    //   SDIE("No GPUs available - this build requires GPUs");
    // }
}
