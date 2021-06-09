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

#pragma once

#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "upcxx_utils/colors.h"
#include "gpu_common.hpp"

// Functions that are common to all cuda code; not to be used by upcxx code

#define cudaErrchk(ans) \
  { gpu_utils::gpu_die((ans), __FILE__, __LINE__); }

namespace gpu_utils {

inline void gpu_die(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << KLRED << "<" << file << ":" << line << "> ERROR:" << KNORM << cudaGetErrorString(code) << "\n";
    std::abort();
    // do not throw exceptions -- does not work properly within progress() throw std::runtime_error(outstr);
  }
}

using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

class QuickTimer {
  timepoint_t t;
  double secs = 0;

 public:
  QuickTimer()
      : secs(0) {}

  void start() { t = std::chrono::high_resolution_clock::now(); }

  void stop() {
    std::chrono::duration<double> t_elapsed = std::chrono::high_resolution_clock::now() - t;
    secs += t_elapsed.count();
  }

  double get_elapsed() { return secs; }
};

}  // namespace gpu_utils