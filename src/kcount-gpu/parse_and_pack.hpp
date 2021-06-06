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

#include <vector>

namespace kcount_gpu {

struct ParseAndPackDriverState;

struct SupermerInfo {
  int target;
  int offset;
  uint16_t len;
};

class ParseAndPackGPUDriver {
  // this opaque data type stores CUDA specific variables
  ParseAndPackDriverState *dstate = nullptr;

  int upcxx_rank_me;
  int upcxx_rank_n;
  int max_kmers;
  int kmer_len;
  int qual_offset;
  int num_kmer_longs;
  int minimizer_len;
  double t_func = 0, t_malloc = 0, t_cp = 0, t_kernel = 0;
  char *dev_seqs;
  int *dev_kmer_targets;

  SupermerInfo *dev_supermers;
  char *dev_packed_seqs;
  unsigned int *dev_num_supermers;
  unsigned int *dev_num_valid_kmers;

 public:
  std::vector<SupermerInfo> supermers;
  std::string packed_seqs;

  ParseAndPackGPUDriver(int upcxx_rank_me, int upcxx_rank_n, int qual_offset, int kmer_len, int num_kmer_longs, int minimizer_len,
                        double &init_time);
  ~ParseAndPackGPUDriver();
  bool process_seq_block(const std::string &seqs, unsigned int &num_valid_kmers);
  void pack_seq_block(const std::string &seqs);
  bool kernel_is_done();

  std::tuple<double, double> get_elapsed_times();
};

}  // namespace kcount_gpu
