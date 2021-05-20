#pragma once

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

#include "locassem_struct.hpp"

namespace locassm_driver {

struct accum_data {
  std::vector<uint32_t> ht_sizes;
  std::vector<uint32_t> l_reads_count;
  std::vector<uint32_t> r_reads_count;
  std::vector<uint32_t> ctg_sizes;
};

struct ctg_bucket {
  std::vector<CtgWithReads> ctg_vec;
  accum_data sizes_vec;
  uint32_t l_max, r_max, max_contig_sz;
  ctg_bucket()
      : l_max{0}
      , r_max{0}
      , max_contig_sz{0} {}
  void clear();
};

inline void revcomp(char* str, char* str_rc, int size) {
  int size_rc = 0;
  for (int i = size - 1; i >= 0; i--) {
    switch (str[i]) {
      case 'A': str_rc[size_rc] = 'T'; break;
      case 'C': str_rc[size_rc] = 'G'; break;
      case 'G': str_rc[size_rc] = 'C'; break;
      case 'T': str_rc[size_rc] = 'A'; break;
      case 'N': str_rc[size_rc] = 'N'; break;
      case 'U':
      case 'R':
      case 'Y':
      case 'K':
      case 'M':
      case 'S':
      case 'W':
      case 'B':
      case 'D':
      case 'H':
      case 'V': str_rc[size_rc] = 'N'; break;
      default: std::cout << "Illegal char:" << str[i] << "\n"; break;
    }
    size_rc++;
  }
}

inline std::string revcomp(std::string instr) {
  std::string str_rc;
  for (int i = instr.size() - 1; i >= 0; i--) {
    switch (instr[i]) {
      case 'A': str_rc += 'T'; break;
      case 'C': str_rc += 'G'; break;
      case 'G': str_rc += 'C'; break;
      case 'T': str_rc += 'A'; break;
      case 'N': str_rc += 'N'; break;
      case 'U':
      case 'R':
      case 'Y':
      case 'K':
      case 'M':
      case 'S':
      case 'W':
      case 'B':
      case 'D':
      case 'H':
      case 'V': str_rc += 'N'; break;
      default: std::cout << "Illegal char:" << instr[i] << "\n"; break;
    }
  }

  return str_rc;
}

size_t get_device_mem(int ranks_per_gpu, int gpu_id);
void local_assem_driver(std::vector<CtgWithReads>& data_in, uint32_t max_ctg_size, uint32_t max_read_size, uint32_t max_r_count,
                        uint32_t max_l_count, int mer_len, int max_kmer_len, accum_data& sizes_outliers, int walk_len_limit,
                        int qual_offset, int ranks, int my_rank, int g_rank_me);
int get_gpu_per_node();

}  // namespace locassm_driver
