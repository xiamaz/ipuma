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
#include <array>
#include <unordered_map>
#include <thread>

#include "hash_funcs.h"
#include "prime.hpp"

namespace kcount_gpu {

using cu_uint64_t = unsigned long long int;
static_assert(sizeof(cu_uint64_t) == 8);
#define KEY_EMPTY (-1)
using count_t = uint32_t;

struct KmerCountsArray {
  count_t data[9];
};

template <int MAX_K>
struct KmerArray {
  static const int N_LONGS = (MAX_K + 31) / 32;
  cu_uint64_t longs[N_LONGS];

  KmerArray() {}
  KmerArray(const uint64_t *x);
};

template <int MAX_K>
struct KmerAndExts {
  KmerArray<MAX_K> kmer;
  count_t count;
  uint8_t left, right;
};

template <int MAX_K>
class HashTableGPUDriver {
  static const int N_LONGS = (MAX_K + 31) / 32;
  struct HashTableDriverState;
  // stores CUDA specific variables
  HashTableDriverState *dstate = nullptr;
  primes::Prime prime;

  int upcxx_rank_me;
  int upcxx_rank_n;
  int kmer_len;
  int num_buff_entries = 0;
  std::vector<KmerArray<MAX_K>> output_keys;
  std::vector<KmerCountsArray> output_vals;
  size_t output_index = 0;
  // Arrays for keys and values. They are separate because the keys get initialized with max number and the vals with zero
  KmerArray<MAX_K> *keys_dev = nullptr;
  KmerCountsArray *vals_dev = nullptr;
  // for buffering elements in the host memory
  KmerAndExts<MAX_K> *elem_buff_host = nullptr;
  // for transferring host memory buffer to device
  KmerAndExts<MAX_K> *elem_buff_dev = nullptr;

  int64_t ht_capacity = 0;
  int64_t num_dropped_inserts = 0;
  int64_t num_attempted_inserts = 0;
  int64_t num_new_inserts = 0;
  int64_t num_purged = 0;
  int num_gpu_calls = 0;

  void insert_kmer_block();

 public:
  HashTableGPUDriver();
  ~HashTableGPUDriver();

  void init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int max_elems, size_t gpu_avail_mem, double &init_time,
            size_t &gpu_bytes_reqd);

  // FIXME: this should be in kmer_dht and the insert_kmer_block should be public
  void insert_kmer(const uint64_t *kmer, count_t kmer_count, char left, char right, bool is_last);
  void done_inserts();

  std::pair<KmerArray<MAX_K> *, KmerCountsArray *> get_next_entry();

  static int get_N_LONGS();

  void get_elapsed_time(double &insert_time, double &kernel_time, double &memcpy_time);
  int64_t get_capacity();
  int64_t get_num_attempted_inserts();
  int64_t get_num_dropped();
  int64_t get_num_entries();
  int64_t get_num_purged();
  int get_num_gpu_calls();
};

}  // namespace kcount_gpu
