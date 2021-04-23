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

#include "hash_funcs.h"
#include "prime.hpp"

namespace kcount_gpu {

using KmerCountsArray = std::array<uint16_t, 9>;

template <int MAX_K>
class KmerArray {
  static const int N_LONGS = (MAX_K + 31) / 32;
  std::array<uint64_t, N_LONGS> longs;

 public:
  KmerArray() {}
  KmerArray(const uint64_t *x);

  void operator=(const uint64_t *kmer);

  const uint64_t *to_array() const { return longs.data(); }

  bool operator==(const KmerArray &o) const;

  size_t hash() const;

  static int get_N_LONGS() { return N_LONGS; }
};

template <int MAX_K>
struct KeyValue {
  KmerArray<MAX_K> key;
  KmerCountsArray val;
};

template <int MAX_K>
struct KmerAndExts {
  KmerArray<MAX_K> kmer;
  uint16_t count;
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
  double t_func = 0, t_malloc = 0, t_cp = 0, t_kernel = 0;
  int num_buff_entries = 0;
  std::vector<KeyValue<MAX_K>> output_elems;
  size_t output_index = 0;
  // array of key-value pairs
  // KeyValue<MAX_K> *elems_dev = nullptr;
  std::vector<KeyValue<MAX_K>> elems_dev;
  // locks for mutexes
  int8_t *locks_dev = nullptr;
  // for buffering elements in the host memory
  KmerAndExts<MAX_K> *elem_buff_host = nullptr;

  int64_t ht_capacity = 0;
  int64_t num_elems = 0;
  int64_t num_dropped_entries = 0;
  int64_t num_attempted_inserts = 0;

  void insert_kmer_block(int64_t &num_new_elems, int64_t &num_inserts, int64_t &num_dropped);

 public:
  HashTableGPUDriver();
  ~HashTableGPUDriver();

  bool init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int max_elems, int gpu_avail_mem, double &init_time,
            size_t &gpu_bytes_reqd);

  void insert_kmer(const uint64_t *kmer, uint16_t kmer_count, char left, char right);

  void done_inserts();

  KeyValue<MAX_K> *get_next_entry();

  static int get_N_LONGS();

  double get_kernel_elapsed_time();

  int64_t get_num_elems();
  int64_t get_num_inserts();
  int64_t get_num_dropped();
  double get_load_factor();
};

}  // namespace kcount_gpu

namespace std {

template <int MAX_K>
struct hash<kcount_gpu::KmerArray<MAX_K>> {
  static const int N_LONGS = (MAX_K + 31) / 32;

  size_t operator()(kcount_gpu::KmerArray<MAX_K> const &kmer_array) const { return kmer_array.hash(); }
};

}  // namespace std
