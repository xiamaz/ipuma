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

//#include <stdarg.h>

//#include <algorithm>
//#include <chrono>
//#include <fstream>
//#include <iostream>
//#include <limits>
#include <map>
#include <iterator>
#include <upcxx/upcxx.hpp>

#include "utils.hpp"
#include "bloom.hpp"
#include "kmer.hpp"
#include "upcxx_utils/flat_aggr_store.hpp"
#include "upcxx_utils/three_tier_aggr_store.hpp"

#ifdef ENABLE_GPUS
#include "gpu-utils/gpu_utils.hpp"
#include "kcount-gpu/gpu_hash_table.hpp"
#endif

enum PASS_TYPE { BLOOM_SET_PASS, BLOOM_COUNT_PASS, NO_BLOOM_PASS, CTG_BLOOM_SET_PASS, CTG_KMERS_PASS };

using kcount_gpu::HashTableGPUDriver;

using ext_count_t = uint16_t;
using kmer_count_t = uint16_t;

// global variables to avoid passing dist objs to rpcs
inline double _dynamic_min_depth = 0;
inline int _dmin_thres = 2.0;

struct ExtCounts {
  ext_count_t count_A;
  ext_count_t count_C;
  ext_count_t count_G;
  ext_count_t count_T;

  std::array<std::pair<char, int>, 4> get_sorted() {
    std::array<std::pair<char, int>, 4> counts = {std::make_pair('A', (int)count_A), std::make_pair('C', (int)count_C),
                                                  std::make_pair('G', (int)count_G), std::make_pair('T', (int)count_T)};
    std::sort(std::begin(counts), std::end(counts), [](const auto &elem1, const auto &elem2) {
      if (elem1.second == elem2.second)
        return elem1.first > elem2.first;
      else
        return elem1.second > elem2.second;
    });
    return counts;
  }

  bool is_zero() {
    if (count_A + count_C + count_G + count_T == 0) return true;
    return false;
  }

  ext_count_t inc_with_limit(int count1, int count2) {
    count1 += count2;
    return std::min(count1, (int)numeric_limits<ext_count_t>::max());
  }

  void inc(char ext, int count) {
    switch (ext) {
      case 'A': count_A = inc_with_limit(count_A, count); break;
      case 'C': count_C = inc_with_limit(count_C, count); break;
      case 'G': count_G = inc_with_limit(count_G, count); break;
      case 'T': count_T = inc_with_limit(count_T, count); break;
    }
  }

  void add(ExtCounts &ext_counts) {
    count_A = inc_with_limit(count_A, ext_counts.count_A);
    count_C = inc_with_limit(count_C, ext_counts.count_C);
    count_G = inc_with_limit(count_G, ext_counts.count_G);
    count_T = inc_with_limit(count_T, ext_counts.count_T);
  }

  char get_ext(kmer_count_t count) {
    auto sorted_counts = get_sorted();
    int top_count = sorted_counts[0].second;
    int runner_up_count = sorted_counts[1].second;
    // set dynamic_min_depth to 1.0 for single depth data (non-metagenomes)
    int dmin_dyn = std::max((int)((1.0 - _dynamic_min_depth) * count), _dmin_thres);
    if (top_count < dmin_dyn) return 'X';
    if (runner_up_count >= dmin_dyn) return 'F';
    return sorted_counts[0].first;
  }
};

struct FragElem;

// total bytes: 2+8+8=18
struct KmerCounts {
  // how many times this kmer has occurred: don't need to count beyond 65536
  // count of high quality forward and backward exts
  ExtCounts left_exts;
  ExtCounts right_exts;
  global_ptr<FragElem> uutig_frag;
  kmer_count_t count;
  // the final extensions chosen - A,C,G,T, or F,X
  char left, right;
  bool from_ctg;

  char get_left_ext() { return left_exts.get_ext(count); }

  char get_right_ext() { return right_exts.get_ext(count); }
};

template <int MAX_K>
class KmerDHT {
 public:
  struct KmerAndExt {
    Kmer<MAX_K> kmer;
    kmer_count_t count;
    char left, right;
    UPCXX_SERIALIZED_FIELDS(kmer, count, left, right);
  };

  using KmerMap = HASH_TABLE<Kmer<MAX_K>, KmerCounts>;

 private:
  upcxx::dist_object<KmerMap> kmers;
  // The first bloom filter stores all kmers and is used to check for single occurrences to filter out
  upcxx::dist_object<BloomFilter> bloom_filter1;
  // the second bloom filer stores only kmers that are above the repeat depth, and is used for correctly sizing the kmer hash table
  upcxx::dist_object<BloomFilter> bloom_filter2;
#ifdef ENABLE_GPUS
  dist_object<HashTableGPUDriver<MAX_K>> gpu_driver;
#endif
#ifndef FLAT_AGGR_STORE
  upcxx_utils::ThreeTierAggrStore<Kmer<MAX_K>, dist_object<BloomFilter> &, dist_object<BloomFilter> &> kmer_store_bloom;
  upcxx_utils::ThreeTierAggrStore<KmerAndExt, dist_object<KmerMap> &, dist_object<BloomFilter> &,
                                  dist_object<HashTableGPUDriver<MAX_K>> &>
      kmer_store;
#else
  upcxx_utils::FlatAggrStore<Kmer<MAX_K>, dist_object<BloomFilter> &, dist_object<BloomFilter> &> kmer_store_bloom;
  upcxx_utils::FlatAggrStore<KmerAndExt, dist_object<KmerMap> &, dist_object<BloomFilter> &> kmer_store;
#endif
  int64_t max_kmer_store_bytes;
  int64_t initial_kmer_dht_reservation;
  int64_t my_num_kmers;
  int max_rpcs_in_flight;
  int64_t bloom1_cardinality;
  double estimated_error_rate;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_t;
  PASS_TYPE pass_type;
  bool use_bloom;
  int64_t bytes_sent = 0;
  int minimizer_len = 15;

  static void update_bloom_set(Kmer<MAX_K> kmer, dist_object<BloomFilter> &bloom_filter1, dist_object<BloomFilter> &bloom_filter2);

  static void update_ctg_bloom_set(Kmer<MAX_K> kmer, dist_object<BloomFilter> &bloom_filter1,
                                   dist_object<BloomFilter> &bloom_filter2);

  static void update_bloom_count(KmerAndExt kmer_and_ext, dist_object<KmerMap> &kmers, dist_object<BloomFilter> &bloom_filter,
                                 dist_object<HashTableGPUDriver<MAX_K>> &gpu_driver);

  static void update_count(KmerAndExt kmer_and_ext, dist_object<KmerMap> &kmers, dist_object<BloomFilter> &bloom_filter,
                           dist_object<HashTableGPUDriver<MAX_K>> &gpu_driver);

  static void update_ctg_kmers_count(KmerAndExt kmer_and_ext, dist_object<KmerMap> &kmers, dist_object<BloomFilter> &bloom_filter,
                                     dist_object<HashTableGPUDriver<MAX_K>> &gpu_driver);

 public:
  KmerDHT(uint64_t my_num_kmers, int max_kmer_store_bytes, int max_rpcs_in_flight, bool force_bloom, bool useHHSS);

  void clear();

  void clear_stores();

  ~KmerDHT();

  void reserve_space_and_clear_bloom1();

  bool get_use_bloom();

  pair<int64_t, int64_t> get_bytes_sent();

  void set_pass(PASS_TYPE pass_type);

  int get_minimizer_len();

  int64_t get_num_kmers(bool all = false);

  float max_load_factor();

  void print_load_factor();

  int64_t get_local_num_kmers(void);

  double get_estimated_error_rate();

  upcxx::intrank_t get_kmer_target_rank(const Kmer<MAX_K> &kmer, const Kmer<MAX_K> *kmer_rc = nullptr) const;

  KmerCounts *get_local_kmer_counts(Kmer<MAX_K> &kmer);

#ifdef DEBUG
  bool kmer_exists(Kmer<MAX_K> kmer);
#endif

  void add_kmer(Kmer<MAX_K> kmer, char left_ext, char right_ext, kmer_count_t count, int target_rank = -1);

  void flush_updates();

  void purge_kmers(int threshold);

  void compute_kmer_exts();

  // one line per kmer, format:
  // KMERCHARS LR N
  // where L is left extension and R is right extension, one char, either X, F or A, C, G, T
  // where N is the count of the kmer frequency
  void dump_kmers(int k);

  typename KmerMap::iterator local_kmers_begin();

  typename KmerMap::iterator local_kmers_end();

  int32_t get_time_offset_us();
};

// Reduce compile time by instantiating templates of common types
// extern template declarations are in kmer_dht.hpp
// template instantiations each happen in src/CMakeLists via kmer_dht-extern-template.in.cpp

#define __MACRO_KMER_DHT__(KMER_LEN, MODIFIER) MODIFIER class KmerDHT<KMER_LEN>;

__MACRO_KMER_DHT__(32, extern template);

#if MAX_BUILD_KMER >= 64

__MACRO_KMER_DHT__(64, extern template);

#endif
#if MAX_BUILD_KMER >= 96

__MACRO_KMER_DHT__(96, extern template);

#endif
#if MAX_BUILD_KMER >= 128

__MACRO_KMER_DHT__(128, extern template);

#endif
#if MAX_BUILD_KMER >= 160

__MACRO_KMER_DHT__(160, extern template);

#endif
