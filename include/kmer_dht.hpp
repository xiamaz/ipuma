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

#include <map>
#include <iterator>
#include <upcxx/upcxx.hpp>

#include "utils.hpp"
#include "kmer.hpp"
#include "upcxx_utils/flat_aggr_store.hpp"
#include "upcxx_utils/three_tier_aggr_store.hpp"

#ifdef ENABLE_GPUS
#include "gpu-utils/gpu_utils.hpp"
#include "kcount-gpu/gpu_hash_table.hpp"
#else
// fake it out
namespace kcount_gpu {
template <int MAX_K>
struct HashTableGPUDriver {};
}  // namespace kcount_gpu
#endif

enum PASS_TYPE { READ_KMERS_PASS, CTG_KMERS_PASS };

using ext_count_t = uint16_t;
using kmer_count_t = uint16_t;

using kcount_gpu::HashTableGPUDriver;

// global variables to avoid passing dist objs to rpcs
inline int _dmin_thres = 2.0;

struct ExtCounts {
  ext_count_t count_A;
  ext_count_t count_C;
  ext_count_t count_G;
  ext_count_t count_T;

  void set(uint16_t *counts) {
    count_A = counts[0];
    count_C = counts[1];
    count_G = counts[2];
    count_T = counts[3];
  }

  void set(uint32_t *counts) {
    count_A = static_cast<uint16_t>(counts[0]);
    count_C = static_cast<uint16_t>(counts[1]);
    count_G = static_cast<uint16_t>(counts[2]);
    count_T = static_cast<uint16_t>(counts[3]);
  }

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
    return std::min(count1, (int)std::numeric_limits<ext_count_t>::max());
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
    int dmin_dyn = std::max((int)((1.0 - DYN_MIN_DEPTH) * count), _dmin_thres);
    if (top_count < dmin_dyn) return 'X';
    if (runner_up_count >= dmin_dyn) return 'F';
    return sorted_counts[0].first;
  }

  string to_string() {
    ostringstream os;
    os << count_A << "," << count_C << "," << count_G << "," << count_T;
    return os.str();
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
struct KmerAndExt {
  Kmer<MAX_K> kmer;
  kmer_count_t count;
  char left, right;
  UPCXX_SERIALIZED_FIELDS(kmer, count, left, right);
};

struct Supermer {
  string seq;
  string quals;
  kmer_count_t count;

  UPCXX_SERIALIZED_FIELDS(seq, quals, count);

  // int get_bytes_compressed() { return ceil((double)seq.length() * 0.375) + sizeof(int) + sizeof(kmer_count_t); }
  int get_bytes_compressed() { return ceil((double)seq.length() * 2.0) + sizeof(int) + sizeof(kmer_count_t); }

  /*
  static pair<uint8_t *, size_t> pack_dna_seq(const string &seq) {
    size_t bits = 4;
    size_t packed_len = (seq.length() + bits - 1) / bits;
    uint8_t *packed_seq = new uint8_t[packed_len];
    memset(packed_seq, 0, packed_len);
    for (int i = 0; i < seq.length(); ++i) {
      int j = i % bits;
      int l = i / bits;
      assert(seq[i] != '\0');
      uint8_t x;
      x = ((seq[i]) & 4) >> 1;
      packed_seq[l] |= ((x + ((x ^ (seq[i] & 2)) >> 1)) << (2 * (bits - 1 - j)));
    }
    return {packed_seq, packed_len};
  }

  static string unpack_dna_seq(uint8_t *packed_seq, size_t seq_len) {
    size_t bits = 4;
    string seq;
    seq.resize(seq_len);
    for (int i = 0; i < seq_len; i++) {
      int j = i % bits;
      int l = i / bits;
      switch (((packed_seq[l]) >> (2 * (bits - 1 - j))) & 0x03) {
        case 0x00: seq[i] = 'A'; break;
        case 0x01: seq[i] = 'C'; break;
        case 0x02: seq[i] = 'G'; break;
        case 0x03: seq[i] = 'T'; break;
      }
    }
    return seq;
  }

  static size_t get_packed_dna_seq_size(size_t len) { return (len + 3) / 4; }

  static pair<uint8_t *, size_t> pack_bool_seq(const string &seq) {
    uint8_t bits[8] = {0};
    int n_whole = (seq.length() / 8) * 8;
    size_t packed_len = (seq.length() + 7) / 8;
    uint8_t *packed_seq = new uint8_t[packed_len];
    for (int i = 0; i < n_whole; i += 8) {
      bits[0] = seq[i + 7];
      bits[1] = seq[i + 6] << 1;
      bits[2] = seq[i + 5] << 2;
      bits[3] = seq[i + 4] << 3;
      bits[4] = seq[i + 3] << 4;
      bits[5] = seq[i + 2] << 5;
      bits[6] = seq[i + 1] << 6;
      bits[7] = seq[i + 0] << 7;
      packed_seq[i / 8] = bits[0] | bits[1] | bits[2] | bits[3] | bits[4] | bits[5] | bits[6] | bits[7];
    }
    if (seq.length() > n_whole) {
      int last_byte = n_whole / 8;
      memset(packed_seq + last_byte, 0, 8);
      for (int i = n_whole; i < seq.length(); ++i) {
        packed_seq[last_byte] |= (seq[i] ? 1 << (7 - (i - n_whole)) : 0);
      }
    }
    return {packed_seq, packed_len};
  }

  static string unpack_bool_seq(uint8_t *packed_seq, size_t seq_len) {
    string seq;
    seq.resize(seq_len);
    int n_whole = (seq.length() / 8) * 8;
    size_t packed_len = (seq.length() + 7) / 8;
    for (int i = 0; i < n_whole; i += 8) {
      uint8_t bits = packed_seq[i / 8];
      seq[i + 7] = (bits)&0x01;
      seq[i + 6] = (bits >> 1) & 0x01;
      seq[i + 5] = (bits >> 2) & 0x01;
      seq[i + 4] = (bits >> 3) & 0x01;
      seq[i + 3] = (bits >> 4) & 0x01;
      seq[i + 2] = (bits >> 5) & 0x01;
      seq[i + 1] = (bits >> 6) & 0x01;
      seq[i + 0] = (bits >> 7) & 0x01;
    }
    if (seq.length() > n_whole) {
      int last_byte = n_whole / 8;
      uint8_t bits = packed_seq[last_byte];
      for (int i = n_whole; i < seq.length(); ++i) {
        seq[i] = (bits >> (7 - (i - n_whole))) & 0x01;
      }
    }
    return seq;
  }

  static size_t get_packed_bool_seq_size(size_t len) { return (len + 7) / 8; }

  struct upcxx_serialization {
    template <typename Writer>
    static void serialize(Writer &writer, Supermer const &supermer) {
      uint16_t seq_len = supermer.seq.length();
      writer.write(seq_len);
      auto [packed_seq, packed_len] = pack_dna_seq(supermer.seq);
      for (int i = 0; i < packed_len; i++) writer.write(packed_seq[i]);
      delete[] packed_seq;
      auto [packed_quals, packed_quals_len] = pack_bool_seq(supermer.quals);
      for (int i = 0; i < packed_quals_len; i++) writer.write(packed_quals[i]);
      delete[] packed_quals;
      writer.write(supermer.count);
    }

    template <typename Reader>
    static Supermer *deserialize(Reader &reader, void *storage) {
      Supermer *supermer = new (storage) Supermer();
      uint16_t seq_len = reader.template read<uint16_t>();
      size_t packed_len = get_packed_dna_seq_size(seq_len);
      uint8_t *packed_seq = new uint8_t[packed_len];
      for (int i = 0; i < packed_len; i++) packed_seq[i] = reader.template read<uint8_t>();
      supermer->seq = unpack_dna_seq(packed_seq, seq_len);
      delete[] packed_seq;
      size_t packed_quals_len = get_packed_bool_seq_size(seq_len);
      uint8_t *packed_quals = new uint8_t[packed_quals_len];
      for (int i = 0; i < packed_quals_len; i++) packed_quals[i] = reader.template read<uint8_t>();
      supermer->quals = unpack_bool_seq(packed_quals, seq_len);
      delete[] packed_quals;
      supermer->count = reader.template read<uint16_t>();
      return supermer;
    }
  };

  size_t get_packed_size() const {
    return 2 + get_packed_dna_seq_size(seq.length()) + get_packed_bool_seq_size(seq.length()) + 2;
  }
  */
};

template <int MAX_K>
class KmerDHT {
 public:
  using KmerMap = HASH_TABLE<Kmer<MAX_K>, KmerCounts>;

 private:
  upcxx::dist_object<KmerMap> kmers;
  dist_object<HashTableGPUDriver<MAX_K>> ht_gpu_driver;

  upcxx_utils::ThreeTierAggrStore<Supermer, dist_object<KmerMap> &, dist_object<HashTableGPUDriver<MAX_K>> &> kmer_store;
  int64_t max_kmer_store_bytes;
  int64_t initial_kmer_dht_reservation;
  int64_t my_num_kmers;
  int max_rpcs_in_flight;
  double estimated_error_rate;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_t;
  PASS_TYPE pass_type;
  int64_t bytes_sent = 0;

  int minimizer_len = 15;
  bool using_ctg_kmers = false;

  static void update_count(Supermer supermer, dist_object<KmerMap> &kmers, dist_object<HashTableGPUDriver<MAX_K>> &ht_gpu_driver);

  static void update_ctg_kmers_count(Supermer supermer, dist_object<KmerMap> &kmers,
                                     dist_object<HashTableGPUDriver<MAX_K>> &ht_gpu_driver);

  void purge_kmers(int threshold);

  void insert_from_gpu_hashtable();

  static void get_kmers_and_exts(const Supermer &supermer, vector<KmerAndExt<MAX_K>> &kmers_and_exts);

 public:
  KmerDHT(uint64_t my_num_kmers, int max_kmer_store_bytes, int max_rpcs_in_flight, bool useHHSS);

  void clear();

  void clear_stores();

  ~KmerDHT();

  pair<int64_t, int64_t> get_bytes_sent();

  void init_ctg_kmers(int64_t max_elems);

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

  void add_supermer(Supermer &supermer, int target_rank);

  void flush_updates();

  void compute_kmer_exts();

  // one line per kmer, format:
  // KMERCHARS LR N
  // where L is left extension and R is right extension, one char, either X, F or A, C, G, T
  // where N is the count of the kmer frequency
  void dump_kmers();

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
