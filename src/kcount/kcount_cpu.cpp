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

#include "upcxx_utils.hpp"
#include "kcount.hpp"
#include "kmer_dht.hpp"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

template <int MAX_K>
struct SeqBlockInserter<MAX_K>::SeqBlockInserterState {
  int64_t bytes_kmers_sent = 0;
  int64_t bytes_supermers_sent = 0;
  int64_t num_kmers = 0;
  vector<Kmer<MAX_K>> kmers;
};

template <int MAX_K>
SeqBlockInserter<MAX_K>::SeqBlockInserter(int qual_offset, int minimizer_len) {
  state = new SeqBlockInserterState();
}

template <int MAX_K>
SeqBlockInserter<MAX_K>::~SeqBlockInserter() {
  if (state) delete state;
}

template <int MAX_K>
void SeqBlockInserter<MAX_K>::process_seq(string &seq, kmer_count_t depth, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  if (!depth) depth = 1;
  auto kmer_len = Kmer<MAX_K>::get_k();
  Kmer<MAX_K>::get_kmers(kmer_len, seq, state->kmers);
  for (int i = 0; i < state->kmers.size(); i++) {
    state->bytes_kmers_sent += sizeof(KmerAndExt<MAX_K>);
    Kmer<MAX_K> kmer_rc = state->kmers[i].revcomp();
    if (kmer_rc < state->kmers[i]) state->kmers[i] = kmer_rc;
  }

  Supermer supermer{.seq = seq.substr(0, kmer_len + 1), .count = (kmer_count_t)depth};
  auto prev_target_rank = kmer_dht->get_kmer_target_rank(state->kmers[1]);
  for (int i = 1; i < (int)(seq.length() - kmer_len); i++) {
    auto &kmer = state->kmers[i];
    auto target_rank = kmer_dht->get_kmer_target_rank(kmer);
    if (target_rank == prev_target_rank) {
      supermer.seq += seq[i + kmer_len];
    } else {
      state->bytes_supermers_sent += supermer.get_bytes();
      kmer_dht->add_supermer(supermer, prev_target_rank);
      supermer.seq = seq.substr(i - 1, kmer_len + 2);
      prev_target_rank = target_rank;
    }
  }
  if (supermer.seq.length() >= kmer_len + 2) {
    state->bytes_supermers_sent += supermer.get_bytes();
    kmer_dht->add_supermer(supermer, prev_target_rank);
  }
  state->num_kmers += seq.length() - 2 - kmer_len;
}

template <int MAX_K>
void SeqBlockInserter<MAX_K>::done_processing(dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  auto tot_supermers_bytes_sent = reduce_one(state->bytes_supermers_sent, op_fast_add, 0).wait();
  auto tot_kmers_bytes_sent = reduce_one(state->bytes_kmers_sent, op_fast_add, 0).wait();
  SLOG_VERBOSE("Total bytes sent in compressed supermers ", get_size_str(tot_supermers_bytes_sent), " (compression is ", fixed,
               setprecision(3), (double)tot_kmers_bytes_sent / tot_supermers_bytes_sent, " over kmers)\n");
  auto all_num_kmers = reduce_one(state->num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_kmers, " kmers\n");
}

struct ExtCounts {
  kmer_count_t count_A;
  kmer_count_t count_C;
  kmer_count_t count_G;
  kmer_count_t count_T;

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

  kmer_count_t inc_with_limit(int count1, int count2) {
    count1 += count2;
    return std::min(count1, (int)std::numeric_limits<kmer_count_t>::max());
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

struct KmerExtsCounts {
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
using KmerMapExts = HASH_TABLE<Kmer<MAX_K>, KmerExtsCounts>;

template <int MAX_K>
static void get_kmers_and_exts(Supermer &supermer, vector<KmerAndExt<MAX_K>> &kmers_and_exts) {
  vector<bool> quals;
  quals.resize(supermer.seq.length());
  for (int i = 0; i < supermer.seq.length(); i++) {
    quals[i] = isupper(supermer.seq[i]);
    supermer.seq[i] = toupper(supermer.seq[i]);
  }
  auto kmer_len = Kmer<MAX_K>::get_k();
  vector<Kmer<MAX_K>> kmers;
  Kmer<MAX_K>::get_kmers(kmer_len, supermer.seq, kmers);
  kmers_and_exts.clear();
  for (int i = 1; i < (int)(supermer.seq.length() - kmer_len); i++) {
    Kmer<MAX_K> kmer = kmers[i];
    char left_ext = supermer.seq[i - 1];
    if (!quals[i - 1]) left_ext = '0';
    char right_ext = supermer.seq[i + kmer_len];
    if (!quals[i + kmer_len]) right_ext = '0';
    // get the lexicographically smallest
    Kmer<MAX_K> kmer_rc = kmer.revcomp();
    if (kmer_rc < kmer) {
      kmer = kmer_rc;
      swap(left_ext, right_ext);
      left_ext = comp_nucleotide(left_ext);
      right_ext = comp_nucleotide(right_ext);
    };
    kmers_and_exts.push_back({.kmer = kmer, .count = supermer.count, .left = left_ext, .right = right_ext});
  }
}

template <int MAX_K>
static void insert_supermer_from_read(Supermer &supermer, dist_object<KmerMapExts<MAX_K>> &kmers) {
  auto kmer_len = Kmer<MAX_K>::get_k();
  vector<KmerAndExt<MAX_K>> kmers_and_exts;
  kmers_and_exts.reserve(supermer.seq.length() - kmer_len);
  get_kmers_and_exts(supermer, kmers_and_exts);
  for (auto &kmer_and_ext : kmers_and_exts) {
    // find it - if it isn't found then insert it, otherwise increment the counts
    const auto it = kmers->find(kmer_and_ext.kmer);
    if (it == kmers->end()) {
      KmerExtsCounts kmer_counts = {.left_exts = {0},
                                    .right_exts = {0},
                                    .uutig_frag = nullptr,
                                    .count = kmer_and_ext.count,
                                    .left = 'X',
                                    .right = 'X',
                                    .from_ctg = false};
      kmer_counts.left_exts.inc(kmer_and_ext.left, kmer_and_ext.count);
      kmer_counts.right_exts.inc(kmer_and_ext.right, kmer_and_ext.count);
      auto prev_bucket_count = kmers->bucket_count();
      kmers->insert({kmer_and_ext.kmer, kmer_counts});
      // since sizes are an estimate this could happen, but it will impact performance
      if (prev_bucket_count < kmers->bucket_count())
        SWARN("Hash table on rank 0 was resized from ", prev_bucket_count, " to ", kmers->bucket_count());
    } else {
      auto kmer_count = &it->second;
      int count = kmer_count->count + kmer_and_ext.count;
      if (count > numeric_limits<kmer_count_t>::max()) count = numeric_limits<kmer_count_t>::max();
      kmer_count->count = count;
      kmer_count->left_exts.inc(kmer_and_ext.left, kmer_and_ext.count);
      kmer_count->right_exts.inc(kmer_and_ext.right, kmer_and_ext.count);
    }
  }
}

template <int MAX_K>
static void insert_supermer_from_ctg(Supermer &supermer, dist_object<KmerMapExts<MAX_K>> &kmers) {
  auto kmer_len = Kmer<MAX_K>::get_k();
  vector<KmerAndExt<MAX_K>> kmers_and_exts;
  kmers_and_exts.reserve(supermer.seq.length() - kmer_len);
  get_kmers_and_exts(supermer, kmers_and_exts);
  for (auto &kmer_and_ext : kmers_and_exts) {
    // insert a new kmer derived from the previous round's contigs
    const auto it = kmers->find(kmer_and_ext.kmer);
    bool insert = false;
    if (it == kmers->end()) {
      // if it isn't found then insert it
      insert = true;
    } else {
      auto kmer_counts = &it->second;
      if (!kmer_counts->from_ctg) {
        // existing kmer is from a read, only replace with new contig kmer if the existing kmer is not UU
        char left_ext = kmer_counts->get_left_ext();
        char right_ext = kmer_counts->get_right_ext();
        if (left_ext == 'X' || left_ext == 'F' || right_ext == 'X' || right_ext == 'F') {
          // non-UU, replace
          insert = true;
        }
      } else {
        // existing kmer from previous round's contigs
        // update kmer counts
        if (!kmer_counts->count) {
          // previously must have been a conflict and set to zero, so don't do anything
        } else {
          // will always insert, although it may get purged later for a conflict
          insert = true;
          char left_ext = kmer_counts->get_left_ext();
          char right_ext = kmer_counts->get_right_ext();
          if (left_ext != kmer_and_ext.left || right_ext != kmer_and_ext.right) {
            // if the two contig kmers disagree on extensions, set up to purge by setting the count to 0
            kmer_and_ext.count = 0;
          } else {
            // multiple occurrences of the same kmer derived from different contigs or parts of contigs
            // The only way this kmer could have been already found in the contigs only is if it came from a localassm
            // extension. In which case, all such kmers should not be counted again for each contig, because each
            // contig can use the same reads independently, and the depth will be oversampled.
            kmer_and_ext.count = min(kmer_and_ext.count, kmer_counts->count);
            // kmer_and_ext.count += kmer_counts->count;
          }
        }
      }
    }
    if (insert) {
      kmer_count_t count = kmer_and_ext.count;
      KmerExtsCounts kmer_counts = {
          .left_exts = {0}, .right_exts = {0}, .uutig_frag = nullptr, .count = count, .left = 'X', .right = 'X', .from_ctg = true};
      kmer_counts.left_exts.inc(kmer_and_ext.left, count);
      kmer_counts.right_exts.inc(kmer_and_ext.right, count);
      (*kmers)[kmer_and_ext.kmer] = kmer_counts;
    }
  }
}

template <int MAX_K>
struct HashTableInserter<MAX_K>::HashTableInserterState {
  int64_t initial_max_kmers = 0;
  bool using_ctg_kmers = false;
  dist_object<KmerMapExts<MAX_K>> kmers;

  HashTableInserterState()
      : kmers({}) {}
};

template <int MAX_K>
HashTableInserter<MAX_K>::HashTableInserter() {}

template <int MAX_K>
HashTableInserter<MAX_K>::~HashTableInserter() {
  if (state) delete state;
}

template <int MAX_K>
void HashTableInserter<MAX_K>::init(int max_elems) {
  state = new HashTableInserterState();
  state->using_ctg_kmers = false;
  state->kmers->reserve(max_elems);
  state->initial_max_kmers = max_elems;
}

template <int MAX_K>
void HashTableInserter<MAX_K>::init_ctg_kmers(int max_elems) {
  state->using_ctg_kmers = true;
}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_supermer(const std::string &supermer_seq, kmer_count_t supermer_count) {
  Supermer supermer = {.seq = supermer_seq, .count = supermer_count};
  for (int i = 0; i < supermer.seq.length(); i++) {
    char base = toupper(supermer.seq[i]);
    if (base != 'A' && base != 'C' && base != 'G' && base != 'T' && base != 'N')
      DIE("bad char '", supermer.seq[i], "' in supermer seq int val ", (int)supermer.seq[i], " length ", supermer.seq.length(),
          " supermer ", supermer.seq);
  }
  if (!state->using_ctg_kmers)
    insert_supermer_from_read(supermer, state->kmers);
  else
    insert_supermer_from_ctg(supermer, state->kmers);
}

template <int MAX_K>
void HashTableInserter<MAX_K>::flush_inserts() {
  int64_t tot_num_kmers_est = state->initial_max_kmers * rank_n();
  int64_t tot_num_kmers = reduce_one(state->kmers->size(), op_fast_add, 0).wait();
  SLOG_VERBOSE("Originally reserved ", tot_num_kmers_est, " and now have ", tot_num_kmers, " elements\n");
  auto avg_load_factor = reduce_one(state->kmers->load_factor(), op_fast_add, 0).wait() / upcxx::rank_n();
  SLOG_VERBOSE("kmer DHT load factor: ", avg_load_factor, "\n");
  barrier();
  auto avg_kmers_processed = reduce_one(state->kmers->size(), op_fast_add, 0).wait() / rank_n();
  auto max_kmers_processed = reduce_one(state->kmers->size(), op_fast_max, 0).wait();
  SLOG_VERBOSE("Avg kmers processed per rank ", avg_kmers_processed, " (balance ",
               (double)avg_kmers_processed / max_kmers_processed, ")\n");
}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_into_local_hashtable(dist_object<KmerMap<MAX_K>> &local_kmers) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_good_kmers = state->kmers->size();
  for (auto &elem : *(state->kmers)) {
    auto &kmer_ext_counts = elem.second;
    if ((kmer_ext_counts.count < 2) || (kmer_ext_counts.left_exts.is_zero() && kmer_ext_counts.right_exts.is_zero()))
      num_good_kmers--;
  }
  local_kmers->reserve(num_good_kmers);
  int64_t num_purged = 0;
  for (auto &elem : *(state->kmers)) {
    auto &kmer = elem.first;
    auto &kmer_ext_counts = elem.second;
    if (kmer_ext_counts.count < 2) {
      num_purged++;
      continue;
    }
    KmerCounts kmer_counts = {.uutig_frag = nullptr,
                              .count = kmer_ext_counts.count,
                              .left = kmer_ext_counts.get_left_ext(),
                              .right = kmer_ext_counts.get_right_ext()};
    if (kmer_counts.left == 'X' && kmer_counts.right == 'X') {
      num_purged++;
      continue;
    }
    const auto it = local_kmers->find(kmer);
    if (it != local_kmers->end())
      WARN("Found a duplicate kmer ", kmer.to_string(), " - shouldn't happen: existing count ", it->second.count, " new count ",
           kmer_counts.count);
    local_kmers->insert({kmer, kmer_counts});
  }
  barrier();
  auto tot_num_purged = reduce_one(num_purged, op_fast_add, 0).wait();
  auto tot_num_kmers = reduce_one(state->kmers->size(), op_fast_add, 0).wait();
  SLOG_VERBOSE("Purged ", tot_num_purged, " kmers ( ", perc_str(tot_num_purged, tot_num_kmers), ")\n");
}

template <int MAX_K>
void HashTableInserter<MAX_K>::get_elapsed_time(double &insert_time, double &kernel_time) {}

#define SEQ_BLOCK_INSERTER_K(KMER_LEN) template struct SeqBlockInserter<KMER_LEN>;
#define HASH_TABLE_INSERTER_K(KMER_LEN) template struct HashTableInserter<KMER_LEN>;

SEQ_BLOCK_INSERTER_K(32);
HASH_TABLE_INSERTER_K(32);
#if MAX_BUILD_KMER >= 64
SEQ_BLOCK_INSERTER_K(64);
HASH_TABLE_INSERTER_K(64);
#endif
#if MAX_BUILD_KMER >= 96
SEQ_BLOCK_INSERTER_K(96);
HASH_TABLE_INSERTER_K(96);
#endif
#if MAX_BUILD_KMER >= 128
SEQ_BLOCK_INSERTER_K(128);
HASH_TABLE_INSERTER_K(128);
#endif
#if MAX_BUILD_KMER >= 160
SEQ_BLOCK_INSERTER_K(160);
HASH_TABLE_INSERTER_K(160);
#endif
#undef SEQ_BLOCK_INSERTER_K
#undef HASH_TABLE_INSERTER_K
