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
  int64_t num_kmers = 0;
  int64_t bytes_kmers_sent = 0;
  int64_t bytes_supermers_sent = 0;
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

template <int MAX_K>
HashTableInserter<MAX_K>::HashTableInserter() {}

template <int MAX_K>
HashTableInserter<MAX_K>::~HashTableInserter() {}

template <int MAX_K>
void HashTableInserter<MAX_K>::init(int max_elems) {}

template <int MAX_K>
void HashTableInserter<MAX_K>::init_ctg_kmers(int max_elems) {}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_supermer(const std::string &supermer_seq, kmer_count_t supermer_count) {}

template <int MAX_K>
void HashTableInserter<MAX_K>::flush_inserts() {}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_into_local_hashtable(dist_object<KmerMap<MAX_K>> &local_kmers) {}

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
