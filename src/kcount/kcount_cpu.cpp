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

template <int MAX_K>
SeqBlockInserter<MAX_K>::SeqBlockInserter(int qual_offset, int minimizer_len) {}

template <int MAX_K>
SeqBlockInserter<MAX_K>::~SeqBlockInserter() {}

template <int MAX_K>
void SeqBlockInserter<MAX_K>::process_block(unsigned kmer_len, string &seq_block, const vector<kmer_count_t> &depth_block,
                                            dist_object<KmerDHT<MAX_K>> &kmer_dht) {}

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
