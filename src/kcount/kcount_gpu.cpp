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

#include "gpu-utils/gpu_utils.hpp"
#include "kcount-gpu/parse_and_pack.hpp"

using namespace std;
using namespace upcxx_utils;
using namespace upcxx;

static int64_t num_pnp_gpu_waits = 0;
static int64_t bytes_kmers_sent = 0;
static int64_t bytes_supermers_sent = 0;
static int64_t num_kmers = 0;
static int64_t num_Ns = 0;
static int num_block_calls = 0;

static kcount_gpu::ParseAndPackGPUDriver *pnp_gpu_driver;

template <int MAX_K>
void init_parse_and_pack(int qual_offset, int minimizer_len) {
  if (gpu_utils::get_num_node_gpus() <= 0) DIE("GPUs are enabled but no GPU could be configured for kmer counting");
  double init_time;
  num_pnp_gpu_waits = 0;
  bytes_kmers_sent = 0;
  bytes_supermers_sent = 0;
  num_kmers = 0;
  num_Ns = 0;
  num_block_calls = 0;
  pnp_gpu_driver = new kcount_gpu::ParseAndPackGPUDriver(rank_me(), rank_n(), qual_offset, Kmer<MAX_K>::get_k(),
                                                         Kmer<MAX_K>::get_N_LONGS(), minimizer_len, init_time);
  SLOG_GPU("Initialized PnP GPU driver in ", fixed, setprecision(3), init_time, " s\n");
}

void done_parse_and_pack() {
  SLOG_GPU("Number of calls to progress while PnP GPU driver was running on rank 0: ", num_pnp_gpu_waits, "\n");
  SLOG_GPU("Number of calls to PnP GPU kernel for rank 0: ", num_block_calls, "\n");
  auto [gpu_time_tot, gpu_time_kernel] = pnp_gpu_driver->get_elapsed_times();
  SLOG_GPU("Elapsed times for PnP GPU: ", fixed, setprecision(3), " total ", gpu_time_tot, ", kernel ", gpu_time_kernel, "\n");
  delete pnp_gpu_driver;
  pnp_gpu_driver = nullptr;
  auto tot_supermers_bytes_sent = reduce_one(bytes_supermers_sent, op_fast_add, 0).wait();
  auto tot_kmers_bytes_sent = reduce_one(bytes_kmers_sent, op_fast_add, 0).wait();
  SLOG_VERBOSE("Total bytes sent in compressed supermers ", get_size_str(tot_supermers_bytes_sent),
               " and what would have been sent in kmers ", get_size_str(tot_kmers_bytes_sent), " compression is ", fixed,
               setprecision(3), (double)tot_kmers_bytes_sent / tot_supermers_bytes_sent, "\n");
  auto all_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed a total of ", all_num_kmers, " kmers\n");
  auto all_num_Ns = reduce_one(num_Ns, op_fast_add, 0).wait();
  if (all_num_Ns) SLOG_VERBOSE("Found ", perc_str(all_num_Ns, all_num_kmers), " kmers with Ns\n");
}

template <int MAX_K>
void process_block(unsigned kmer_len, string &seq_block, const vector<kmer_count_t> &depth_block,
                   dist_object<KmerDHT<MAX_K>> &kmer_dht) {
  num_block_calls++;
  bool from_ctgs = !depth_block.empty();
  unsigned int num_valid_kmers = 0;
  if (!pnp_gpu_driver->process_seq_block(seq_block, num_valid_kmers))
    DIE("seq length is too high, ", seq_block.length(), " >= ", KCOUNT_GPU_SEQ_BLOCK_SIZE);
  bytes_kmers_sent += sizeof(KmerAndExt<MAX_K>) * num_valid_kmers;
  while (!pnp_gpu_driver->kernel_is_done()) {
    num_pnp_gpu_waits++;
    progress();
  }
  pnp_gpu_driver->pack_seq_block(seq_block);
  int num_targets = (int)pnp_gpu_driver->supermers.size();
  for (int i = 0; i < num_targets; i++) {
    auto target = pnp_gpu_driver->supermers[i].target;
    auto offset = pnp_gpu_driver->supermers[i].offset;
    auto len = pnp_gpu_driver->supermers[i].len;
    Supermer supermer;
    int packed_len = len / 2;
    if (offset % 2 || len % 2) packed_len++;
    supermer.seq = pnp_gpu_driver->packed_seqs.substr(offset / 2, packed_len);
    if (offset % 2) supermer.seq[0] &= 15;
    if ((offset + len) % 2) supermer.seq[supermer.seq.length() - 1] &= 240;
    supermer.count = (from_ctgs ? depth_block[offset + 1] : (kmer_count_t)1);
    bytes_supermers_sent += supermer.get_bytes();
    kmer_dht->add_supermer(supermer, target);
    progress();
  }
}

#define INIT_PNP_K(KMER_LEN) template void init_parse_and_pack<KMER_LEN>(int qual_offset, int minimizer_len)
#define PROCESS_BLOCK_K(KMER_LEN)                                                                            \
  template void process_block(unsigned kmer_len, string &seq_block, const vector<kmer_count_t> &depth_block, \
                              dist_object<KmerDHT<KMER_LEN>> &kmer_dht)
INIT_PNP_K(32);
PROCESS_BLOCK_K(32);
#if MAX_BUILD_KMER >= 64
INIT_PNP_K(64);
PROCESS_BLOCK_K(64);
#endif
#if MAX_BUILD_KMER >= 96
INIT_PNP_K(96);
PROCESS_BLOCK_K(96);
#endif
#if MAX_BUILD_KMER >= 128
INIT_PNP_K(128);
PROCESS_BLOCK_K(128);
#endif
#if MAX_BUILD_KMER >= 160
INIT_PNP_K(160);
PROCESS_BLOCK_K(160);
#endif

#undef PROCESS_BLOCK_K
#undef INIT_PNP_K
