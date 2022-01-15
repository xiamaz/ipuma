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

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>
#include <string_view>
#include <unordered_set>

#include <algorithm>
#include <iostream>
#include <thread>
#include <upcxx/upcxx.hpp>

#include "ipuma-sw/popinit.hpp"
#include "klign.hpp"
#include "kmer.hpp"
#include "ssw.hpp"
#include "upcxx_utils/fixed_size_cache.hpp"
#include "upcxx_utils/limit_outstanding.hpp"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/progress_bar.hpp"
#include "upcxx_utils/three_tier_aggr_store.hpp"
#include "utils.hpp"
#include "zstr.hpp"
#include "aligner_cpu.hpp"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

using cid_t = int64_t;

static IntermittentTimer aln_cpu_bypass_timer(__FILENAME__ + string(":CPU_BSW-bypass"));
static IntermittentTimer fetch_ctg_seqs_timer(__FILENAME__ + string(":Fetch ctg seqs"));
static IntermittentTimer compute_alns_timer(__FILENAME__ + string(":Compute alns"));
static IntermittentTimer get_ctgs_timer(__FILENAME__ + string(":Get ctgs with kmer"));
static IntermittentTimer aln_kernel_timer(__FILENAME__ + string(":GPU_BSW"));

void init_aligner(AlnScoring &aln_scoring, int rlen_limit);
void cleanup_aligner();
void kernel_align_block(CPUAligner &cpu_aligner, vector<Aln> &kernel_alns, vector<string> &ctg_seqs, vector<string> &read_seqs,
                        Alns *alns, future<> &active_kernel_fut, int read_group_id, int max_clen, int max_rlen,
                        IntermittentTimer &aln_kernel_timer);

struct CtgLoc {
  cid_t cid;
  global_ptr<char> seq_gptr;
  int clen;
  float depth;
  int pos_in_ctg;
  bool is_rc;
};

struct ReadAndCtgLoc {
  int pos_in_read;
  bool read_is_rc;  // FIXME  pack this bool better, if possible
  CtgLoc ctg_loc;
};

template <int MAX_K>
struct KmerAndCtgLoc {
  Kmer<MAX_K> kmer;
  CtgLoc ctg_loc;
  UPCXX_SERIALIZED_FIELDS(kmer, ctg_loc);
};

struct ReadRecord {
  string id;
  string seq;

  HASH_TABLE<cid_t, ReadAndCtgLoc> aligned_ctgs_map;

  ReadRecord(const string &id, const string &seq)
      : id(id)
      , seq(seq)
      , aligned_ctgs_map{} {}
  ~ReadRecord() {
    id.clear();
    seq.clear();
  }
  bool is_valid() const { return !id.empty() && !seq.empty(); }
};

struct KmerToRead {
  ReadRecord *read_record;
  int pos_in_read;
  bool is_rc;
};

template <int MAX_K>
class KmerCtgDHT {
  // maps a kmer to a contig - the first part of the pair is set to true if this is a conflict,
  // with a kmer mapping to multiple contigs
  using local_kmer_map_t = HASH_TABLE<Kmer<MAX_K>, pair<bool, CtgLoc>>;
  using kmer_map_t = dist_object<local_kmer_map_t>;
  kmer_map_t kmer_map;
  vector<global_ptr<char>> global_ctg_seqs;
  ThreeTierAggrStore<KmerAndCtgLoc<MAX_K>> kmer_store;
  dist_object<int64_t> num_dropped_seed_to_ctgs;

 public:
  size_t kmer_seed_lookups = 0;
  size_t unique_kmer_seed_lookups = 0;
  unsigned kmer_len;

  KmerCtgDHT(int max_store_size, int max_rpcs_in_flight)
      : kmer_map({})
      , global_ctg_seqs({})
      , kmer_store()
      , num_dropped_seed_to_ctgs(0) {
    kmer_len = Kmer<MAX_K>::get_k();
    kmer_store.set_size("insert ctg seeds", max_store_size, max_rpcs_in_flight);
    kmer_store.set_update_func([&kmer_map = this->kmer_map,
                                &num_dropped_seed_to_ctgs = this->num_dropped_seed_to_ctgs](KmerAndCtgLoc<MAX_K> kmer_and_ctg_loc) {
      CtgLoc ctg_loc = kmer_and_ctg_loc.ctg_loc;
      const auto it = kmer_map->find(kmer_and_ctg_loc.kmer);
      if (it == kmer_map->end()) {
        kmer_map->insert({kmer_and_ctg_loc.kmer, {false, ctg_loc}});
      } else {
        // in this case, we have a conflict, i.e. the kmer maps to multiple contigs
        it->second.first = true;
        (*num_dropped_seed_to_ctgs)++;
      }
    });
  }

  void clear() {
    LOG("aggregated kmer seed lookups ", perc_str(kmer_seed_lookups - unique_kmer_seed_lookups, kmer_seed_lookups), ", total ",
        kmer_seed_lookups, " approx ",
        get_size_str(unique_kmer_seed_lookups * sizeof(Kmer<MAX_K>) + kmer_seed_lookups * sizeof(KmerToRead)), "\n");
    for (auto &gptr : global_ctg_seqs) upcxx::deallocate(gptr);
    local_kmer_map_t().swap(*kmer_map);  // release all memory
    kmer_store.clear();
  }

  ~KmerCtgDHT() { clear(); }

  void reserve_ctg_seqs(size_t sz) { global_ctg_seqs.reserve(sz); }

  global_ptr<char> add_ctg_seq(string seq) {
    auto seq_gptr = upcxx::allocate<char>(seq.length() + 1);
    global_ctg_seqs.push_back(seq_gptr);  // remember to dealloc!
    strcpy(seq_gptr.local(), seq.c_str());
    return seq_gptr;
  }

  void reserve(int64_t mysize) { kmer_map->reserve(mysize); }

  int64_t size() const { return kmer_map->size(); }

  intrank_t get_target_rank(const Kmer<MAX_K> &kmer) const { return std::hash<Kmer<MAX_K>>{}(kmer) % rank_n(); }

  int64_t get_num_kmers(bool all = false) {
    if (!all) return reduce_one(kmer_map->size(), op_fast_add, 0).wait();
    return reduce_all(kmer_map->size(), op_fast_add).wait();
  }

  int64_t get_num_dropped_seed_to_ctgs(bool all = false) {
    if (!all) return reduce_one(*num_dropped_seed_to_ctgs, op_fast_add, 0).wait();
    return reduce_all(*num_dropped_seed_to_ctgs, op_fast_add).wait();
  }

  void add_kmer(const Kmer<MAX_K> &kmer_fw, CtgLoc &ctg_loc) {
    Kmer<MAX_K> kmer_rc = kmer_fw.revcomp();
    ctg_loc.is_rc = false;
    const Kmer<MAX_K> *kmer_lc = &kmer_fw;
    if (kmer_rc < kmer_fw) {
      kmer_lc = &kmer_rc;
      ctg_loc.is_rc = true;
    }
    KmerAndCtgLoc<MAX_K> kmer_and_ctg_loc = {*kmer_lc, ctg_loc};
    kmer_store.update(get_target_rank(*kmer_lc), kmer_and_ctg_loc);

    // FIXME: add kmer-ctg to local hash table for quick lookups
  }

  void flush_add_kmers() {
    BarrierTimer timer(__FILEFUNC__, false);  // barrier on exit, not entrance
    kmer_store.flush_updates();
    kmer_store.clear();
  }

  future<vector<KmerAndCtgLoc<MAX_K>>> get_ctgs_with_kmers(int target_rank, vector<Kmer<MAX_K>> &kmers) {
    DBG_VERBOSE("Sending request for ", kmers.size(), " to ", target_rank, "\n");
    return rpc(
        target_rank,
        [](vector<Kmer<MAX_K>> kmers, kmer_map_t &kmer_map) {
          vector<KmerAndCtgLoc<MAX_K>> kmer_ctg_locs;
          kmer_ctg_locs.reserve(kmers.size());
          for (auto &kmer : kmers) {
            assert(kmer.is_least());
            assert(kmer.is_valid());
            const auto it = kmer_map->find(kmer);
            if (it == kmer_map->end()) continue;
            // skip conflicts
            if (it->second.first) continue;
            // now add it
            kmer_ctg_locs.push_back({kmer, it->second.second});
          }
          DBG_VERBOSE("processed get_ctgs_with_kmers ", kmers.size(), " ", get_size_str(kmers.size() * sizeof(Kmer<MAX_K>)),
                      ", returning ", kmer_ctg_locs.size(), " ", get_size_str(kmer_ctg_locs.size() * sizeof(KmerAndCtgLoc<MAX_K>)),
                      "\n");
          return kmer_ctg_locs;
        },
        kmers, kmer_map);
  }

  void dump_ctg_kmers() {
    BarrierTimer timer(__FILEFUNC__, false);  // barrier on exit not entrance
    string dump_fname = "ctg_kmers-" + to_string(kmer_len) + ".txt.gz";
    get_rank_path(dump_fname, rank_me());
    zstr::ofstream dump_file(dump_fname);
    ostringstream out_buf;
    ProgressBar progbar(kmer_map->size(), "Dumping kmers to " + dump_fname);
    int64_t i = 0;
    for (auto &elem : *kmer_map) {
      // FIXME this was broken when I got here.... made my best guess as to what the fields are supposed to be. -Rob
      auto &pair_ctg_loc = elem.second;
      auto &ctg_loc = pair_ctg_loc.second;
      out_buf << elem.first << " " << ctg_loc.cid << " " << ctg_loc.clen << " " << ctg_loc.depth << " " << ctg_loc.pos_in_ctg << " "
              << ctg_loc.is_rc << "\n";
      i++;
      if (!(i % 1000)) {
        dump_file << out_buf.str();
        out_buf = ostringstream();
      }
      progbar.update();
    }
    if (!out_buf.str().empty()) dump_file << out_buf.str();
    dump_file.close();
    progbar.done();
    SLOG_VERBOSE("Dumped ", this->get_num_kmers(), " kmers\n");
  }
};

class Aligner {
  int64_t num_alns;
  int64_t num_perfect_alns;
  int64_t num_overlaps;
  int kmer_len;

  vector<Aln> kernel_alns;
  vector<string> ctg_seqs;
  vector<string> read_seqs;

  int64_t total_ctg_len = 0;
  int64_t total_read_len = 0;

  future<> active_kernel_fut;

  int64_t max_clen = 0;
  int64_t max_rlen = 0;
  CPUAligner cpu_aligner;

  Alns *alns;

 private:
  int64_t ctg_bytes_fetched = 0;
  // using ctg_cache_t = FixedSizeCache<cid_t, string>;
  using ctg_cache_t = HASH_TABLE<cid_t, string>;
  ctg_cache_t ctg_cache;
  std::unordered_set<cid_t> local_ctgs;
  int64_t ctg_cache_hits = 0;
  int64_t ctg_lookups = 0;
  int64_t ctg_local_hits = 0;

  void align_read(const string &rname, int64_t cid, const string_view &rseq, const string_view &cseq, int rstart, int rlen,
                  int cstart, int clen, char orient, int overlap_len, int read_group_id) {
    if (cseq.compare(0, overlap_len, rseq, rstart, overlap_len) == 0) {
      num_perfect_alns++;
      int rstop = rstart + overlap_len;
      int cstop = cstart + overlap_len;
      if (orient == '-') switch_orient(rstart, rstop, rlen);
      int score1 = overlap_len * cpu_aligner.aln_scoring.match;
      int identity = 100 * score1 / cpu_aligner.aln_scoring.match / rlen;
      Aln aln(rname, cid, rstart, rstop, rlen, cstart, cstop, clen, orient, score1, 0, identity, 0, read_group_id);
      assert(aln.is_valid());
      if (cpu_aligner.ssw_filter.report_cigar) set_sam_string(aln, rseq, to_string(overlap_len) + "=");  // exact match '=' not 'M'
      alns->add_aln(aln);
    } else {
      max_clen = max((int64_t)cseq.size(), max_clen);
      max_rlen = max((int64_t)rseq.size(), max_rlen);
      int64_t num_alns = kernel_alns.size() + 1;
      unsigned max_matrix_size = (max_clen + 1) * (max_rlen + 1);
      int64_t tot_mem_est = num_alns * (max_clen + max_rlen + 2 * sizeof(int) + 5 * sizeof(short));
      // contig is the ref, read is the query - done this way so that we can potentially do multiple alns to each read
      // this is also the way it's done in meraligner
      kernel_alns.emplace_back(rname, cid, 0, 0, rlen, cstart, 0, clen, orient);
      ctg_seqs.emplace_back(cseq);
      read_seqs.emplace_back(rseq);
      if (num_alns >= KLIGN_GPU_BLOCK_SIZE) {
        // for (auto &&x : ctg_seqs) {
        //   SLOG_VERBOSE("HistC: ", x.size(), "\n"); 
        // }
        // for (auto &&x : read_seqs) {
        //   SLOG_VERBOSE("HistR: ", x.size(), "\n"); 
        // }
         
        kernel_align_block(cpu_aligner, kernel_alns, ctg_seqs, read_seqs, alns, active_kernel_fut, read_group_id, max_clen,
                           max_rlen, aln_kernel_timer);
        clear_aln_bufs();
      }
    }
  }

  void clear() {
    if (kernel_alns.size() || !active_kernel_fut.ready())
      DIE("clear called with alignments in the buffer or active kernel - was flush_remaining called before destrutor?\n");
    clear_aln_bufs();
    ctg_cache.clear();
  }

 public:
  Aligner(int kmer_len, Alns &alns, int rlen_limit, bool compute_cigar, int all_num_ctgs)
      : num_alns(0)
      , num_perfect_alns(0)
      , num_overlaps(0)
      , kmer_len(kmer_len)
      , kernel_alns({})
      , ctg_seqs({})
      , read_seqs({})
      , active_kernel_fut(make_future())
      , cpu_aligner(compute_cigar)
      , alns(&alns) {
    // ctg_cache.set_invalid_key(std::numeric_limits<cid_t>::max());
    ctg_cache.reserve(2 * all_num_ctgs / rank_n());
    init_aligner(cpu_aligner.aln_scoring, rlen_limit);
  }

  ~Aligner() {
    clear();
    cleanup_aligner();
  }

  int64_t get_num_perfect_alns(bool all = false) {
    if (!all) return reduce_one(num_perfect_alns, op_fast_add, 0).wait();
    return reduce_all(num_perfect_alns, op_fast_add).wait();
  }

  int64_t get_num_alns(bool all = false) {
    if (!all) return reduce_one(num_alns, op_fast_add, 0).wait();
    return reduce_all(num_alns, op_fast_add).wait();
  }

  int64_t get_num_overlaps(bool all = false) {
    if (!all) return reduce_one(num_overlaps, op_fast_add, 0).wait();
    return reduce_all(num_overlaps, op_fast_add).wait();
  }

  void clear_aln_bufs() {
    kernel_alns.clear();
    ctg_seqs.clear();
    read_seqs.clear();
    max_clen = 0;
    max_rlen = 0;
  }

  void flush_remaining(int read_group_id) {
    BaseTimer t(__FILEFUNC__);
    t.start();
    auto num = kernel_alns.size();
    if (num) {
      kernel_align_block(cpu_aligner, kernel_alns, ctg_seqs, read_seqs, alns, active_kernel_fut, read_group_id, max_clen, max_rlen,
                         aln_kernel_timer);
      clear_aln_bufs();
    }
    bool is_ready = active_kernel_fut.ready();
    active_kernel_fut.wait();
    t.stop();
  }

  void compute_alns_for_read(HASH_TABLE<cid_t, ReadAndCtgLoc> *aligned_ctgs_map, const string &rname, const string &rseq_fw,
                             int read_group_id) {
    int rlen = rseq_fw.length();
    string rseq_rc;
    string tmp_ctg;
    for (auto &elem : *aligned_ctgs_map) {
      progress();
      int pos_in_read = elem.second.pos_in_read;
      bool read_kmer_is_rc = elem.second.read_is_rc;
      CtgLoc ctg_loc = elem.second.ctg_loc;
      char orient = '+';
      string_view rseq_ptr;
      if (ctg_loc.is_rc != read_kmer_is_rc) {
        // it's revcomp in either contig or read, but not in both or neither
        orient = '-';
        pos_in_read = rlen - (kmer_len + pos_in_read);
        if (rseq_rc.empty()) rseq_rc = revcomp(rseq_fw);
        rseq_ptr = string_view(rseq_rc);
      } else {
        rseq_ptr = string_view(rseq_fw);
      }
      // calculate available bases before and after the seeded kmer
      int ctg_bases_left_of_kmer = ctg_loc.pos_in_ctg;
      int ctg_bases_right_of_kmer = ctg_loc.clen - ctg_bases_left_of_kmer - kmer_len;
      int read_bases_left_of_kmer = pos_in_read;
      int read_bases_right_of_kmer = rlen - kmer_len - pos_in_read;
      int left_of_kmer = min(read_bases_left_of_kmer, ctg_bases_left_of_kmer);
      int right_of_kmer = min(read_bases_right_of_kmer, ctg_bases_right_of_kmer);

      int cstart = ctg_loc.pos_in_ctg - left_of_kmer;
      int rstart = pos_in_read - left_of_kmer;
      int overlap_len = left_of_kmer + kmer_len + right_of_kmer;

      // use the whole read, to account for possible indels
      string_view read_subseq = rseq_ptr.substr(0, rlen);

      assert(cstart >= 0 && cstart + overlap_len <= ctg_loc.clen);
      assert(overlap_len <= 2 * rlen);

      string_view ctg_seq;
      bool found = false;
      bool on_node = ctg_loc.seq_gptr.is_local();
#ifdef DEBUG
      // test both on node and off node ctg cache
      if (on_node) on_node = (ctg_loc.seq_gptr.where() % 2) == (rank_me() % 2);
#endif
      if (on_node) {
        // on same node already
        ctg_seq = string_view(ctg_loc.seq_gptr.local(), ctg_loc.clen);
        ctg_local_hits++;
        auto it = local_ctgs.find(ctg_loc.cid);
        if (it == local_ctgs.end())
          local_ctgs.insert(ctg_loc.cid);
        else
          found = true;
      } else {
        ctg_lookups++;
        auto it = ctg_cache.find(ctg_loc.cid);
        auto get_start = cstart, get_len = overlap_len;
        if (it != ctg_cache.end()) {
          string &ctg_str = it->second;  // the actual underlying string, not view
          ctg_seq = string_view(ctg_str.data(), ctg_str.size());
          found = true;

          // find the first and last blank within the overlap region on cached contig (if any)
          for (int i = 0; i < overlap_len; i++) {
            if (ctg_seq[get_start] == ' ') {
              found = false;
              break;
            }
            get_start++;
            get_len--;
          }
          if (!found) {
            while (get_len > 0) {
              if (ctg_seq[get_start + get_len - 1] == ' ') break;
              get_len--;
            }
          }
        } else {
          auto itpair = ctg_cache.insert({ctg_loc.cid, string(ctg_loc.clen, ' ')});
          if (itpair.second) {
            // successful insert
            it = itpair.first;
            assert(it != ctg_cache.end());
            string &ctg_str = it->second;  // the actual underlying string, not view
            ctg_seq = string_view(ctg_str.data(), ctg_str.size());
          } else {
            // use the temporary buffer...
            assert(found == false);
            WARN("FIXME using temporary buffer for invalid cid: ", ctg_loc.cid, "\n");
            tmp_ctg = string(ctg_loc.clen, ' ');
            ctg_seq = string_view(tmp_ctg.data(), tmp_ctg.size());
          }
        }
        if (!found) {
          if (ctg_seq.size() < cstart + overlap_len || ctg_seq.size() != ctg_loc.clen)
            WARN("ctg_seq size mismatch size=", ctg_seq.size(), " cstart=", cstart, " overlap_len=", overlap_len,
                 " ctg_loc=.cid=", ctg_loc.cid, " .clen=", ctg_loc.clen, " .pos=", ctg_loc.pos_in_ctg, "\n");
          assert(ctg_seq.size() >= cstart + overlap_len);
          assert(ctg_seq.size() == ctg_loc.clen);
          // also get extra bordering blank bases on either side of the contig for negligable extra overhead and likely fewer rgets
          const int extra_bases = 128;
          for (int i = 0; i < extra_bases; i++) {
            if (get_start == 0) break;
            if (ctg_seq[get_start - 1] == ' ') {
              get_start--;
              get_len++;
            } else {
              break;
            }
          }
          for (int i = 0; i < extra_bases; i++) {
            if (get_start + get_len >= ctg_seq.size()) break;
            if (ctg_seq[get_start + get_len] == ' ')
              get_len++;
            else
              break;
          }

          assert(get_start >= 0);
          assert(get_start + get_len <= ctg_seq.size());
          fetch_ctg_seqs_timer.start();
          // write directly to the cached string in active scope (represented by the string view, so okay to const_cast)
          rget(ctg_loc.seq_gptr + get_start, const_cast<char *>(ctg_seq.data()) + get_start, get_len).wait();
          fetch_ctg_seqs_timer.stop();
          ctg_bytes_fetched += get_len;
        } else {
          ctg_cache_hits++;
        }
      }
      // set the subsequence to be the overlap region on the contig
      string_view ctg_subseq = string_view(ctg_seq.data() + cstart, overlap_len);

      assert(pos_in_read + kmer_len <= rseq_ptr.size() && "kmer fits in read");
      assert(ctg_loc.pos_in_ctg + kmer_len <= ctg_seq.size() && "kmer fits in ctg");
      assert(memcmp(rseq_ptr.data() + pos_in_read, ctg_seq.data() + ctg_loc.pos_in_ctg, kmer_len) == 0 &&
             "kmer seed exact matches read and ctg");
      align_read(rname, ctg_loc.cid, read_subseq, ctg_subseq, rstart, rlen, cstart, ctg_loc.clen, orient, overlap_len,
                 read_group_id);
      num_alns++;
    }
  }

  void sort_alns() {
    if (!kernel_alns.empty()) {
      DIE("sort_alns called while alignments are still pending to be processed - ", kernel_alns.size());
    }
    if (!active_kernel_fut.ready()) {
      SWARN("Waiting for active_kernel - has flush_remaining() been called?\n");
    }
    active_kernel_fut.wait();
    alns->sort_alns().wait();
  }

  void log_ctg_bytes_fetched() {
    auto all_ctg_bytes_fetched = reduce_one(ctg_bytes_fetched, op_fast_add, 0).wait();
    auto max_ctg_bytes_fetched = reduce_one(ctg_bytes_fetched, op_fast_max, 0).wait();
    SLOG_VERBOSE("Contig bytes fetched ", get_size_str(all_ctg_bytes_fetched), " balance ",
                 (double)all_ctg_bytes_fetched / (rank_n() * max_ctg_bytes_fetched), "\n");
  }

  void print_cache_stats() {
    auto all_ctg_local_hits = reduce_one(ctg_local_hits, op_fast_add, 0).wait();
    auto all_ctg_cache_hits = reduce_one(ctg_cache_hits, op_fast_add, 0).wait();
    auto all_ctg_lookups = reduce_one(ctg_lookups + ctg_local_hits, op_fast_add, 0).wait();
    SLOG_VERBOSE("Hits on ctg cache: ", perc_str(all_ctg_cache_hits, all_ctg_lookups), " cache size ", ctg_cache.size(), /*" of ",
                  ctg_cache.capacity(), " clobberings ", ctg_cache.get_clobberings(),*/
                 "\n");
    SLOG_VERBOSE("Local contig hits bypassing cache: ", perc_str(all_ctg_local_hits, all_ctg_lookups), "\n");
  }
};

template <int MAX_K>
static void build_alignment_index(KmerCtgDHT<MAX_K> &kmer_ctg_dht, Contigs &ctgs, unsigned min_ctg_len) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_kmers = 0;
  ProgressBar progbar(ctgs.size(), "Extracting seeds from contigs");
  // estimate and reserve room in the local map to avoid excessive reallocations
  int64_t est_num_kmers = 0;
  for (auto it = ctgs.begin(); it != ctgs.end(); ++it) {
    auto ctg = it;
    auto len = ctg->seq.length();
    if (len < min_ctg_len) continue;
    est_num_kmers += len - kmer_ctg_dht.kmer_len + 1;
  }
  est_num_kmers = upcxx::reduce_all(est_num_kmers, upcxx::op_fast_add).wait();
  auto my_reserve = 1.2 * est_num_kmers / rank_n() + 2000;  // 120% to keep the map fast
  kmer_ctg_dht.reserve(my_reserve);
  kmer_ctg_dht.reserve_ctg_seqs(ctgs.size());
  vector<Kmer<MAX_K>> kmers;
  for (auto it = ctgs.begin(); it != ctgs.end(); ++it) {
    auto ctg = it;
    progbar.update();
    if (ctg->seq.length() < min_ctg_len) continue;
    global_ptr<char> seq_gptr = kmer_ctg_dht.add_ctg_seq(ctg->seq);
    CtgLoc ctg_loc = {.cid = ctg->id, .seq_gptr = seq_gptr, .clen = (int)ctg->seq.length(), .depth = (float)ctg->depth};
    Kmer<MAX_K>::get_kmers(kmer_ctg_dht.kmer_len, string_view(ctg->seq.data(), ctg->seq.size()), kmers, true);
    num_kmers += kmers.size();
    for (unsigned i = 0; i < kmers.size(); i++) {
      ctg_loc.pos_in_ctg = i;
      if (!kmers[i].is_valid()) continue;
      kmer_ctg_dht.add_kmer(kmers[i], ctg_loc);
    }
    progress();
  }
  auto fut = progbar.set_done();
  kmer_ctg_dht.flush_add_kmers();
  auto tot_num_kmers = reduce_one(num_kmers, op_fast_add, 0).wait();
  fut.wait();
  auto num_kmers_in_ht = kmer_ctg_dht.get_num_kmers();
  LOG("Estimated room for ", my_reserve, " my final count ", kmer_ctg_dht.size(), "\n");
  SLOG_VERBOSE("Processed ", tot_num_kmers, " seeds from contigs, added ", num_kmers_in_ht, "\n");
  auto num_dropped_seed_to_ctgs = kmer_ctg_dht.get_num_dropped_seed_to_ctgs();
  if (num_dropped_seed_to_ctgs)
    SLOG_VERBOSE("Dropped ", num_dropped_seed_to_ctgs, " non-unique seed-to-contig mappings (", setprecision(2), fixed,
                 (100.0 * num_dropped_seed_to_ctgs / tot_num_kmers), "%)\n");
}

template <int MAX_K>
static int align_kmers(KmerCtgDHT<MAX_K> &kmer_ctg_dht, Aligner &aligner,
                       HASH_TABLE<Kmer<MAX_K>, vector<KmerToRead>> &kmer_read_map, vector<ReadRecord *> &read_records,
                       int64_t &num_excess_alns_reads, int &read_group_id, int64_t &kmer_bytes_sent, int64_t &kmer_bytes_received) {
  // get the contigs that match one read
  // extract a list of kmers for each target rank
  auto kmer_lists = new vector<Kmer<MAX_K>>[rank_n()];
  for (auto &elem : kmer_read_map) {
    auto &kmer_fw = elem.first;
    Kmer<MAX_K> kmer_rc = kmer_fw.revcomp();
    const Kmer<MAX_K> *kmer_lc = (kmer_rc < kmer_fw) ? &kmer_rc : &kmer_fw;
    kmer_lists[kmer_ctg_dht.get_target_rank(*kmer_lc)].push_back(*kmer_lc);
  }
  get_ctgs_timer.start();
  future<> fut_serial_results = make_future();
  // fetch ctgs for each set of kmers from target ranks
  auto lranks = local_team().rank_n();
  auto nnodes = rank_n() / lranks;
  for (auto target_rank : upcxx_utils::foreach_rank_by_node()) {  // stagger by rank_me, round robin by node
    progress();
    // skip targets that have no ctgs - this should reduce communication at scale
    if (kmer_lists[target_rank].empty()) continue;
    kmer_bytes_sent += kmer_lists[target_rank].size() * sizeof(Kmer<MAX_K>);
    auto fut_get_ctgs = kmer_ctg_dht.get_ctgs_with_kmers(target_rank, kmer_lists[target_rank]);
    kmer_lists[target_rank].clear();
    auto fut_rpc_returned = fut_get_ctgs.then([target_rank, &kmer_read_map, &num_excess_alns_reads,
                                               &kmer_bytes_received](const vector<KmerAndCtgLoc<MAX_K>> kmer_ctg_locs) {
      // iterate through the kmers, each one has an associated ctg location
      for (auto &kmer_ctg_loc : kmer_ctg_locs) {
        kmer_bytes_received += sizeof(kmer_ctg_loc.ctg_loc) + sizeof(Kmer<MAX_K>);
        // get the reads that this kmer mapped to
        assert(kmer_ctg_loc.kmer.is_valid());
        assert(kmer_ctg_loc.kmer.is_least());
        auto kmer_read_map_it = kmer_read_map.find(kmer_ctg_loc.kmer);
        if (kmer_read_map_it == kmer_read_map.end()) DIE("Could not find kmer ", kmer_ctg_loc.kmer);
        // this is a list of the reads
        auto &kmer_to_reads = kmer_read_map_it->second;
        // now add the ctg loc to all the reads
        for (auto &kmer_to_read : kmer_to_reads) {
          auto read_record = kmer_to_read.read_record;
          assert(read_record->is_valid());
          int pos_in_read = kmer_to_read.pos_in_read;
          bool read_is_rc = kmer_to_read.is_rc;
          if (KLIGN_MAX_ALNS_PER_READ && read_record->aligned_ctgs_map.size() >= KLIGN_MAX_ALNS_PER_READ) {
            // too many mappings for this read, stop adding to it
            num_excess_alns_reads++;
            continue;
          }
          // this here ensures that we don't insert duplicate mappings
          read_record->aligned_ctgs_map.insert({kmer_ctg_loc.ctg_loc.cid, {pos_in_read, read_is_rc, kmer_ctg_loc.ctg_loc}});
        }
      }
    });

    upcxx_utils::limit_outstanding_futures(fut_rpc_returned, std::max(nnodes * 2, lranks * 4)).wait();
  }

  upcxx_utils::flush_outstanding_futures();

  get_ctgs_timer.stop();
  delete[] kmer_lists;
  kmer_read_map.clear();

  compute_alns_timer.start();
  int num_reads_aligned = 0;
  // compute alignments for each read
  for (auto read_record : read_records) {
    assert(read_record->is_valid());
    progress();
    // compute alignments
    if (read_record->aligned_ctgs_map.size()) {
      num_reads_aligned++;
      aligner.compute_alns_for_read(&read_record->aligned_ctgs_map, read_record->id, read_record->seq, read_group_id);
    }
    delete read_record;
  }
  read_records.clear();
  compute_alns_timer.stop();
  return num_reads_aligned;
}

template <int MAX_K>
static double do_alignments(KmerCtgDHT<MAX_K> &kmer_ctg_dht, vector<PackedReads *> &packed_reads_list, Alns &alns, int rlen_limit,
                            int seed_space, int64_t all_num_ctgs, bool compute_cigar) {
  BarrierTimer timer(__FILEFUNC__);
  SLOG_VERBOSE("Using a seed space of ", seed_space, "\n");
  int64_t tot_num_kmers = 0;
  int64_t num_reads = 0;
  int64_t num_reads_aligned = 0, num_excess_alns_reads = 0;
  Aligner aligner(Kmer<MAX_K>::get_k(), alns, rlen_limit, compute_cigar, all_num_ctgs);
  aligner.clear_aln_bufs();
  barrier();
  int64_t kmer_bytes_received = 0;
  int64_t kmer_bytes_sent = 0;
  upcxx::future<> all_done = make_future();
  int read_group_id = 0;
  HASH_TABLE<Kmer<MAX_K>, vector<KmerToRead>> kmer_read_map;
  kmer_read_map.reserve(KLIGN_CTG_FETCH_BUF_SIZE);
  for (auto packed_reads : packed_reads_list) {
    packed_reads->reset();
    string read_id, read_seq, quals;
    ProgressBar progbar(packed_reads->get_local_num_reads(), "Aligning reads to contigs");
    vector<ReadRecord *> read_records;
    vector<Kmer<MAX_K>> kmers;
    while (true) {
      progress();
      if (!packed_reads->get_next_read(read_id, read_seq, quals)) break;
      progbar.update();
      // this happens when a placeholder read with just a single N character is added after merging reads
      if (kmer_ctg_dht.kmer_len > read_seq.length()) continue;
      Kmer<MAX_K>::get_kmers(kmer_ctg_dht.kmer_len, string_view(read_seq.data(), read_seq.size()), kmers, true);
      tot_num_kmers += kmers.size();
      ReadRecord *read_record = new ReadRecord(read_id, read_seq);
      read_records.push_back(read_record);
      bool filled = false;
      for (int i = 0; i < kmers.size(); i += seed_space) {
        const Kmer<MAX_K> &kmer_fw = kmers[i];
        if (!kmer_fw.is_valid()) continue;
        const Kmer<MAX_K> kmer_rc = kmer_fw.revcomp();
        const Kmer<MAX_K> *kmer_lc = &kmer_fw;
        assert(kmer_fw.is_valid() && kmer_rc.is_valid());
        bool is_rc = false;
        if (kmer_rc < kmer_fw) {
          kmer_lc = &kmer_rc;
          is_rc = true;
        }
        assert(kmer_lc->is_least());
        auto it = kmer_read_map.find(*kmer_lc);
        if (it == kmer_read_map.end()) {
          auto pairit = kmer_read_map.insert({*kmer_lc, {}});
          it = pairit.first;
          assert(pairit.second);
          kmer_ctg_dht.unique_kmer_seed_lookups++;
        }
        kmer_ctg_dht.kmer_seed_lookups++;
        it->second.push_back({read_record, i, is_rc});
      }
      if (kmer_read_map.size() + kmers.size() * 2 >= KLIGN_CTG_FETCH_BUF_SIZE) filled = true;
      if (filled) {
        num_reads_aligned += align_kmers(kmer_ctg_dht, aligner, kmer_read_map, read_records, num_excess_alns_reads, read_group_id,
                                         kmer_bytes_sent, kmer_bytes_received);
        assert(read_records.empty());
        assert(kmer_read_map.empty());
      }
      num_reads++;
    }
    if (read_records.size()) {
      num_reads_aligned += align_kmers(kmer_ctg_dht, aligner, kmer_read_map, read_records, num_excess_alns_reads, read_group_id,
                                       kmer_bytes_sent, kmer_bytes_received);
    }
    assert(read_records.empty());
    assert(kmer_read_map.empty());
    aligner.flush_remaining(read_group_id);
    read_group_id++;
    all_done = when_all(all_done, progbar.set_done());
  }
  // free some memory
  HASH_TABLE<Kmer<MAX_K>, vector<KmerToRead>>().swap(kmer_read_map);

  aligner.print_cache_stats();
  // make sure to do any outstanding kernel block alignments
  auto tot_num_reads_fut = reduce_one(num_reads, op_fast_add, 0);
  auto num_excess_alns_reads_fut = reduce_one(num_excess_alns_reads, op_fast_add, 0);
  auto num_seeds_fut = reduce_one(tot_num_kmers, op_fast_add, 0);
  auto tot_num_reads_aligned_fut = reduce_one(num_reads_aligned, op_fast_add, 0);

  aligner.sort_alns();
  auto num_overlaps = aligner.get_num_overlaps();
  all_done.wait();
  barrier();

  auto tot_num_reads = tot_num_reads_fut.wait();
  SLOG_VERBOSE("Parsed ", tot_num_reads, " reads, with ", num_seeds_fut.wait(), " seeds\n");
  auto tot_num_alns = aligner.get_num_alns();
  SLOG_VERBOSE("Found ", tot_num_alns, " alignments of which ", perc_str(aligner.get_num_perfect_alns(), tot_num_alns),
               " are perfect\n");
  auto tot_excess_alns_reads = num_excess_alns_reads_fut.wait();
  if (num_excess_alns_reads)
    SLOG_VERBOSE("Dropped ", tot_excess_alns_reads, " reads because of alignments in excess of ", KLIGN_MAX_ALNS_PER_READ, "\n");
  if (num_overlaps) SLOG_VERBOSE("Dropped ", perc_str(num_overlaps, tot_num_alns), " alignments becasue of overlaps\n");
  auto tot_num_reads_aligned = tot_num_reads_aligned_fut.wait();
  SLOG_VERBOSE("Mapped ", perc_str(tot_num_reads_aligned, tot_num_reads), " reads to contigs\n");
  SLOG_VERBOSE("Average mappings per read ", (double)tot_num_alns / tot_num_reads_aligned, "\n");
  auto all_kmer_bytes_sent = reduce_one(kmer_bytes_sent, op_fast_add, 0).wait();
  auto all_kmer_bytes_received = reduce_one(kmer_bytes_received, op_fast_add, 0).wait();

  SLOG_VERBOSE("Sent ", get_size_str(all_kmer_bytes_sent), " and received ", get_size_str(all_kmer_bytes_received), " of kmers\n");

  aligner.log_ctg_bytes_fetched();

  fetch_ctg_seqs_timer.done_all();
  aln_cpu_bypass_timer.done_all();
  get_ctgs_timer.done_all();
  compute_alns_timer.done_all();
  aln_kernel_timer.done_all();
  double aln_kernel_secs = aln_kernel_timer.get_elapsed();
  fetch_ctg_seqs_timer.clear();
  aln_cpu_bypass_timer.clear();
  get_ctgs_timer.clear();
  compute_alns_timer.clear();
  aln_kernel_timer.clear();
  return aln_kernel_secs;
}

template <int MAX_K>
double find_alignments(unsigned kmer_len, vector<PackedReads *> &packed_reads_list, int max_store_size, int max_rpcs_in_flight,
                       Contigs &ctgs, Alns &alns, int seed_space, int rlen_limit, bool use_kmer_cache, bool compute_cigar,
                       int min_ctg_len) {
  BarrierTimer timer(__FILEFUNC__);
  Kmer<MAX_K>::set_k(kmer_len);
  SLOG_VERBOSE("Aligning with seed size of ", kmer_len, "\n");
  auto all_num_ctgs = reduce_all(ctgs.size(), op_fast_add).wait();
  KmerCtgDHT<MAX_K> kmer_ctg_dht(max_store_size, max_rpcs_in_flight);
  barrier();
  build_alignment_index(kmer_ctg_dht, ctgs, min_ctg_len);
#ifdef DEBUG
// kmer_ctg_dht.dump_ctg_kmers();
#endif
  double kernel_elapsed = do_alignments(kmer_ctg_dht, packed_reads_list, alns, rlen_limit, seed_space, all_num_ctgs, compute_cigar);
  barrier();
  auto num_alns = alns.size();
  auto num_dups = alns.get_num_dups();
  if (num_dups) SLOG_VERBOSE("Number of duplicate alignments ", perc_str(num_dups, num_alns), "\n");
  barrier();
  return kernel_elapsed;
}
