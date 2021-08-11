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

#include "aligner_cpu.hpp"

#ifdef __PPC64__  // FIXME remove after solving Issues #60 #35 #49
#define NO_KLIGN_CPU_WORK_STEAL
#endif

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

AlignBlockData::AlignBlockData(vector<Aln> &_kernel_alns, vector<string> &_ctg_seqs, vector<string> &_read_seqs, int64_t max_clen,
                               int64_t max_rlen, int read_group_id, AlnScoring &aln_scoring)
    : max_clen(max_clen)
    , max_rlen(max_rlen)
    , read_group_id(read_group_id)
    , aln_scoring(aln_scoring) {
  // copy/swap/reserve necessary data and configs
  kernel_alns.swap(_kernel_alns);
  kernel_alns.reserve(_kernel_alns.size());
  ctg_seqs.swap(_ctg_seqs);
  ctg_seqs.reserve(_ctg_seqs.size());
  read_seqs.swap(_read_seqs);
  read_seqs.reserve(_read_seqs.size());
  alns = make_shared<Alns>();
  alns->reserve(_kernel_alns.size());
}

int get_cigar_length(const string &cigar) {
  // check that cigar string length is the same as the sequence, but only if the sequence is included
  int base_count = 0;
  string num = "";
  for (char c : cigar) {
    switch (c) {
      case 'M':
      case 'S':
      case '=':
      case 'X':
      case 'I':
        base_count += stoi(num);
        num = "";
        break;
      case 'D':
        // base_count -= stoi(num);
        num = "";
        break;
      default:
        if (!isdigit(c)) DIE("Invalid char detected in cigar: '", c, "'");
        num += c;
        break;
    }
  }
  return base_count;
}

void set_sam_string(Aln &aln, string_view read_seq, string cigar) {
  assert(aln.is_valid());
  aln.sam_string = aln.read_id + "\t";
  string tmp;
  if (aln.orient == '-') {
    aln.sam_string += "16\t";
    if (read_seq != "*") {
      tmp = revcomp(string(read_seq.data(), read_seq.size()));
      read_seq = string_view(tmp.data(), tmp.size());
    }
    // reverse(read_quals.begin(), read_quals.end());
  } else {
    aln.sam_string += "0\t";
  }
  aln.sam_string += "Contig" + to_string(aln.cid) + "\t" + to_string(aln.cstart + 1) + "\t";
  uint32_t mapq;
  // for perfect match, set to same maximum as used by minimap or bwa
  if (aln.score2 == 0) {
    mapq = 60;
  } else {
    mapq = -4.343 * log(1 - (double)abs(aln.score1 - aln.score2) / (double)aln.score1);
    mapq = (uint32_t)(mapq + 4.99);
    mapq = mapq < 254 ? mapq : 254;
  }
  aln.sam_string += to_string(mapq) + "\t";
  // aln.sam_string += cigar + "\t*\t0\t0\t" + read_subseq + "\t*\t";
  // Don't output either the read sequence or quals - that causes the SAM file to bloat up hugely, and that info is already
  // available in the read files
  aln.sam_string += cigar + "\t*\t0\t0\t*\t*\t";
  aln.sam_string +=
      "AS:i:" + to_string(aln.score1) + "\tNM:i:" + to_string(aln.mismatches) + "\tRG:Z:" + to_string(aln.read_group_id);
  // for debugging
  // aln.sam_string += " rstart " + to_string(aln.rstart) + " rstop " + to_string(aln.rstop) + " cstop " + to_string(aln.cstop) +
  //                  " clen " + to_string(aln.clen) + " alnlen " + to_string(aln.rstop - aln.rstart);
  /*
#ifdef DEBUG
  // only used if we actually include the read seq and quals in the SAM, which we don't
  int base_count = get_cigar_length(cigar);
  if (base_count != read_seq.length())
    DIE("number of bases in cigar != aln rlen, ", base_count, " != ", read_subseq.length(), "\nsam string ", aln.sam_string);
#endif
  */
}

CPUAligner::CPUAligner(bool compute_cigar)
    : ssw_aligner() {
  // default for normal alignments in the pipeline, but for final alignments, uses minimap2 defaults
  if (!compute_cigar)
    aln_scoring = {.match = ALN_MATCH_SCORE,
                   .mismatch = ALN_MISMATCH_COST,
                   .gap_opening = ALN_GAP_OPENING_COST,
                   .gap_extending = ALN_GAP_EXTENDING_COST,
                   .ambiguity = ALN_AMBIGUITY_COST};
  else
    aln_scoring = {.match = 2, .mismatch = 4, .gap_opening = 4, .gap_extending = 2, .ambiguity = 1};
  SLOG_VERBOSE("Alignment scoring parameters: ", aln_scoring.to_string(), "\n");

  // aligner construction: SSW internal defaults are 2 2 3 1
  ssw_aligner.Clear();
  ssw_aligner.ReBuild(aln_scoring.match, aln_scoring.mismatch, aln_scoring.gap_opening, aln_scoring.gap_extending,
                      aln_scoring.ambiguity);
  ssw_filter.report_cigar = compute_cigar;
}

void CPUAligner::ssw_align_read(StripedSmithWaterman::Aligner &ssw_aligner, StripedSmithWaterman::Filter &ssw_filter, Alns *alns,
                                AlnScoring &aln_scoring, Aln &aln, const string_view &cseq, const string_view &rseq,
                                int read_group_id) {
  assert(aln.clen >= cseq.length() && "contig seq is contained within the greater contig");
  assert(aln.rlen >= rseq.length() && "read seq is contained with the greater read");

  StripedSmithWaterman::Alignment ssw_aln;
  // align query, ref, reflen
  ssw_aligner.Align(cseq.data(), cseq.length(), rseq.data(), rseq.length(), ssw_filter, &ssw_aln,
                    max((int)(rseq.length() / 2), 15));

  aln.rstop = aln.rstart + ssw_aln.ref_end + 1;
  aln.rstart += ssw_aln.ref_begin;
  aln.cstop = aln.cstart + ssw_aln.query_end + 1;
  aln.cstart += ssw_aln.query_begin;
  if (aln.orient == '-') switch_orient(aln.rstart, aln.rstop, aln.rlen);

  aln.score1 = ssw_aln.sw_score;
  aln.score2 = ssw_aln.sw_score_next_best;
  aln.mismatches = ssw_aln.mismatches;
  aln.identity = (unsigned)100 * (unsigned)ssw_aln.sw_score / (unsigned)aln_scoring.match / (unsigned)aln.rlen;
  aln.read_group_id = read_group_id;
  if (ssw_filter.report_cigar) set_sam_string(aln, rseq, ssw_aln.cigar_string);
  alns->add_aln(aln);
}

void CPUAligner::ssw_align_read(Alns *alns, Aln &aln, const string &cseq, const string &rseq, int read_group_id) {
  ssw_align_read(ssw_aligner, ssw_filter, alns, aln_scoring, aln, cseq, rseq, read_group_id);
}

upcxx::future<> CPUAligner::ssw_align_block(shared_ptr<AlignBlockData> aln_block_data, Alns *alns) {
  AsyncTimer t("ssw_align_block (thread)");
  future<> fut = upcxx_utils::execute_in_thread_pool(
      [&ssw_aligner = this->ssw_aligner, &ssw_filter = this->ssw_filter, &aln_scoring = this->aln_scoring, aln_block_data, t]() {
        t.start();
        assert(!aln_block_data->kernel_alns.empty());
        DBG_VERBOSE("Starting _ssw_align_block of ", aln_block_data->kernel_alns.size(), "\n");
        auto alns_ptr = aln_block_data->alns.get();
        for (int i = 0; i < aln_block_data->kernel_alns.size(); i++) {
          Aln &aln = aln_block_data->kernel_alns[i];
          string &cseq = aln_block_data->ctg_seqs[i];
          string &rseq = aln_block_data->read_seqs[i];
          ssw_align_read(ssw_aligner, ssw_filter, alns_ptr, aln_scoring, aln, cseq, rseq, aln_block_data->read_group_id);
        }
        t.stop();
      });
  fut = fut.then([alns = alns, aln_block_data, t]() {
    SLOG_VERBOSE("Finished CPU SSW aligning block of ", aln_block_data->kernel_alns.size(), " in ", t.get_elapsed(), " s (",
                 (t.get_elapsed() > 0 ? aln_block_data->kernel_alns.size() / t.get_elapsed() : 0.0), " aln/s)\n");
    DBG_VERBOSE("appending and returning ", aln_block_data->alns->size(), "\n");
    alns->append(*(aln_block_data->alns));
  });

  return fut;
}
