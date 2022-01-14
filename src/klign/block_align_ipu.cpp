#include "klign.hpp"
#include "kmer.hpp"
#include "aligner_cpu.hpp"

#include "ipuma-sw/ipulib.hpp"
#include "ipuma-sw/ipu_batch_affine.hpp"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

void kernel_align_block(CPUAligner &cpu_aligner, vector<Aln> &kernel_alns, vector<string> &ctg_seqs, vector<string> &read_seqs,
                        Alns *alns, future<> &active_kernel_fut, int read_group_id, int max_clen, int max_rlen,
                        IntermittentTimer &aln_kernel_timer) {
  if (!kernel_alns.empty()) {
    active_kernel_fut.wait();  // should be ready already
    shared_ptr<AlignBlockData> aln_block_data =
        make_shared<AlignBlockData>(kernel_alns, ctg_seqs, read_seqs, max_clen, max_rlen, read_group_id, cpu_aligner.aln_scoring);
    assert(kernel_alns.empty());
    active_kernel_fut = cpu_aligner.ssw_align_block(aln_block_data, alns);
  }
}

void init_aligner(AlnScoring &aln_scoring, int rlen_limit) {
}

void cleanup_aligner() {
}