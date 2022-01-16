#ifndef IPU_BATCH_AFFINE_HPP
#define IPU_BATCH_AFFINE_HPP

#include "ipu_base.h"

using namespace poplar;

namespace ipu {
namespace batchaffine {

const std::string STREAM_A = "a-write";
const std::string STREAM_A_LEN = "a-len-write";
const std::string STREAM_B = "b-write";
const std::string STREAM_B_LEN = "b-len-write";
const std::string STREAM_SCORES = "scores-read";
const std::string STREAM_MISMATCHES = "mismatches-read";
const std::string STREAM_A_RANGE = "a-range-read";
const std::string STREAM_B_RANGE = "b-range-read";


struct BlockAlignmentResults {
  std::vector<int32_t> &scores;
  std::vector<int32_t> &mismatches;
  std::vector<int32_t> &a_range_result;
  std::vector<int32_t> &b_range_result;
};

class SWAlgorithm : public IPUAlgorithm {
 private:
  std::vector<char> a;
  std::vector<int32_t> a_len;
  std::vector<char> b;
  std::vector<int32_t> b_len;

  std::vector<int32_t> scores;
  std::vector<int32_t> mismatches;
  std::vector<int32_t> a_range_result;
  std::vector<int32_t> b_range_result;
  int maxAB;

 public:
  SWAlgorithm(SWConfig config, int activeTiles = 1472, int maxAB = 300, int maxBatches = 100, int bufsize = 30000);

  BlockAlignmentResults get_result();

  void compare(const std::vector<std::string>& A, const std::vector<std::string>& B);
};
}  // namespace batchaffine
}  // namespace ipu
#endif  // IPU_BATCH_AFFINE_HPP