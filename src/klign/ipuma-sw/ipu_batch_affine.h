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

const std::string IPU_AFFINE_CPP = "SWAffine";
const std::string IPU_AFFINE_ASM = "SWAffineAsm";

enum class VertexType {cpp, assembly};

static const std::string typeString[] = {"SWAffine", "SWAffineAsm"};

struct IPUAlgoConfig {
  int tilesUsed = 1; // number of active vertices
  int maxAB = 300; // maximum length of a single comparison
  int maxBatches = 20; // maximum number of comparisons in a single batch
  int bufsize = 3000; // total size of buffer for A and B individually
  VertexType vtype = VertexType::cpp;

  /**
   * @brief This calculates the total number of comparisons that can be computed in a single engine run on the IPU.
   * 
   * @return int 
   */
  int getTotalNumberOfComparisons();

  /**
   * @brief This calculated the required buffer size for input strings across all vertices.
   * 
   * @return int 
   */
  int getTotalBufferSize();

  std::string getVertexTypeString();
};

struct BlockAlignmentResults {
  std::vector<int32_t> scores;
  std::vector<int32_t> mismatches;
  std::vector<int32_t> a_range_result;
  std::vector<int32_t> b_range_result;
};

class SWAlgorithm : public IPUAlgorithm {
 private:
  std::vector<size_t> bucket_pairs;

  std::vector<char> a;
  std::vector<int32_t> a_len;
  std::vector<char> b;
  std::vector<int32_t> b_len;

  std::vector<int32_t> scores;
  std::vector<int32_t> mismatches;
  std::vector<int32_t> a_range_result;
  std::vector<int32_t> b_range_result;

  IPUAlgoConfig algoconfig;

  void fillBuckets(const std::vector<std::string>& A, const std::vector<std::string>& B);

 public:
  SWAlgorithm(SWConfig config, IPUAlgoConfig algoconfig);

  BlockAlignmentResults get_result();

  void compare(const std::vector<std::string>& A, const std::vector<std::string>& B);
};
}  // namespace batchaffine
}  // namespace ipu
#endif  // IPU_BATCH_AFFINE_HPP