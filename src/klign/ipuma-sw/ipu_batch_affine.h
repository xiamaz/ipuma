#ifndef IPU_BATCH_AFFINE_HPP
#define IPU_BATCH_AFFINE_HPP

#include "ipu_base.h"
#include<vector>

using namespace poplar;

namespace ipu {
namespace batchaffine {

  namespace partition {
    enum class Algorithm {fillFirst, roundRobin, greedy};
  }

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

enum class VertexType { cpp, assembly, multi, multiasm, stripedasm };

static const std::string typeString[] = {"SWAffine", "SWAffineAsm", "MultiSWAffine", "MultiSWAffineAsm", "StripedSWAffineAsm"};
std::string vertexTypeToString(VertexType v);

struct IPUAlgoConfig {
  int tilesUsed = 1; // number of active vertices
  int maxAB = 300; // maximum length of a single comparison
  int maxBatches = 20; // maximum number of comparisons in a single batch
  int bufsize = 3000; // total size of buffer for A and B individually
  VertexType vtype = VertexType::cpp;
  partition::Algorithm fillAlgo = partition::Algorithm::fillFirst;

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
};

struct BlockAlignmentResults {
  std::vector<int32_t> scores;
  std::vector<int32_t> mismatches;
  std::vector<int32_t> a_range_result;
  std::vector<int32_t> b_range_result;
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

  IPUAlgoConfig algoconfig;
 public:
  SWAlgorithm(SWConfig config, IPUAlgoConfig algoconfig);

  static std::vector<std::tuple<int, int>> fillBuckets(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B);
  static void checkSequenceSizes(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B);

  BlockAlignmentResults get_result();

  // Local Buffers
  void compare_local(const std::vector<std::string>& A, const std::vector<std::string>& B);

  // Remote bufffer
  void prepared_remote_compare(char* a,  int32_t* a_len,  char* b,  int32_t* b_len, int32_t * scores, int32_t *mismatches, int32_t * a_range_result, int32_t * b_range_result);
  static void prepare_remote(IPUAlgoConfig& algoconfig, const std::vector<std::string>& A, const std::vector<std::string>& B,  char* a,  int32_t* a_len,  char* b,  int32_t* b_len, std::vector<int>& deviceMapping);
};
}  // namespace batchaffine
}  // namespace ipu
#endif  // IPU_BATCH_AFFINE_HPP