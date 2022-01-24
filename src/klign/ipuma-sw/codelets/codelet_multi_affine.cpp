#include <poplar/Vertex.hpp>
#include <poplar/FieldTypes.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/StackSizeDefs.hpp>
#include <poplar/AvailableVTypes.h>
#include "poplar/TileConstants.hpp"
#include <print.h>
#include <type_traits>

inline int max(int a, int b) {
    return a > b ? a : b;
}

/**
 * Single SW operations for cell i, j
 * based on Fig. 1, Wozniak et al 1997
 */
class MultiSWAffine : public poplar::MultiVertex {
private:
    poplar::Vector<int, poplar::VectorLayout::ONE_PTR> C;
    poplar::Vector<int, poplar::VectorLayout::ONE_PTR> bG;
public:
    // Fields
    poplar::Vector<poplar::Input<poplar::Vector<int, poplar::VectorLayout::ONE_PTR>>> simMatrix;
    poplar::Input<size_t> maxNPerTile;
    poplar::Input<int> gapInit;
    poplar::Input<int> gapExt;
    poplar::Input<int> bufSize;
    poplar::Input<int> maxAB;
    poplar::Input<poplar::Vector<unsigned char, poplar::VectorLayout::ONE_PTR>> A;
    poplar::Input<poplar::Vector<unsigned char, poplar::VectorLayout::ONE_PTR>> B;
    poplar::Input<poplar::Vector<int, poplar::VectorLayout::ONE_PTR>> Alen;
    poplar::Input<poplar::Vector<int, poplar::VectorLayout::ONE_PTR>> Blen;
    poplar::Output<poplar::Vector<int, poplar::VectorLayout::ONE_PTR>> score;
    poplar::Output<poplar::Vector<int, poplar::VectorLayout::ONE_PTR>> mismatches;
    poplar::Output<poplar::Vector<int, poplar::VectorLayout::ONE_PTR>> ARange;
    poplar::Output<poplar::Vector<int, poplar::VectorLayout::ONE_PTR>> BRange;

    bool compute(unsigned workerId) {
        printf("Weird %d\n", workerId);
        for (int i = workerId; i < maxNPerTile; ++i) {
            printf("Yo %d %d %d\n", workerId, i, score[i]);
            score[i] += 1;
        }
        return true;
    }
};