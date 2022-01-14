#include <poplar/Vertex.hpp>
#include <poplar/FieldTypes.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/StackSizeDefs.hpp>
#include <poplar/AvailableVTypes.h>
#include "poplar/TileConstants.hpp"
#include <print.h>
#include <type_traits>

#include "codelet_shared.hpp"

static constexpr auto COMPACT_DELTAN = poplar::VectorListLayout::COMPACT_DELTAN;

#ifdef __IPU__
#include <arch/gc_tile_defines.h>
#include <ipu_memory_intrinsics>

static __attribute__((always_inline)) unsigned maskForRepeat(unsigned input) {
  return input & CSR_W_REPEAT_COUNT__VALUE__MASK;
}

void inline setZeroPart(void* sa_out, unsigned count) {
    int2 * o = reinterpret_cast<int2*>(sa_out);
    const int2 zzero = {0, 0}; 
    const unsigned loopCount = maskForRepeat(count);
    for (unsigned i = 0; i < loopCount; i++) {
      ipu::store_postinc(&o, zzero, 1);
    }
}
#else

static __attribute__((always_inline)) unsigned maskForRepeat(unsigned input) {
  return input;
}

void inline setZeroPart(void* sa_out, unsigned count) {
  memset(sa_out, 0, count);
}
#endif

template<class T>T max(std::initializer_list<T> l) {
    T v = *l.begin();
    for (auto i = l.begin() + 1; i != l.end(); ++i) {
        if (v < *i) {
            v = *i;
        }
    }
    return v;
}

template<class T>T max(T a, T b) {
    return a > b ? a : b;
}

template<class T>
inline T compareElementsT(char a, char b) {
    if (a == b)
        return 1.0;
    return -1.0;
}

/**
 * Single SW operations for cell i, j
 * based on Fig. 1, Wozniak et al 1997
 */
template<class T>
class SWAffine : public poplar::Vertex {
private:
    poplar::Vector<T, poplar::VectorLayout::ONE_PTR> C;
    poplar::Vector<T, poplar::VectorLayout::ONE_PTR> bG;
public:
    // Fields
    poplar::Vector<poplar::Input<poplar::Vector<T, poplar::VectorLayout::ONE_PTR>>> simMatrix;
    poplar::Input<T> gapInit;
    poplar::Input<T> gapExt;
    poplar::Input<int> bufSize;
    poplar::Input<poplar::Vector<unsigned char, poplar::VectorLayout::ONE_PTR>> A;
    poplar::Input<poplar::Vector<unsigned char, poplar::VectorLayout::ONE_PTR>> B;
    poplar::Output<int> score;

    bool compute() {
        // find size of string A and B
        int Alen = 0, Blen = 0;
        for (; (Alen < bufSize - 1) && A[Alen] != simMatrix.size(); ++Alen) {
        }
        for (; (Blen < bufSize - 1) && B[Blen] != simMatrix.size(); ++Blen) {
        }

        // setZeroPart(&(C[0]), (*bufSize * sizeof(T) + 7) >> 3);
        // setZeroPart(&(bG[0]), (*bufSize * sizeof(T) + 7) >> 3);
        memset(&(C[0]), 0, bufSize * sizeof(T));
        memset(&(bG[0]), 0, bufSize * sizeof(T));

        T gI = *gapInit;
        T gE = *gapExt;
        T s = 0;
        T lastNoGap, prevNoGap;
        for (int i = 0; i < Blen; ++i) {
            T aGap;
            lastNoGap = prevNoGap = 0;
            aGap = gapInit;
            for (unsigned j = 0; j < Alen; ++j) {
                aGap = max(lastNoGap + gI + gE, aGap + gE);
                bG[j] = max(C[j] + gI + gE, bG[j] + gE);
                lastNoGap = max(static_cast<T>(prevNoGap + simMatrix[A[j]][B[i]]), aGap);
                lastNoGap = max(lastNoGap, bG[j]);
                lastNoGap = max(lastNoGap, static_cast<T>(0));
                prevNoGap = C[j];
                C[j] = lastNoGap;
                s = max(s, lastNoGap);
            }
        }
        *score = s;
        return true;
    }
};

template class SWAffine<int>;
template class SWAffine<float>;
template class SWAffine<half>;
template class SWAffine<short>;