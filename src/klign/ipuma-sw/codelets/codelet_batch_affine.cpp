#include <poplar/Vertex.hpp>
#include <poplar/FieldTypes.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/StackSizeDefs.hpp>
#include <poplar/AvailableVTypes.h>
#include "poplar/TileConstants.hpp"
#include <print.h>
#include <type_traits>

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

template<class T>T max(T a, T b) {
    return a > b ? a : b;
}

/**
 * Single SW operations for cell i, j
 * based on Fig. 1, Wozniak et al 1997
 */
class SWAffine : public poplar::Vertex {
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

    bool compute() {

        int gI = *gapInit;
        int gE = *gapExt;

        int i_offset = 0;
        int j_offset = 0;
        
        for (int n = 0; n < maxNPerTile; ++n) {
            int lastNoGap, prevNoGap;
            int s = 0;
            uint16_t Astart = 0;
            uint16_t Bstart = 0;
            uint16_t Aend = 0;
            uint16_t Bend = 0;

            auto a_len = Alen[n];
            auto b_len = Blen[n];
            if (a_len == 0 || b_len == 0) break;

            memset(&(C[0]), 0, maxAB * sizeof(int));
            memset(&(bG[0]), 0, maxAB * sizeof(int));

            // forward pass
            for (int i = 0; i < b_len; ++i) {
                int aGap;
                lastNoGap = prevNoGap = 0;
                aGap = gapInit;
                for (unsigned j = 0; j < a_len; ++j) {
                    aGap = max(lastNoGap + gI + gE, aGap + gE);
                    bG[j] = max(C[j] + gI + gE, bG[j] + gE);

                    lastNoGap = max(prevNoGap + simMatrix[A[j_offset + j]][B[i_offset + i]], aGap);
                    lastNoGap = max(lastNoGap, bG[j]);
                    lastNoGap = max(lastNoGap, 0);
                    prevNoGap = C[j];
                    C[j] = lastNoGap;
                    if (lastNoGap > s) {
                        Aend = j;
                        Bend = i;
                        s = lastNoGap;
                    }
                }
            }
            score[n] = s;

            s = 0;

            memset(&(C[0]), 0, maxAB * sizeof(int));
            memset(&(bG[0]), 0, maxAB * sizeof(int));
            // reverse pass
            for (int i = Bend; i >= 0; --i) {
                int aGap;
                lastNoGap = prevNoGap = 0;
                aGap = gapInit;
                for (int j = Aend; j >= 0; --j) {
                    aGap = max(lastNoGap + gI + gE, aGap + gE);
                    bG[j] = max(C[j] + gI + gE, bG[j] + gE);
                    lastNoGap = max(prevNoGap + simMatrix[A[j_offset + j]][B[i_offset + i]], aGap);
                    lastNoGap = max(lastNoGap, bG[j]);
                    lastNoGap = max(lastNoGap, 0);
                    prevNoGap = C[j];
                    C[j] = lastNoGap;
                    if (lastNoGap > s) {
                        Astart = j;
                        Bstart = i;
                        s = lastNoGap;
                    }
                }
            }

            uint16_t range[2];
            range[0] = Astart;
            range[1] = Aend;
            ARange[n] = *reinterpret_cast<uint32_t*>(range);
            range[0] = Bstart;
            range[1] = Bend;
            BRange[n] = *reinterpret_cast<uint32_t*>(range);

            i_offset += b_len;
            j_offset += a_len;
        }
        return true;
    }
};