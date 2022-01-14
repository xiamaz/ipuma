#include <poplar/Vertex.hpp>
#include <poplar/FieldTypes.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/StackSizeDefs.hpp>
#include <poplar/AvailableVTypes.h>
#include <print.h>
#include "ExternalCodelet.hpp"

#include "codelet_shared.hpp"

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
class SWAsm : public poplar::Vertex {
private:
    poplar::Vector<T, poplar::VectorLayout::ONE_PTR, 8, true> C;
    poplar::Vector<T, poplar::VectorLayout::ONE_PTR, 8, true> bG;
public:
    // Fields
    poplar::Input<poplar::Vector<T, poplar::VectorLayout::ONE_PTR>> simMatrix;
    poplar::Input<unsigned int> simWidth;
    poplar::Input<T> gapInit;
    poplar::Input<T> gapExt;
    poplar::Input<int> bufSize;
    poplar::Input<poplar::Vector<unsigned char, poplar::VectorLayout::ONE_PTR>> A;
    poplar::Input<unsigned int> Alen;
    poplar::Input<poplar::Vector<unsigned char, poplar::VectorLayout::ONE_PTR>> B;
    poplar::Input<unsigned int> Blen;
    poplar::Output<int> score;

		IS_EXTERNAL_CODELET((std::is_same<T, float>()));

    bool compute() {
        return true;
    }
};

template class SWAsm<float>;