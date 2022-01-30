
#ifndef __DRIVER_H__
#define __DRIVER_H__

#include "ipuma-sw/driver.hpp"
#include "ipuma-sw/ipu_base.h"
#include "ipuma-sw/ipu_batch_affine.h"

static const ipu::SWConfig SW_CONFIGURATION = {
              .gapInit = -(ALN_GAP_OPENING_COST - ALN_GAP_EXTENDING_COST),
              .gapExtend = -ALN_GAP_EXTENDING_COST,
              .matchValue = ALN_MATCH_SCORE,
              .mismatchValue = -ALN_MISMATCH_COST,
              .ambiguityValue = -ALN_AMBIGUITY_COST,
              .similarity = swatlib::Similarity::nucleicAcid,
              .datatype = swatlib::DataType::nucleicAcid,
};

static const ipu::batchaffine::IPUAlgoConfig ALGO_CONFIGURATION = {
            KLIGN_IPU_TILES,
            KLIGN_IPU_MAXAB_SIZE,
            KLIGN_IPU_MAX_BATCHES,
            KLIGN_IPU_BUFSIZE,
            ipu::batchaffine::VertexType::cpp,
            ipu::batchaffine::partition::Algorithm::fillFirst
};

ipu::batchaffine::SWAlgorithm* getDriver();
void init_single_ipu(ipu::SWConfig config, ipu::batchaffine::IPUAlgoConfig algoconfig);



#endif // __DRIVER_H__