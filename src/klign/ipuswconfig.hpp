
#include "driver.hpp"
#include "ipu_batch_affine.h"
#include "ipu_base.h"
#include "ipu_batch_affine.h"

const static       ipu::SWConfig MHM_SWCONFIG = {
          .gapInit = -(ALN_GAP_OPENING_COST - ALN_GAP_EXTENDING_COST),
          .gapExtend = -ALN_GAP_EXTENDING_COST,
          .matchValue = ALN_MATCH_SCORE,
          .mismatchValue = -ALN_MISMATCH_COST,
          .ambiguityValue = -ALN_AMBIGUITY_COST,
          .similarity = swatlib::Similarity::nucleicAcid,
          .datatype = swatlib::DataType::nucleicAcid,
      };

const static       ipu::IPUAlgoConfig MHM_ALGOCONFIG = {
          .tilesUsed = KLIGN_IPU_TILES,         // number of active vertices
          .maxAB = KLIGN_IPU_MAXAB_SIZE,        // maximum length of a single comparison
          .maxBatches = KLIGN_IPU_MAX_BATCHES,  // maximum number of comparisons in a single batch
          .bufsize = KLIGN_IPU_BUFSIZE,         // total size of buffer for A and B individually
          .vtype = ipu::VertexType::multiasm,
          .fillAlgo = ipu::Algorithm::roundRobin,
          .forwardOnly = false,  // do not calculate the start position of a match, this should approx 2x performance, as no reverse
                                 // pass is needed
          .useRemoteBuffer = false,
          .transmissionPrograms = 1,  // number of separate transmission programs, use only with remote!
          .ioTiles = 0,
      };