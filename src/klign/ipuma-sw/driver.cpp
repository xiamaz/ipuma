#include "ipuma-sw/driver.hpp"
#include "ipuma-sw/ipu_base.h"
#include "ipuma-sw/ipu_batch_affine.h"

static ipu::batchaffine::SWAlgorithm *ipu_driver = nullptr;

ipu::batchaffine::SWAlgorithm* getDriver() {
        return ipu_driver;
}

void init_single_ipu(ipu::SWConfig config, int tiles, int maxAB, int maxBatches, int bufsize) {
  if (ipu_driver == NULL) {
    ipu_driver =
        new ipu::batchaffine::SWAlgorithm(config, tiles, maxAB, maxBatches, bufsize);
  }
}
