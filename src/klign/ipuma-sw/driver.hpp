
#ifndef __DRIVER_H__
#define __DRIVER_H__

#include "ipuma-sw/driver.hpp"
#include "ipuma-sw/ipu_base.h"
#include "ipuma-sw/ipu_batch_affine.h"


ipu::batchaffine::SWAlgorithm* getDriver();
void init_single_ipu(ipu::SWConfig config, int tiles, int maxAB, int maxBatches, int bufsize);



#endif // __DRIVER_H__