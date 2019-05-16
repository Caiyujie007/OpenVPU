#ifndef SRC_VPU_POOL_H_
#define SRC_VPU_POOL_H_

#include "vpu_basic.h"

void RunPool(unsigned int kernel_width,unsigned int kernel_height,unsigned int x_stride,unsigned int y_stride,unsigned int pooling_method,
		unsigned int pad_left,unsigned int pad_right,unsigned int pad_up,unsigned int pad_down,
		struct Mapped_Feature *feature_in,struct Mapped_Feature *feature_out);

void RunPool_soft(unsigned int kernel_width,unsigned int kernel_height,unsigned int x_stride,unsigned int y_stride,unsigned int pooling_method,
		unsigned int pad_left,unsigned int pad_right,unsigned int pad_up,unsigned int pad_down,
		struct Mapped_Feature *feature_in,struct Mapped_Feature *feature_out);

#endif /* SRC_VPU_POOL_H_ */
