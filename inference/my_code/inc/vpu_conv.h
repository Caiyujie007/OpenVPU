#ifndef SRC_VPU_CONV_H_
#define SRC_VPU_CONV_H_

#include "vpu_basic.h"

struct Conv_Cfg{
	int CHin;int Win;int CHout;
	int overlap;int Kx;int Ky;int Sx;int Sy;int pad_x;int pad_y;
	int dat_banks;int method;
	unsigned int out_width;unsigned int out_height;
	int entries_per_line;int wt_size;
	int K;
	int in_height_first;int in_height_middle;int in_height_last;
	int out_height_first;int out_height_middle;int out_height_last;
	int N;
	int out_ch_slice;
	int out_ch_slice_last;
};

struct Conv_Cfg Get_Conv_Cfg(unsigned int Hin,unsigned int Win,unsigned int CHin,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,
		unsigned int pad_left,unsigned int pad_right,unsigned int pad_up,unsigned int pad_down);

void RunConv_Simplest(struct Conv_Cfg conv_cfg,unsigned int relu_en,short lut[LUT_ENTRIES],
		struct Mapped_Feature *feature_in,struct Mapped_Weight *wt,struct Mapped_Feature *feature_out);

void RunConv_With_Bias(struct Conv_Cfg conv_cfg,unsigned int relu_en,short lut[LUT_ENTRIES],
		struct Mapped_Feature *feature_in,struct Mapped_Weight *wt,struct Mapped_Feature *bias,struct Mapped_Feature *feature_out);

void RunConv_With_Element_Wise(struct Conv_Cfg conv_cfg,unsigned int relu_en,short lut[LUT_ENTRIES],unsigned int element_wise_op,//0: add, 1: mul, 2: minus, 3: not defined
		struct Mapped_Feature *feature_in,struct Mapped_Weight *wt,struct Mapped_Feature *feature_second,struct Mapped_Feature *feature_out);

void RunConv_Simplest_Soft(
		struct Conv_Cfg conv_cfg,unsigned int relu_en,short lut[LUT_ENTRIES],
		struct Mapped_Feature *feature_in,
		struct Mapped_Weight *wt,
		struct Mapped_Feature *feature_out);

void RunConv_With_Bias_Soft(
		struct Conv_Cfg conv_cfg,unsigned int relu_en,short lut[LUT_ENTRIES],
		struct Mapped_Feature *feature_in,
		struct Mapped_Weight *wt,struct Mapped_Feature *bias,
		struct Mapped_Feature *feature_out);

//////////////////////////////////////////////
//internal function, Don't Call them
void Config_Conv_Path(unsigned int CHin,unsigned int Hin,unsigned int Win,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,
		unsigned int pad_x,unsigned int pad_y,
		unsigned int feature_in_base,unsigned int feature_in_surface_stride,unsigned int feature_in_line_stride,
		unsigned int wt_base_addr,unsigned int wt_size,
		unsigned int feature_out_base,unsigned int feature_out_surface_stride,unsigned int feature_out_line_stride,
		unsigned int out_width,unsigned int out_height,unsigned int dat_buf_num,unsigned int cdma_dat_reuse,unsigned int cdma_wt_reuse);

void RunConv_Simplest_single_time(unsigned int CHin,unsigned int Hin,unsigned int Win,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,
		unsigned int pad_x,unsigned int pad_y,
		unsigned int relu_en,unsigned int use_lut,
		unsigned int feature_in_base,unsigned int feature_in_surface_stride,unsigned int feature_in_line_stride,unsigned int feature_in_precision,
		unsigned int wt_base_addr,unsigned int wt_size,unsigned int wt_precision,
		unsigned int feature_out_base,unsigned int feature_out_surface_stride,unsigned int feature_out_line_stride,unsigned int feature_out_precision,
		unsigned int out_width,unsigned int out_height,unsigned int dat_buf_num,unsigned int cdma_dat_reuse,unsigned int cdma_wt_reuse);

void RunConv_With_Bias_single_time(unsigned int CHin,unsigned int Hin,unsigned int Win,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,
		unsigned int pad_x,unsigned int pad_y,
		unsigned int relu_en,unsigned int use_lut,
		unsigned int feature_in_base,unsigned int feature_in_surface_stride,unsigned int feature_in_line_stride,unsigned int feature_in_precision,
		unsigned int wt_base_addr,unsigned int wt_size,unsigned int wt_precision,
		unsigned int bias_base,unsigned int bias_precision,
		unsigned int feature_out_base,unsigned int feature_out_surface_stride,unsigned int feature_out_line_stride,unsigned int feature_out_precision,
		unsigned int out_width,unsigned int out_height,unsigned int dat_buf_num,unsigned int cdma_dat_reuse,unsigned int cdma_wt_reuse);

void RunConv_With_Element_Wise_single_time(unsigned int CHin,unsigned int Hin,unsigned int Win,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,
		unsigned int pad_x,unsigned int pad_y,
		unsigned int relu_en,unsigned int use_lut,unsigned int element_wise_op,
		unsigned int feature_in_base,unsigned int feature_in_surface_stride,unsigned int feature_in_line_stride,unsigned int feature_in_precision,
		unsigned int wt_base_addr,unsigned int wt_size,unsigned int wt_precision,
		unsigned int second_tensor_base,unsigned int second_tensor_surface_stride,unsigned int second_tensor_line_stride,unsigned int second_tensor_precision,
		unsigned int feature_out_base,unsigned int feature_out_surface_stride,unsigned int feature_out_line_stride,unsigned int feature_out_precision,
		unsigned int out_width,unsigned int out_height,unsigned int dat_buf_num,unsigned int cdma_dat_reuse,unsigned int cdma_wt_reuse);
///////////////////////////////////////////////

#endif /* SRC_VPU_CONV_H_ */
