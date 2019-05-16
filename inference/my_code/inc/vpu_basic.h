#ifndef SRC_VPU_BASIC_H_
#define SRC_VPU_BASIC_H_

#include <memory.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <math.h>
#include "FPGA_DDR.h"

//#define GET_PERFORMANCE

#define Tc 32					//Parallel factor of CH_in
#define Tk 16					//Parallel factor of CH_out
#define Tp 16					//Maxinum Stripe length
#define Logic_MEM_DEP 256
#define Logic_MEM_NUM 16
#define PDP_CORE_2D_LINEBUFF_LEN 49
#define LUT_ENTRIES 128
#define log2LUT_ENTRIES 7

#define SDP_REG_BIAS 64
#define PDP_REG_BIAS 256

struct Mapped_Feature
{
	short *payload;
	unsigned int payload_size;
	unsigned int surface_stride;
	unsigned int line_stride;
	unsigned int precision;
	unsigned int precision_for_conv_out_sft;
	unsigned int height;
	unsigned int width;
	unsigned int channel;
};

struct Mapped_Weight
{
	short *payload;
	unsigned int payload_size;
	unsigned int precision;
	unsigned int Ky;
	unsigned int Kx;
	unsigned int in_ch;
	unsigned int out_ch;
};

void VPU_Init();

struct Mapped_Feature *Malloc_Feature(unsigned int height,unsigned int width,unsigned int ch,unsigned int precision,unsigned int precision_for_conv_out_sft,int line_stride,int surface_stride);
void Feature_CPY_H2C(struct Mapped_Feature *feature);
void Feature_CPY_C2H(struct Mapped_Feature *feature);
void Free_Feature(struct Mapped_Feature *feature);
void Fill_Feature_Soft(short value,struct Mapped_Feature *feature);//Use Software to fill DDR on PC
void Fill_Feature(short value,struct Mapped_Feature *feature);//Call Hardware to fill DDR on FPGA board

struct Mapped_Weight *Malloc_Weight(unsigned int Ky,unsigned int Kx,unsigned int in_ch,unsigned int out_ch,unsigned int precision);
void Weight_CPY_H2C(struct Mapped_Weight *weight);
void Free_Weight(struct Mapped_Weight *weight);

void CSB_Write(unsigned int addr,unsigned int data);
unsigned int CSB_Read(unsigned int addr);

void Map_Feature(short *in,struct Mapped_Feature *feature);
void Load_Feature_From_File(struct Mapped_Feature *feature,const char *filename);
void DeMap_Feature(struct Mapped_Feature *feature,short *out);

void Load_Weight_From_File(struct Mapped_Weight *weight,const char *filename);
void Map_Weight(short *kernel,struct Mapped_Weight *weight);

short* Get_Element(struct Mapped_Feature *feature,unsigned int row,unsigned int col,unsigned int ch);
short* Get_Weight(struct Mapped_Weight *weight,unsigned int n_h,unsigned int n_w,unsigned int n_cin,unsigned int n_cout);

void Get_LUT(unsigned int precision_in,unsigned int precision_out,double (*func)(double),short lut[LUT_ENTRIES]);

void Nonlinear_OP(unsigned int relu_en,short lut[LUT_ENTRIES],struct Mapped_Feature *feature_in,struct Mapped_Feature *feature_out);

void Element_Wise(unsigned int relu_en,short lut[LUT_ENTRIES],unsigned int element_wise_op,//0: add, 1: mul, 2: minus, 3: not defined
		struct Mapped_Feature *feature_1,struct Mapped_Feature *feature_2,struct Mapped_Feature *feature_out);

#endif /* SRC_VPU_BASIC_H_ */
