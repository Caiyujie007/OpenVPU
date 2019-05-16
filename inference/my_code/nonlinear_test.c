#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vpu_basic.h"
#include "vpu_conv.h"
#include "vpu_pool.h"
#include "FPGA_DDR.h"

#define WIDTH 1 
#define HEIGHT 1
#define CH 64

#define ELEMENT_WISE_OP 0 //0: add, 1: mul, 2: minus, 3: not defined

#define TENSOR_IN_Presion 12
#define TENSOR_OUT_Presion 9

double sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}

double tanh(double x)
{
	return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

int main()
{
	VPU_Init();
	
	struct Mapped_Feature *feature_in=Malloc_Feature(HEIGHT,WIDTH,CH,TENSOR_IN_Presion,0,-1,-1);
	struct Mapped_Feature *feature_out=Malloc_Feature(HEIGHT,WIDTH,CH,TENSOR_OUT_Presion,0,-1,-1);

	short lut[LUT_ENTRIES];
	Get_LUT(feature_in->precision,feature_out->precision,tanh,lut);

	for(int i=0;i<HEIGHT;i++)
		for(int j=0;j<WIDTH;j++)
			for(int k=0;k<CH;k++)
			{
				double tp=(k*6.0)/CH-3.0;//range:[-3,3)
				*Get_Element(feature_in,i,i,k)=(short)(tp*(1<<TENSOR_IN_Presion));
				//printf("IN[%d][%d][%d]=%d\n",i,j,k,*Get_Element(feature_in,i,i,k));
			}

	Nonlinear_OP(0,lut,feature_in,feature_out);

	for(int i=0;i<HEIGHT;i++)
		for(int j=0;j<WIDTH;j++)
			for(int k=0;k<CH;k++)
			{
				printf("OUT[%d][%d][%d]=%f\n",i,j,k,(*Get_Element(feature_out,i,i,k)*1.0)/(1<<TENSOR_OUT_Presion));
				//printf("OUT[%d][%d][%d]=%d\n",i,j,k,*Get_Element(feature_out,i,i,k));
			}
}
