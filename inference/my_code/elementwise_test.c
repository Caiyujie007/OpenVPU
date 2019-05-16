#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vpu_basic.h"
#include "vpu_conv.h"
#include "vpu_pool.h"
#include "FPGA_DDR.h"

#define WIDTH 30
#define HEIGHT 30
#define CH 64

#define ELEMENT_WISE_OP 0 //0: add, 1: mul, 2: minus, 3: not defined

#define TENSOR_1_Presion 8
#define TENSOR_2_Presion 7
#define TENSOR_OUT_Presion 6

int main()
{
	VPU_Init();

	struct Mapped_Feature *feature_1=Malloc_Feature(HEIGHT,WIDTH,CH,TENSOR_1_Presion,0,-1,-1);
	struct Mapped_Feature *feature_2=Malloc_Feature(HEIGHT,WIDTH,CH,TENSOR_2_Presion,0,-1,-1);
	struct Mapped_Feature *feature_out=Malloc_Feature(HEIGHT,WIDTH,CH,TENSOR_OUT_Presion,0,-1,-1);

	Fill_Feature(1<<TENSOR_1_Presion,feature_1);//1.0
	Fill_Feature(1<<TENSOR_2_Presion,feature_2);//1.0

	Element_Wise(0,NULL,ELEMENT_WISE_OP,feature_1,feature_2,feature_out);

	for(int i=0;i<HEIGHT;i++)
		for(int j=0;j<WIDTH;j++)
			for(int k=0;k<CH;k++)
			{
				printf("OUT[%d][%d][%d]=%f\n",i,j,k,(*Get_Element(feature_out,i,i,k)*1.0)/(1<<TENSOR_OUT_Presion));
			}
}
