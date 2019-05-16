#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vpu_basic.h"
#include "vpu_conv.h"
#include "vpu_pool.h"
#include "FPGA_DDR.h"

#define POOL_METHOD 0 //0: min, 1: max, 2: mean

#define IN_CH 16
#define IN_WIDTH 200
#define IN_HEIGHT 300

#define KERNEL_WIDTH 2
#define KERNEL_HEIGHT 2
#define X_STRIDE 2
#define Y_STRIDE 2

#define OUT_CH IN_CH
#define OUT_WIDTH ((IN_WIDTH-KERNEL_WIDTH)/X_STRIDE+1)
#define OUT_HEIGHT ((IN_HEIGHT-KERNEL_HEIGHT)/Y_STRIDE+1)

short dat_in[IN_HEIGHT][IN_WIDTH][IN_CH];
short dat_out[OUT_HEIGHT][OUT_WIDTH][OUT_CH];

int main()
{
	int flag=1;
	short expected_result;

	printf("Hello World_pool\r\n");

	VPU_Init();

	struct Mapped_Feature *feature_in=Malloc_Feature(IN_HEIGHT,IN_WIDTH,IN_CH,0,0,-1,-1);
	if(feature_in==NULL)
	{
		printf("failed to malloc feature_in\r\n");
		return  0;
	}

	struct Mapped_Feature *feature_out=Malloc_Feature(OUT_HEIGHT,OUT_WIDTH,OUT_CH,0,0,-1,-1);
	if(feature_out==NULL)
	{
		printf("failed to malloc feature_out\r\n");
		return 0;
	}

	for(int i=0;i<IN_HEIGHT;i++)
		for(int j=0;j<IN_WIDTH;j++)
			for(int k=0;k<IN_CH;k++)
			{
				dat_in[i][j][k]=i*IN_WIDTH+j-30000;
			}

	Map_Feature(dat_in[0][0],feature_in);
	RunPool(KERNEL_WIDTH,KERNEL_HEIGHT,X_STRIDE,Y_STRIDE,POOL_METHOD,0,0,0,0,feature_in,feature_out);
	printf("Hardware Run Finish\n");

	DeMap_Feature(feature_out,dat_out[0][0]);

	for(int i=0;i<OUT_HEIGHT;i++)
		for(int j=0;j<OUT_WIDTH;j++)
			for(int k=0;k<OUT_CH;k++)
			{
				if(POOL_METHOD==0)//min
					expected_result=dat_in[i*Y_STRIDE][j*X_STRIDE][k];
				else
					if(POOL_METHOD==1)//max
						expected_result=dat_in[i*Y_STRIDE+KERNEL_HEIGHT-1][j*X_STRIDE+KERNEL_WIDTH-1][k];
					else
						if(POOL_METHOD==2)//mean
							expected_result=(dat_in[i*Y_STRIDE+KERNEL_HEIGHT-1][j*X_STRIDE+KERNEL_WIDTH-1][k]+dat_in[i*Y_STRIDE][j*X_STRIDE][k])/2;
				if(dat_out[i][j][k]!=expected_result)
				{
					flag=0;
					printf("dat_out[%0d][%0d][%0d]=%0d,expected_result=%0d\n",i,j,k,dat_out[i][j][k],expected_result);
				}
			}

	if(flag==1)
		printf("\n==================================\n\tresult match\n==================================\n");
	else
		printf("\n==================================\n\tresult mismatch\n==================================\n");

	return 0;
}
