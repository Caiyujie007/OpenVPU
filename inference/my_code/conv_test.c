#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vpu_basic.h"
#include "vpu_conv.h"
#include "vpu_pool.h"
#include "FPGA_DDR.h"

#define IN_WIDTH 50
#define IN_HEIGHT 50 
#define IN_CH 16

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5
#define X_STRIDE 1
#define Y_STRIDE 1
#define X_PADDING 1
#define Y_PADDING 1

#define OUT_CH 16
#define OUT_WIDTH ((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1)
#define OUT_HEIGHT ((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1)

#define RELU_EN 1
#define DAT_IN_Precision 3
#define WT_Precision 6
#define DAT_OUT_Precision 9

#define BIAS_Precision 9

#define ELEMENT_WISE_OP 0 //0: add, 1: mul, 2: minus, 3: not defined
#define SECOND_TENSOR_OUT_Precision 9

short dat_in[IN_HEIGHT][IN_WIDTH][IN_CH];
short wt[KERNEL_HEIGHT][KERNEL_WIDTH][IN_CH][OUT_CH];
short bias[OUT_CH];
short sencond_tensor[OUT_HEIGHT][OUT_WIDTH][OUT_CH];
short dat_out[OUT_HEIGHT][OUT_WIDTH][OUT_CH];
short dat_out_soft[OUT_HEIGHT][OUT_WIDTH][OUT_CH];

int main()
{
	int flag=1;
	
	printf("Hello World_conv\r\n");
	VPU_Init();

	struct Mapped_Feature *feature_in=Malloc_Feature(IN_HEIGHT,IN_WIDTH,IN_CH,DAT_IN_Precision,0,-1,-1);
	if(feature_in==NULL)
	{
		printf("failed to malloc feature_in\n");
		return  0;
	}

	struct Mapped_Feature *feature_second=Malloc_Feature(OUT_HEIGHT,OUT_WIDTH,OUT_CH,SECOND_TENSOR_OUT_Precision,0,-1,-1);
	if(feature_second==NULL)
	{
		printf("failed to malloc feature_second\n");
		return 0;
	}

	struct Mapped_Feature *feature_bias=Malloc_Feature(1,1,OUT_CH,BIAS_Precision,0,-1,-1);
	if(feature_bias==NULL)
	{
		printf("failed to malloc feature_bias\n");
		return 0;
	}

	struct Mapped_Feature *feature_out=Malloc_Feature(OUT_HEIGHT,OUT_WIDTH,OUT_CH,DAT_OUT_Precision,DAT_OUT_Precision,-1,-1);
	struct Mapped_Feature *feature_out_soft=Malloc_Feature(OUT_HEIGHT,OUT_WIDTH,OUT_CH,DAT_OUT_Precision,DAT_OUT_Precision,-1,-1);
	if((feature_out==NULL) || (feature_out_soft==NULL))
	{
		printf("failed to malloc feature_out and feature_out_soft\n");
		return 0;
	}

	struct Mapped_Weight *weight=Malloc_Weight(KERNEL_HEIGHT,KERNEL_WIDTH,IN_CH,OUT_CH,WT_Precision);
	if(weight==NULL)
	{
		printf("failed to malloc weight\n");
		return 0;
	}

	Debug_mcb();

	for(int i=0;i<IN_HEIGHT;i++)
		for(int j=0;j<IN_WIDTH;j++)
			for(int k=0;k<IN_CH;k++)
			{
				dat_in[i][j][k]=((i+j-k)%21)-10;//(rand()%21)-10;//i+j+1;//`IN_WIDTH*i+j;
			}

	for(int i=0;i<KERNEL_HEIGHT;i++)
		for(int j=0;j<KERNEL_WIDTH;j++)
			for(int k=0;k<IN_CH;k++)
				for(int l=0;l<OUT_CH;l++)
				{
					wt[i][j][k][l]=((i-j+k-l)%21)-10;//i+j+1;//i*`KERNEL_WIDTH+j+1;
				}

	for(int i=0;i<OUT_CH;i++)
		bias[i]=-i;//-i;

	for(int i=0;i<OUT_HEIGHT;i++)
		for(int j=0;j<OUT_WIDTH;j++)
			for(int k=0;k<OUT_CH;k++)
				sencond_tensor[i][j][k]=-(i+j+k)*20;

	printf("Vector initialize Finish\n");

	Map_Feature(dat_in[0][0],feature_in);
	Map_Weight(wt[0][0][0],weight);
	Map_Feature(sencond_tensor[0][0],feature_second);
	Map_Feature(bias,feature_bias);

	printf("Mapping Feature and Weight Finish\n");

	struct Conv_Cfg conv_cfg=Get_Conv_Cfg(IN_HEIGHT,IN_WIDTH,IN_CH,OUT_CH,
						KERNEL_WIDTH,KERNEL_HEIGHT,X_STRIDE,Y_STRIDE,
						X_PADDING,X_PADDING,Y_PADDING,Y_PADDING);

	printf("K=%d,N=%d\r\n",conv_cfg.K,conv_cfg.N);
	printf("Out_H=%d,Out_W=%d\n",OUT_HEIGHT,OUT_WIDTH);

//	Fill_Feature(0,feature_out);//Call Hardware to fill DDR on FPGA board
	RunConv_Simplest(conv_cfg,RELU_EN,NULL,
			feature_in,weight,feature_out);

//	RunConv_With_Bias(conv_cfg,RELU_EN,NULL,
//			feature_in,weight,feature_bias,feature_out);

//	RunConv_With_Element_Wise(conv_cfg,RELU_EN,NULL,ELEMENT_WISE_OP,
//			feature_in,weight,feature_second,feature_out);

	printf("Hardware Run Finish\n");

	unsigned int dat_num=((IN_CH+Tk-1)/Tk)*Tk*IN_HEIGHT*IN_WIDTH+((IN_CH+Tc-1)/Tc)*Tc*OUT_CH*KERNEL_WIDTH*KERNEL_HEIGHT+((OUT_CH+Tk-1)/Tk)*Tk*OUT_HEIGHT*OUT_WIDTH;
	unsigned int mac_num=OUT_HEIGHT*OUT_WIDTH*OUT_CH*IN_CH*KERNEL_HEIGHT*KERNEL_WIDTH;
	printf("dat_num=%d,mac_num=%d\r\n",dat_num,OUT_HEIGHT*OUT_WIDTH*OUT_CH*IN_CH*KERNEL_HEIGHT*KERNEL_WIDTH);

	DeMap_Feature(feature_out,dat_out[0][0]);

	RunConv_Simplest_Soft(conv_cfg,RELU_EN,NULL,
			feature_in,weight,feature_out_soft);

	DeMap_Feature(feature_out_soft,dat_out_soft[0][0]);

	printf("Software Run Finish\n");

	for(int i=0;i<OUT_HEIGHT;i++)
		for(int j=0;j<OUT_WIDTH;j++)
			for(int k=0;k<OUT_CH;k++)
			{
				if(dat_out[i][j][k]!=(dat_out_soft[i][j][k]))
				//if(dat_out[i][j][k]!=(dat_out_soft[i][j][k]+bias[k]))
				//if(dat_out[i][j][k]!=(dat_out_soft[i][j][k]+sencond_tensor[i][j][k]))
				{
					flag=0;
					printf("dat_out     [%0d][%0d][%0d]=%0d\n",i,j,k,dat_out[i][j][k]);
					printf("dat_out_soft[%0d][%0d][%0d]=%0d\n",i,j,k,dat_out_soft[i][j][k]);
				}
			}

	if(flag==1)
		printf("\n==================================\n\tresult match\n==================================\n");
	else
		printf("\n==================================\n\tresult mismatch\n==================================\n");

	Free_Feature(feature_in);
//	Free_Feature(feature_out);

	return 0;
}
