#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vpu_basic.h"
#include "vpu_conv.h"
#include "vpu_pool.h"
#include "FPGA_DDR.h"

#include "rdMNIST.h"
#include "MNIST_LARGE_cfg.h"

double Sigmoid(double in)
{
	return 1.0/(1.0+exp(-in));
}

int main()
{
	unsigned long int time;
	VPU_Init();

	struct Mapped_Weight *W_conv1=Malloc_Weight(5,5,1,32,PTR_W_CONV1);
	Load_Weight_From_File(W_conv1,"MNIST_LARGE_DAT/W_conv1.bin");
	struct Mapped_Feature *b_conv1=Malloc_Feature(1,1,32,PTR_B_CONV1,0,-1,-1);
	Load_Feature_From_File(b_conv1,"MNIST_LARGE_DAT/b_conv1.bin");
	short conv1_lut[LUT_ENTRIES];
	Get_LUT(PTR_H_CONV1_BEFORE_LUT,PTR_H_CONV1,Sigmoid,conv1_lut);
	
	struct Mapped_Weight *W_conv2=Malloc_Weight(5,5,32,64,PTR_W_CONV2);
	Load_Weight_From_File(W_conv2,"MNIST_LARGE_DAT/W_conv2.bin");
	struct Mapped_Feature *b_conv2=Malloc_Feature(1,1,64,PTR_B_CONV2,0,-1,-1);
	Load_Feature_From_File(b_conv2,"MNIST_LARGE_DAT/b_conv2.bin");
	short conv2_lut[LUT_ENTRIES];
	Get_LUT(PTR_H_CONV2_BEFORE_LUT,PTR_H_CONV2,Sigmoid,conv2_lut);
	
	struct Mapped_Weight *W_fc1=Malloc_Weight(7,7,64,1024,PTR_W_FC1);
	Load_Weight_From_File(W_fc1,"MNIST_LARGE_DAT/W_fc1.bin");
	struct Mapped_Feature *b_fc1=Malloc_Feature(1,1,1024,PTR_B_FC1,0,-1,-1);
	Load_Feature_From_File(b_fc1,"MNIST_LARGE_DAT/b_fc1.bin");
	short fc1_lut[LUT_ENTRIES];
	Get_LUT(PTR_H_FC1_BEFORE_LUT,PTR_H_FC1,Sigmoid,fc1_lut);

	struct Mapped_Weight *W_fc2=Malloc_Weight(1,1,1024,10,PTR_W_FC2);
	Load_Weight_From_File(W_fc2,"MNIST_LARGE_DAT/W_fc2.bin");
	struct Mapped_Feature *b_fc2=Malloc_Feature(1,1,10,PTR_B_FC2,0,-1,-1);
	Load_Feature_From_File(b_fc2,"MNIST_LARGE_DAT/b_fc2.bin");

	printf("Copied weight and bias from file to FPGA_DDR\n");

	struct Mapped_Feature *image=Malloc_Feature(28,28,1,PTR_IMG,0,-1,-1);

	struct Mapped_Feature *h_conv1=Malloc_Feature(28,28,32,PTR_H_CONV1,PTR_H_CONV1_BEFORE_LUT,-1,-1);
	struct Mapped_Feature *h_pool1=Malloc_Feature(14,14,32,PTR_H_POOL1,0,-1,-1);

	struct Mapped_Feature *h_conv2=Malloc_Feature(14,14,64,PTR_H_CONV2,PTR_H_CONV2_BEFORE_LUT,-1,-1);
	struct Mapped_Feature *h_pool2=Malloc_Feature(7,7,64,PTR_H_POOL2,0,-1,-1);

	struct Mapped_Feature *h_fc1=Malloc_Feature(1,1,1024,PTR_H_FC1,PTR_H_FC1_BEFORE_LUT,-1,-1);
	struct Mapped_Feature *out=Malloc_Feature(1,1,10,PTR_H_FC2,0,-1,-1);

	Debug_mcb();

	if((image==NULL)||(h_conv1==NULL)||(h_pool1==NULL)||(h_conv2==NULL)||(h_pool2==NULL)||(h_fc1==NULL)||(out==NULL))
	{
		printf("Failed to Malloc Memory\n");
		return 0;
	}
	else
	{
		printf("Malloc Memory Fisnsh\n");
	}

	int i;
	int correct=0;
	int sum=0;

	short ImgIn[28][28][1];
	int label[1];

	for(i=0;i<10000;i++)
	{
		short buf[1][1][1024];
	
		rdMNISTs28_test(i,i+1,(short (*)[28][28])ImgIn,(int *)&label);
		Map_Feature(ImgIn[0][0],image);

		//First Convolutional Layer
		//RunConv_Simplest(conv1_cfg,1,image,W_conv1,h_conv1);
		RunConv_With_Bias(conv1_cfg,0,conv1_lut,image,W_conv1,b_conv1,h_conv1);
		RunPool(2,2,2,2,1,0,0,0,0,h_conv1,h_pool1);

		//Second Convolutional Layer
		//RunConv_Simplest(conv2_cfg,1,h_pool1,W_conv2,h_conv2);
		RunConv_With_Bias(conv2_cfg,0,conv2_lut,h_pool1,W_conv2,b_conv2,h_conv2);
		RunPool(2,2,2,2,1,0,0,0,0,h_conv2,h_pool2);

		//FC1
		//RunConv_Simplest(fc1_cfg,1,h_pool2,W_fc1,h_fc1);
		RunConv_With_Bias(fc1_cfg,0,fc1_lut,h_pool2,W_fc1,b_fc1,h_fc1);

		//FC2
		//RunConv_Simplest(fc2_cfg,0,h_fc1,W_fc2,out);
		RunConv_With_Bias(fc2_cfg,0,NULL,h_fc1,W_fc2,b_fc2,out);

		short max=-32768;int predict=0;
		for(int j=0;j<10;j++)
		{
			if(*Get_Element(out,0,0,j) >max)
			{
				max=*Get_Element(out,0,0,j);
				predict=j;
			}
		}
		//printf("predict=%d,label=%d\n",predict,label[0]);

		sum++;
		if(predict==label[0])
			correct++;

		if(i%100==0)
			printf("Loop %d: Correct Rate:%f\n",i,correct*1.0/sum);

	}

	printf("Loop %d: Correct Rate:%f\n",i,correct*1.0/sum);

	return 0;
}
