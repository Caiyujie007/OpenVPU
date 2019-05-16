#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "vpu_basic.h"
#include "vpu_conv.h"
#include "vpu_pool.h"
#include "FPGA_DDR.h"

#include "VGG_cfg.h"
#include "PTR.h"

void LoadData(short *dat, char *filename, int size);

int main()
{
	unsigned long int time;

	printf("Hello World_mnist_large\n");
	VPU_Init();

	struct Mapped_Weight *W_conv1_1=Malloc_Weight(3,3,3,64,PTR_W_CONV1_1);
	Load_Weight_From_File(W_conv1_1,"VGG16_DAT/conv1_1_W.bin");
	struct Mapped_Feature *b_conv1_1=Malloc_Feature(1,1,64,PTR_B_CONV1_1,0,-1,-1);
	Load_Feature_From_File(b_conv1_1,"VGG16_DAT/conv1_1_b.bin");
	if((W_conv1_1==NULL)||(b_conv1_1==NULL)) {
		printf("Failed to Malloc conv1_1");return 0;}

	struct Mapped_Weight *W_conv1_2=Malloc_Weight(3,3,64,64,PTR_W_CONV1_2);
	Load_Weight_From_File(W_conv1_2,"VGG16_DAT/conv1_2_W.bin");
	struct Mapped_Feature *b_conv1_2=Malloc_Feature(1,1,64,PTR_B_CONV1_2,0,-1,-1);
	Load_Feature_From_File(b_conv1_2,"VGG16_DAT/conv1_2_b.bin");
	if((W_conv1_2==NULL)||(b_conv1_2==NULL)) {
		printf("Failed to Malloc conv1_2");return 0;}

	struct Mapped_Weight *W_conv2_1=Malloc_Weight(3,3,64,128,PTR_W_CONV2_1);
	Load_Weight_From_File(W_conv2_1,"VGG16_DAT/conv2_1_W.bin");
	struct Mapped_Feature *b_conv2_1=Malloc_Feature(1,1,128,PTR_B_CONV2_1,0,-1,-1);
	Load_Feature_From_File(b_conv2_1,"VGG16_DAT/conv2_1_b.bin");
	if((W_conv2_1==NULL)||(b_conv2_1==NULL)) {
		printf("Failed to Malloc conv2_1");return 0;}

	struct Mapped_Weight *W_conv2_2=Malloc_Weight(3,3,128,128,PTR_W_CONV2_2);
	Load_Weight_From_File(W_conv2_2,"VGG16_DAT/conv2_2_W.bin");
	struct Mapped_Feature *b_conv2_2=Malloc_Feature(1,1,128,PTR_B_CONV2_2,0,-1,-1);
	Load_Feature_From_File(b_conv2_2,"VGG16_DAT/conv2_2_b.bin");
	if((W_conv2_2==NULL)||(b_conv2_2==NULL)) {
		printf("Failed to Malloc conv2_2");return 0;}

	struct Mapped_Weight *W_conv3_1=Malloc_Weight(3,3,128,256,PTR_W_CONV3_1);
	Load_Weight_From_File(W_conv3_1,"VGG16_DAT/conv3_1_W.bin");
	struct Mapped_Feature *b_conv3_1=Malloc_Feature(1,1,256,PTR_B_CONV3_1,0,-1,-1);
	Load_Feature_From_File(b_conv3_1,"VGG16_DAT/conv3_1_b.bin");
	if((W_conv3_1==NULL)||(b_conv3_1==NULL)) {
		printf("Failed to Malloc conv3_1");return 0;}

	struct Mapped_Weight *W_conv3_2=Malloc_Weight(3,3,256,256,PTR_W_CONV3_2);
	Load_Weight_From_File(W_conv3_2,"VGG16_DAT/conv3_2_W.bin");
	struct Mapped_Feature *b_conv3_2=Malloc_Feature(1,1,256,PTR_B_CONV3_2,0,-1,-1);
	Load_Feature_From_File(b_conv3_2,"VGG16_DAT/conv3_2_b.bin");
	if((W_conv3_2==NULL)||(b_conv3_2==NULL)) {
		printf("Failed to Malloc conv3_2");return 0;}

	struct Mapped_Weight *W_conv3_3=Malloc_Weight(3,3,256,256,PTR_W_CONV3_3);
	Load_Weight_From_File(W_conv3_3,"VGG16_DAT/conv3_3_W.bin");
	struct Mapped_Feature *b_conv3_3=Malloc_Feature(1,1,256,PTR_B_CONV3_3,0,-1,-1);
	Load_Feature_From_File(b_conv3_3,"VGG16_DAT/conv3_3_b.bin");
	if((W_conv3_3==NULL)||(b_conv3_3==NULL)) {
		printf("Failed to Malloc conv3_3");return 0;}

	struct Mapped_Weight *W_conv4_1=Malloc_Weight(3,3,256,512,PTR_W_CONV4_1);
	Load_Weight_From_File(W_conv4_1,"VGG16_DAT/conv4_1_W.bin");
	struct Mapped_Feature *b_conv4_1=Malloc_Feature(1,1,512,PTR_B_CONV4_1,0,-1,-1);
	Load_Feature_From_File(b_conv4_1,"VGG16_DAT/conv4_1_b.bin");
	if((W_conv4_1==NULL)||(b_conv4_1==NULL)) {
		printf("Failed to Malloc conv4_1");return 0;}

	struct Mapped_Weight *W_conv4_2=Malloc_Weight(3,3,512,512,PTR_W_CONV4_2);
	Load_Weight_From_File(W_conv4_2,"VGG16_DAT/conv4_2_W.bin");
	struct Mapped_Feature *b_conv4_2=Malloc_Feature(1,1,512,PTR_B_CONV4_2,0,-1,-1);
	Load_Feature_From_File(b_conv4_2,"VGG16_DAT/conv4_2_b.bin");
	if((W_conv4_2==NULL)||(b_conv4_2==NULL)) {
		printf("Failed to Malloc conv4_2");return 0;}

	struct Mapped_Weight *W_conv4_3=Malloc_Weight(3,3,512,512,PTR_W_CONV4_3);
	Load_Weight_From_File(W_conv4_3,"VGG16_DAT/conv4_3_W.bin");
	struct Mapped_Feature *b_conv4_3=Malloc_Feature(1,1,512,PTR_B_CONV4_3,0,-1,-1);
	Load_Feature_From_File(b_conv4_3,"VGG16_DAT/conv4_3_b.bin");
	if((W_conv4_3==NULL)||(b_conv4_3==NULL)) {
		printf("Failed to Malloc conv4_3");return 0;}

	struct Mapped_Weight *W_conv5_1=Malloc_Weight(3,3,512,512,PTR_W_CONV5_1);
	Load_Weight_From_File(W_conv5_1,"VGG16_DAT/conv5_1_W.bin");
	struct Mapped_Feature *b_conv5_1=Malloc_Feature(1,1,512,PTR_B_CONV5_1,0,-1,-1);
	Load_Feature_From_File(b_conv5_1,"VGG16_DAT/conv5_1_b.bin");
	if((W_conv5_1==NULL)||(b_conv5_1==NULL)) {
		printf("Failed to Malloc conv5_1");return 0;}

	struct Mapped_Weight *W_conv5_2=Malloc_Weight(3,3,512,512,PTR_W_CONV5_2);
	Load_Weight_From_File(W_conv5_2,"VGG16_DAT/conv5_2_W.bin");
	struct Mapped_Feature *b_conv5_2=Malloc_Feature(1,1,512,PTR_B_CONV5_2,0,-1,-1);
	Load_Feature_From_File(b_conv5_2,"VGG16_DAT/conv5_2_b.bin");
	if((W_conv5_2==NULL)||(b_conv5_2==NULL)) {
		printf("Failed to Malloc conv5_2");return 0;}

	struct Mapped_Weight *W_conv5_3=Malloc_Weight(3,3,512,512,PTR_W_CONV5_3);
	Load_Weight_From_File(W_conv5_3,"VGG16_DAT/conv5_3_W.bin");
	struct Mapped_Feature *b_conv5_3=Malloc_Feature(1,1,512,PTR_B_CONV5_3,0,-1,-1);
	Load_Feature_From_File(b_conv5_3,"VGG16_DAT/conv5_3_b.bin");
	if((W_conv5_3==NULL)||(b_conv5_3==NULL)) {
		printf("Failed to Malloc conv5_3");return 0;}

	struct Mapped_Weight *W_fc6_part0=Malloc_Weight(7,7,128,4096,PTR_W_FC6);
	Load_Weight_From_File(W_fc6_part0,"VGG16_DAT/fc6_W_part0.bin");
	struct Mapped_Weight *W_fc6_part1=Malloc_Weight(7,7,128,4096,PTR_W_FC6);
	Load_Weight_From_File(W_fc6_part1,"VGG16_DAT/fc6_W_part1.bin");
	struct Mapped_Weight *W_fc6_part2=Malloc_Weight(7,7,128,4096,PTR_W_FC6);
	Load_Weight_From_File(W_fc6_part2,"VGG16_DAT/fc6_W_part2.bin");
	struct Mapped_Weight *W_fc6_part3=Malloc_Weight(7,7,128,4096,PTR_W_FC6);
	Load_Weight_From_File(W_fc6_part3,"VGG16_DAT/fc6_W_part3.bin");

	struct Mapped_Feature *b_fc6=Malloc_Feature(1,1,4096,PTR_B_FC6,0,-1,-1);
	Load_Feature_From_File(b_fc6,"VGG16_DAT/fc6_b.bin");
	if((W_fc6_part3==NULL)||(b_fc6==NULL)) {
		printf("Failed to Malloc fc6");return 0;}

	struct Mapped_Weight *W_fc7=Malloc_Weight(1,1,4096,4096,PTR_W_FC7);
	Load_Weight_From_File(W_fc7,"VGG16_DAT/fc7_W.bin");
	struct Mapped_Feature *b_fc7=Malloc_Feature(1,1,4096,PTR_B_FC7,0,-1,-1);
	Load_Feature_From_File(b_fc7,"VGG16_DAT/fc7_b.bin");
	if((W_fc7==NULL)||(b_fc7==NULL)) {
		printf("Failed to Malloc fc7");return 0;}

	struct Mapped_Weight *W_fc8=Malloc_Weight(1,1,4096,1000,PTR_W_FC8);
	Load_Weight_From_File(W_fc8,"VGG16_DAT/fc8_W.bin");
	struct Mapped_Feature *b_fc8=Malloc_Feature(1,1,1000,PTR_B_FC8,0,-1,-1);
	Load_Feature_From_File(b_fc8,"VGG16_DAT/fc8_b.bin");
	if((W_fc8==NULL)||(b_fc8==NULL)) {
		printf("Failed to Malloc fc8");return 0;}

	printf("Copied weight and bias from file to FPGA_DDR\n");

	struct Mapped_Feature *image=Malloc_Feature(224,224,3,PTR_IMG,0,-1,-1);

	struct Mapped_Feature *h_conv1_1=Malloc_Feature(224,224,64,PTR_H_CONV1_1,PTR_H_CONV1_1,-1,-1);
	struct Mapped_Feature *h_conv1_2=Malloc_Feature(224,224,64,PTR_H_CONV1_2,PTR_H_CONV1_2,-1,-1);
	struct Mapped_Feature *h_pool1=Malloc_Feature(112,112,64,PTR_H_POOL1,0,-1,-1);
	if((h_conv1_1==NULL)||(h_conv1_2==NULL)||(h_pool1==NULL)) {
		printf("Failed to Malloc feature_conv1");return 0;}

	struct Mapped_Feature *h_conv2_1=Malloc_Feature(112,112,128,PTR_H_CONV2_1,PTR_H_CONV2_1,-1,-1);
	struct Mapped_Feature *h_conv2_2=Malloc_Feature(112,112,128,PTR_H_CONV2_2,PTR_H_CONV2_2,-1,-1);
	struct Mapped_Feature *h_pool2=Malloc_Feature(56,56,128,PTR_H_POOL2,0,-1,-1);
	if((h_conv2_1==NULL)||(h_conv2_2==NULL)||(h_pool2==NULL)) {
		printf("Failed to Malloc feature_conv2");return 0;}

	struct Mapped_Feature *h_conv3_1=Malloc_Feature(56,56,256,PTR_H_CONV3_1,PTR_H_CONV3_1,-1,-1);
	struct Mapped_Feature *h_conv3_2=Malloc_Feature(56,56,256,PTR_H_CONV3_2,PTR_H_CONV3_2,-1,-1);
	struct Mapped_Feature *h_conv3_3=Malloc_Feature(56,56,256,PTR_H_CONV3_3,PTR_H_CONV3_3,-1,-1);
	struct Mapped_Feature *h_pool3=Malloc_Feature(28,28,256,PTR_H_POOL3,0,-1,-1);
	if((h_conv3_1==NULL)||(h_conv3_2==NULL)||(h_conv3_3==NULL)||(h_pool3==NULL)) {
		printf("Failed to Malloc feature_conv3");return 0;}

	struct Mapped_Feature *h_conv4_1=Malloc_Feature(28,28,512,PTR_H_CONV4_1,PTR_H_CONV4_1,-1,-1);
	struct Mapped_Feature *h_conv4_2=Malloc_Feature(28,28,512,PTR_H_CONV4_2,PTR_H_CONV4_2,-1,-1);
	struct Mapped_Feature *h_conv4_3=Malloc_Feature(28,28,512,PTR_H_CONV4_3,PTR_H_CONV4_3,-1,-1);
	struct Mapped_Feature *h_pool4=Malloc_Feature(14,14,512,PTR_H_POOL4,0,-1,-1);
	if((h_conv4_1==NULL)||(h_conv4_2==NULL)||(h_conv4_3==NULL)||(h_pool4==NULL)) {
		printf("Failed to Malloc feature_conv4");return 0;}

	struct Mapped_Feature *h_conv5_1=Malloc_Feature(14,14,512,PTR_H_CONV5_1,PTR_H_CONV5_1,-1,-1);
	struct Mapped_Feature *h_conv5_2=Malloc_Feature(14,14,512,PTR_H_CONV5_2,PTR_H_CONV5_2,-1,-1);
	struct Mapped_Feature *h_conv5_3=Malloc_Feature(14,14,512,PTR_H_CONV5_3,PTR_H_CONV5_3,-1,-1);
	struct Mapped_Feature *h_pool5=Malloc_Feature(7,7,512,PTR_H_POOL5,0,-1,-1);
	struct Mapped_Feature h_pool5_part[4];
	for(int i=0;i<4;i++)
	{
		h_pool5_part[i].payload=h_pool5->payload+((h_pool5->channel/4)/Tk)*(h_pool5->surface_stride)/2*i;//OP to short *
		h_pool5_part[i].payload_size=h_pool5->payload_size/4;
		h_pool5_part[i].surface_stride=h_pool5->surface_stride;
		h_pool5_part[i].line_stride=h_pool5->line_stride;
		h_pool5_part[i].precision=h_pool5->precision;
		h_pool5_part[i].precision_for_conv_out_sft=h_pool5->precision_for_conv_out_sft;
		h_pool5_part[i].height=h_pool5->height;
		h_pool5_part[i].width=h_pool5->width;
		h_pool5_part[i].channel=h_pool5->channel/4;
	}
	if((h_conv5_1==NULL)||(h_conv5_2==NULL)||(h_conv5_3==NULL)||(h_pool5==NULL)) {
		printf("Failed to Malloc feature_conv5");return 0;}

	struct Mapped_Feature *h_fc6_part0=Malloc_Feature(1,1,4096,PTR_H_FC6,PTR_H_FC6,-1,-1);
	struct Mapped_Feature *h_fc6_part1=Malloc_Feature(1,1,4096,PTR_H_FC6,PTR_H_FC6,-1,-1);
	struct Mapped_Feature *h_fc6_part2=Malloc_Feature(1,1,4096,PTR_H_FC6,PTR_H_FC6,-1,-1);
	struct Mapped_Feature *h_fc6=Malloc_Feature(1,1,4096,PTR_H_FC6,PTR_H_FC6,-1,-1);
	struct Mapped_Feature *h_fc7=Malloc_Feature(1,1,4096,PTR_H_FC7,PTR_H_FC7,-1,-1);
	struct Mapped_Feature *out=Malloc_Feature(1,1,1000,PTR_H_FC8,PTR_H_FC8,-1,-1);
	if((h_fc6==NULL)||(h_fc7==NULL)||(out==NULL)) {
		printf("Failed to Malloc feature_fc[6,7,8]");return 0;}

	//Debug_mcb();
	printf("Malloc Memory Done\n");

	short image_3D[224][224][3];

	char str[100];
	char cmd[100];
	FILE* fp=fopen("/dat/ILSVR2012/dataset_valid_bin/filelist","r");
	if(fp==NULL) {
		printf("File /dat/ILSVR2012/dataset_valid_bin/filelist cannot be opend\n");}

	int i=0;
	while(fgets(str,100,fp))
	{
		if(i==21) break;
		i++;

		int strlen_origin=strlen(str);
		clock_t start,end;

		str[strlen_origin-1]='\0';//remove "\n"

		LoadData(image_3D[0][0],str,224*224*3*2);
		Map_Feature(image_3D[0][0],image);

		str[strlen_origin-4]='J';str[strlen_origin-3]='P';str[strlen_origin-2]='E';str[strlen_origin-1]='G';
		printf("Inferencing %s\n",str);

		start=clock();
		RunConv_With_Bias(conv1_1_cfg,1,NULL,image,W_conv1_1,b_conv1_1,h_conv1_1);
		RunConv_With_Bias(conv1_2_cfg,1,NULL,h_conv1_1,W_conv1_2,b_conv1_2,h_conv1_2);
		RunPool(2,2,2,2,1,0,0,0,0,h_conv1_2,h_pool1);

		RunConv_With_Bias(conv2_1_cfg,1,NULL,h_pool1,W_conv2_1,b_conv2_1,h_conv2_1);
		RunConv_With_Bias(conv2_2_cfg,1,NULL,h_conv2_1,W_conv2_2,b_conv2_2,h_conv2_2);
		RunPool(2,2,2,2,1,0,0,0,0,h_conv2_2,h_pool2);

		RunConv_With_Bias(conv3_1_cfg,1,NULL,h_pool2,W_conv3_1,b_conv3_1,h_conv3_1);
		RunConv_With_Bias(conv3_2_cfg,1,NULL,h_conv3_1,W_conv3_2,b_conv3_2,h_conv3_2);
		RunConv_With_Bias(conv3_3_cfg,1,NULL,h_conv3_2,W_conv3_3,b_conv3_3,h_conv3_3);
		RunPool(2,2,2,2,1,0,0,0,0,h_conv3_3,h_pool3);

		RunConv_With_Bias(conv4_1_cfg,1,NULL,h_pool3,W_conv4_1,b_conv4_1,h_conv4_1);
		RunConv_With_Bias(conv4_2_cfg,1,NULL,h_conv4_1,W_conv4_2,b_conv4_2,h_conv4_2);
		RunConv_With_Bias(conv4_3_cfg,1,NULL,h_conv4_2,W_conv4_3,b_conv4_3,h_conv4_3);
		RunPool(2,2,2,2,1,0,0,0,0,h_conv4_3,h_pool4);

		RunConv_With_Bias(conv5_1_cfg,1,NULL,h_pool4,W_conv5_1,b_conv5_1,h_conv5_1);
		RunConv_With_Bias(conv5_2_cfg,1,NULL,h_conv5_1,W_conv5_2,b_conv5_2,h_conv5_2);
		RunConv_With_Bias(conv5_3_cfg,1,NULL,h_conv5_2,W_conv5_3,b_conv5_3,h_conv5_3);
		RunPool(2,2,2,2,1,0,0,0,0,h_conv5_3,h_pool5);

		RunConv_Simplest(fc6_cfg,0,NULL,&h_pool5_part[0],W_fc6_part0,h_fc6_part0);				
		RunConv_Simplest(fc6_cfg,0,NULL,&h_pool5_part[1],W_fc6_part1,h_fc6_part1);				
		RunConv_Simplest(fc6_cfg,0,NULL,&h_pool5_part[2],W_fc6_part2,h_fc6_part2);				
		RunConv_With_Bias(fc6_cfg,0,NULL,&h_pool5_part[3],W_fc6_part3,b_fc6,h_fc6);				
		Element_Wise(0,NULL,0,h_fc6_part0,h_fc6,h_fc6);
		Element_Wise(0,NULL,0,h_fc6_part1,h_fc6,h_fc6);
		Element_Wise(1,NULL,0,h_fc6_part2,h_fc6,h_fc6);

		//RunConv_With_Bias(fc6_cfg,0,NULL,&h_pool5_part[0],W_fc6_part0,b_fc6,h_fc6);
		//RunConv_With_Element_Wise(fc6_cfg,0,NULL,0,&h_pool5_part[1],W_fc6_part1,h_fc6,h_fc6);
		//RunConv_With_Element_Wise(fc6_cfg,0,NULL,0,&h_pool5_part[2],W_fc6_part2,h_fc6,h_fc6);
		//RunConv_With_Element_Wise(fc6_cfg,1,NULL,0,&h_pool5_part[3],W_fc6_part3,h_fc6,h_fc6);

		RunConv_With_Bias(fc7_cfg,1,NULL,h_fc6,W_fc7,b_fc7,h_fc7);
		RunConv_With_Bias(fc8_cfg,0,NULL,h_fc7,W_fc8,b_fc8,out);

		end=clock();
		printf("Hardware run Finish, run time %ld us\n",end-start);
		
		short max1=-32768;short max2=-32768;short max3=-32768;
		int predict1,predict2,predict3;
		for(int j = 0; j<1000; j++)
		{
			short value=*Get_Element(out,0,0,j);
			if(value>max1) {
				max1=value;predict1=j;continue;}
			else
				if(value>max2) {
					max2=value;predict2=j;continue;}
				else
					if(value>max3) {
						max3=value;predict3=j;continue;}
		}

		printf("The predicted class is:\n");
		sprintf(cmd,"head -%d VGG16_DAT/imagenet_classes | tail -1",predict1+1);system(cmd);
		sprintf(cmd,"head -%d VGG16_DAT/imagenet_classes | tail -1",predict2+1);system(cmd);
		sprintf(cmd,"head -%d VGG16_DAT/imagenet_classes | tail -1",predict3+1);system(cmd);
		//sprintf(cmd,"display %s",str);system(cmd);//show the picture
		printf("\n\n");
	}
}

void LoadData(short *dat, char *filename, int size)
{
	FILE *fp=fopen(filename,"rb");
	if(fp==NULL)
	{
		printf("Can't open file: %s\n",filename);
		return;
	}
	size_t rd_size=fread(dat,1,size,fp);
	if(rd_size!=size)
		printf("Load Data from file Error\n");

	fclose(fp);
}
