#include "FPGA_DDR.h"
#include "vpu_basic.h"

int main(void)
{
	VPU_Init();
	Mapped_Feature *input=Malloc_Feature(28,28,3,10,10,-1,-1);
	if(input==NULL)
		{printf("Malloc_Feature fail\n");return 0;}
	else
		printf("Malloc_Feature success\n");

	Debug_mcb();
	printf("pay_load=%lx\n",(uint64_t)input->payload);

	short in[28][28][3];
	short out[28][28][3];

	for(int i=0;i<28;i++)
		for(int j=0;j<28;j++)
			for(int k=0;k<3;k++)
				in[i][j][k]=i+j+k;

	Map_Feature(in[0][0],input);
	DeMap_Feature(input,out[0][0]);
	
	for(int i=0;i<28;i++)
		for(int j=0;j<28;j++)
			for(int k=0;k<3;k++)
				printf("out[%d][%d][%d]=%d\n",i,j,k,out[i][j][k]);
	
	Free_Feature(input);
	printf("OK\n");
	return 0;
}
