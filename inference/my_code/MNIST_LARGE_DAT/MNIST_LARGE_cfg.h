#ifndef __MNIST_LARGE_CFG__
#define __MNIST_LARGE_CFG__

struct Conv_Cfg conv1_cfg={1,28,32,4,5,5,1,1,2,2,4,0,28,28,28,51200,1,28,36,28,28,32,32,1,32,32};
struct Conv_Cfg conv2_cfg={32,14,64,4,5,5,1,1,2,2,1,0,14,14,14,102400,1,14,18,14,14,14,14,1,64,64};
struct Conv_Cfg fc1_cfg={64,7,1024,6,7,7,1,1,0,0,1,1,1,1,14,6422528,1,7,18,7,1,12,12,32,32,32};
struct Conv_Cfg fc2_cfg={1024,1,10,0,1,1,1,1,0,0,1,0,1,1,32,20480,1,1,8,1,1,8,8,1,10,10};
#define PTR_IMG 14
#define PTR_W_CONV1 14
#define PTR_B_CONV1 15
#define PTR_H_CONV1_BEFORE_LUT 11
#define PTR_H_CONV1 15
#define PTR_H_POOL1 15
#define PTR_W_CONV2 15
#define PTR_B_CONV2 15
#define PTR_H_CONV2_BEFORE_LUT 10
#define PTR_H_CONV2 14
#define PTR_H_POOL2 14
#define PTR_W_FC1 15
#define PTR_B_FC1 15
#define PTR_H_FC1_BEFORE_LUT 10
#define PTR_H_FC1 14
#define PTR_W_FC2 15
#define PTR_B_FC2 15
#define PTR_H_FC2 10

#endif
