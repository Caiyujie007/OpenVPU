#ifndef RDMNIST_H
#define RDMNIST_H

#include <iostream>
#include <stdio.h>
#include <math.h>
#include "MNIST_LARGE_cfg.h"

using namespace std;

template<typename T>
int rdMNISTs28_train(int begin_num, int end_num, T data_o[][28][28], int _label[])
{
	FILE *fp;
	int err;
	if (begin_num >= end_num)
	{
		cout << "Wrong num!";
		return -1;
	}
	char imagefile[] = "/dat/MNIST/train-images.idx3-ubyte";
	char labelfile[] = "/dat/MNIST/train-labels.idx1-ubyte";
	fp = fopen(imagefile, "rb");
	if (fp == NULL)
	{
		cout << "Can't Open " << imagefile  << "! Error : " << err << endl;
		return -1;
	}
	fseek(fp, begin_num * 784 + 16, SEEK_SET);
	size_t rderr;
	unsigned char picture[28][28] = {0};
	for(int i = 0; i < end_num - begin_num; i++)
		for (int j = 0; j < 28; j++)
		{
			rderr = fread(picture[j], sizeof(bool), 28, fp);
			for (int k = 0; k < 28; k++)
			{
				data_o[i][j][k] = (T)(picture[j][k] / 255.0);
			}
		}
	fclose(fp);
	fp = fopen(labelfile, "rb");
	if (fp == NULL)
	{
		cout << "Can't Open " << labelfile << endl;
		return -1;
	}
	fseek(fp, begin_num + 8, SEEK_SET);
	unsigned char rtn;
	for(int i = 0; i < end_num - begin_num ;i ++)
	{
		rderr = fread(&rtn, sizeof(bool), 1, fp);
		_label[i] = (int)rtn;
	}
	fclose(fp);
	return 0;
}

template<typename T>
int rdMNISTs28_test(int begin_num, int end_num, T data_o[][28][28], int _label[])
{
	FILE *fp;
	if (begin_num >= end_num)
	{
		cout << "Wrong num!";
		return -1;
	}
	char imagefile[] = "/dat/MNIST/t10k-images.idx3-ubyte";
	char labelfile[] = "/dat/MNIST/t10k-labels.idx1-ubyte";
	fp = fopen(imagefile, "rb");
	if (fp == NULL)
	{
		cout << "Can't Open " << imagefile << endl;
		return -1;
	}
	fseek(fp, begin_num * 784 + 16, SEEK_SET);
	size_t rderr;
	unsigned char picture[28][28] = {0};
	for(int i = 0; i < end_num - begin_num; i++)
		for (int j = 0; j < 28; j++)
		{
			rderr = fread(picture[j], sizeof(bool), 28, fp);
			for (int k = 0; k < 28; k++)
			{
				data_o[i][j][k] = (T)((picture[j][k] / 255.0)*pow(2,PTR_IMG));
			}
		}
	fclose(fp);
	fp = fopen(labelfile, "rb");
	if (fp == NULL)
	{
		cout << "Can't Open " << labelfile << endl;
		return -1;
	}
	fseek(fp, begin_num + 8, SEEK_SET);
	unsigned char rtn;
	for(int i = 0; i < end_num - begin_num ;i ++)
	{
		rderr = fread(&rtn, sizeof(bool), 1, fp);
		_label[i] = (int)rtn;
	}
	fclose(fp);
	return 0;
}

#endif
