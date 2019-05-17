# OpenVPU
This is an open CNN accelerator for everyone to use.

The flow contains two steps: train and inference.

In the train phase, we need to train the network, and using "tf_fix.py" to quantilize the network.
We also need to export the trained weights into files in this phase.
A script named "mnist_int16.py" which trained with MNIST database is provided. This scripts shows how to do quantilization and weight exporting. The train phase can be done on any PC with tensorflow installed, so if you don't have a ZCU104 board, you can try the train process on PC and learn how to quantilize the network and exporting the trained weights.

Before the inference phase, we first need a ZCU104 board with a DDR4 bar installed. The type of the DDR4 chip must be MTA8ATF51264HZ-2G1A1 (4G 2133). The DDR4 bar can be found on 淘宝 or 京东 by searching keyword 'MTA8ATF51264HZ'. Then we need to build the PYNQ enviroment on ZCU104 board. Then we can go to the inference phase.

First, copy this database under any folder into ZCU104 filesystem, then execute the load_bitfile.py under <inference> to download the bitfile. The clock of this bitfile is running at 100MHz, but actually it can run at 214MHz. So if you want more performance of CNN acceleration, please contact me according https://space.bilibili.com/10455971. I can give you the 214M version bitfile if you made some interresting DEMOs using this CNN accelerator.
  
There are several DEMOS provided in <my_code> folder: Malloc, CONV, POOL, ELEMENTWISE, NONLINEAR, MNIST_LARGE, VGG16.
The first 5 DEMOS do not need pre-trained weights and datasets, so you can compile them with 'make [DEMONAME]', like make POOL. After the executable is generated, you can execute it.

The DEMOS MNIST_LARGE and VGG16 need pre-trained weights and datasets, you need to download them, then extracting these files into <my_code> folder.

The link of pre-trained weights of MNIST_LARGE is https://pan.baidu.com/s/1WzQhSUXlS8oT42aVDaHEbw 提取码：5zwz 

The link of pre-trained weights of VGG16 is https://pan.baidu.com/s/1l06nfVMn3UHtquIG0L4DRw 提取码：b0jf 

The datasets of DEMO MNIST_LARGE and VGG16 are also needed to download, link is https://pan.baidu.com/s/1ehIBh0i87-rFo4MBhYjXfg 
提取码：kph9. We need to extract it into to the root folder of ZCU104, as /dat/ILSVR2012 and /dat/MNIST.

There is a python script (/dat/ILSVR2012/dataset_valid_bin/JPG2bin.py) to translate any size JPEG pictures into 224x224x3 for VGG16 network, the translated picture has a suffix as '.bin'. The VGG16 demo will read in the .bin pictures according to the /dat/ILSVR2012/dataset_valid_bin/filelist, and run the inference.
