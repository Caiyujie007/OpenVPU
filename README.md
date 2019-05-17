# OpenVPU
This is an open CNN accelerator for everyone to use.

The flow contains two steps: train and inference.

In the train phase, we need to train the network, and using "tf_fix.py" to quantilize the network.
We also need to export the trained weights into files in this phase.
A script named "mnist_int16.py" which trained with MNIST database is provided. This scripts shows how to do quantilization and weight exporting.

Before the inference phase, we first need a ZCU104 board with a DDR4 bar installed. The type of the DDR4 chip must be MTA8ATF51264HZ-2G1A1 (4G 2133). The DDR4 bar can be found on 淘宝 or 京东 by searching keyword 'MTA8ATF51264HZ'. Then we need to build the PYNQ enviroment on ZCU104 board. Then we can go to the inference phase.

First, copy this database under any folder into ZCU104 filesystem, then execute the load_bitfile.py under <inference> to download the bitfile.
  
There are several DEMOS provided in <my_code> folder: Malloc, CONV, POOL, ELEMENTWISE, NONLINEAR, MNIST_LARGE, VGG16.
The first 5 DEMOS do not need datasets, so you can compile them with 'make [DEMONAME]', like make POOL. After the executable is generated, you can execute it.
