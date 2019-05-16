# OpenVPU
This is an open CNN accelerator for everyone to use

The flow contains two steps: train and inference.
In the train phase, we need to train the network, and using "tf_fix.py" to quantilize the network.
We also need to export the trained weights into files in this phase.
A script named "mnist_int16.py" which trained with MNIST database is provided. This scripts shows how to do quantilization and weight exporting.
