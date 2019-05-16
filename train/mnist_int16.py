# -*- coding: utf-8 -*-
import input_data
import tensorflow as tf
import numpy as np
from tf_fix import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'): 
	x = tf.placeholder("float", shape=[None, 784])
	y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1);
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding='SAME')

#First Convolutional Layer
with tf.name_scope('1st_CNN'): 
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1_before_lut = conv2d(x_image, W_conv1)+b_conv1
	h_conv1 = tf.nn.sigmoid( h_conv1_before_lut )
	h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional Layer
with tf.name_scope('2rd_CNN'): 
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2_before_lut = conv2d(h_pool1, W_conv2)+b_conv2
	h_conv2 = tf.nn.sigmoid( h_conv2_before_lut )
	h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer
with tf.name_scope('Densely_NN'): 
	W_fc1 = weight_variable([ 7* 7* 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1_before_lut = tf.matmul(h_pool2_flat , W_fc1)+b_fc1
	h_fc1=tf.nn.sigmoid( h_fc1_before_lut )

#Dropout
with tf.name_scope('Dropout'):
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
with tf.name_scope('Softmax'):
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_conv=tf.nn.softmax(h_fc2)

with tf.name_scope('Loss'):
	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

with tf.name_scope('Train'):
	train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
	#train_step = tf.train.AdamOptimizer(5e-5).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_conv ,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float"))

#merged = tf.merge_all_summaries();
#writer = tf.train.SummaryWriter("logs/",sess.graph) 

tf.initialize_all_variables().run()

for i in range(10000):#000):
	batch = mnist.train.next_batch(200);
	if i%200 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0});
		print("step %d, training accuracy %g"%(i, train_accuracy));
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5});

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

f_cfg = open('./record/MNIST_LARGE_cfg.h', 'w')
f_cfg.write("#ifndef __MNIST_LARGE_CFG__\n");
f_cfg.write("#define __MNIST_LARGE_CFG__\n\n");

Record_Conv_Cfg(28,28,1,32,5,5,1,1,2,2,2,2,"conv1",f_cfg);
Record_Conv_Cfg(14,14,32,64,5,5,1,1,2,2,2,2,"conv2",f_cfg);
Record_Conv_Cfg(7,7,64,1024,7,7,1,1,0,0,0,0,"fc1",f_cfg);
Record_Conv_Cfg(1,1,1024,10,1,1,1,1,0,0,0,0,"fc2",f_cfg);
	
Get_Feature_Fraction_Part(x,"img",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
Record_Weight(W_conv1,"W_conv1",f_cfg)
Record_Bias(b_conv1,"b_conv1",f_cfg)
Get_Feature_Fraction_Part(h_conv1_before_lut,"h_conv1_before_lut",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
Get_Feature_Fraction_Part(h_conv1,"h_conv1",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
Get_Feature_Fraction_Part(h_pool1,"h_pool1",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
	
Record_Weight(W_conv2,"W_conv2",f_cfg)
Record_Bias(b_conv2,"b_conv2",f_cfg)
Get_Feature_Fraction_Part(h_conv2_before_lut,"h_conv2_before_lut",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
Get_Feature_Fraction_Part(h_conv2,"h_conv2",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
Get_Feature_Fraction_Part(h_pool2,"h_pool2",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
	
Record_Weight(tf.reshape(W_fc1,[7,7,64,1024]),"W_fc1",f_cfg)
Record_Bias(b_fc1,"b_fc1",f_cfg)
Get_Feature_Fraction_Part(h_fc1_before_lut,"h_fc1_before_lut",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
Get_Feature_Fraction_Part(h_fc1,"h_fc1",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)
	
Record_Weight(tf.reshape(W_fc2,[1,1,1024,10]),"W_fc2",f_cfg)
Record_Bias(b_fc2,"b_fc2",f_cfg)
Get_Feature_Fraction_Part(h_fc2,"h_fc2",{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},f_cfg)		

f_cfg.write("\n#endif\n");
f_cfg.close();

sess.close()
