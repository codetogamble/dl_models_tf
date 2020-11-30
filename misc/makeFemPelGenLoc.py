import tensorflow as tf
import numpy as np
import cv2
import os
import random
import csv
import sys

neg_data_dir = "./training_data_2/roi_neg/"
neg_data = os.listdir(neg_data_dir)

pos_data_dir = "./training_data_2/roi_pos/"
pos_data = os.listdir(pos_data_dir)

fem_dir = "./training_data_2/tag_pos/roi_posfem/"
fem_files = os.listdir(fem_dir)

pel_dir = "./training_data_2/tag_pos/roi_pospel/"
pel_files = os.listdir(pel_dir)

gen_dir = "./training_data_2/tag_pos/roi_posgen/"
gen_files = os.listdir(gen_dir)


def flatten_layer(layer):
	layer_shape = layer.get_shape()		
	num_features = layer_shape[1:4].num_elements()
	print(num_features)
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat,num_features
def fcNode(input_layer,weights,biases,use_relu=True):
	layer = tf.matmul(input_layer, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
		return layer
	
fil_size_1 = 4
fil_size_2 = 4
fil_size_3 = 4


nchannels1 = 1
nchannels2 = 2
nchannels3 = 2
nchannels4 = 2
numfeat = 5000

nfilters1 = 2
nfilters2 = 2
nfilters3 = 2
nfilters4 = 512
nfilters5 = 128
nfilters6 = 15

shape1 = [fil_size_1,fil_size_1,nchannels1,nfilters1]
shape2 = [fil_size_2,fil_size_2,nchannels2,nfilters2]
shape3 = [fil_size_3,fil_size_3,nchannels3,nfilters3]
shape4 = [numfeat,nfilters4]
shape5 = [nfilters4,nfilters5]
shape6 = [nfilters5,nfilters6]



w1 = tf.get_variable("weightss_1", shape1,initializer=tf.random_normal_initializer(stddev=0.07))
w2 = tf.get_variable("weightss_2", shape2,initializer=tf.random_normal_initializer(stddev=0.07))
w3 = tf.get_variable("weightss_3", shape3,initializer=tf.random_normal_initializer(stddev=0.07))
w4 = tf.get_variable("weightss_4", shape4,initializer=tf.random_normal_initializer(stddev=0.07))
w5 = tf.get_variable("weightss_5", shape5,initializer=tf.random_normal_initializer(stddev=0.07))
w6 = tf.get_variable("weightss_6", shape6,initializer=tf.random_normal_initializer(stddev=0.07))

b1 = tf.get_variable("bias_1", [nfilters1],initializer=tf.constant_initializer(0.05))
b2 = tf.get_variable("bias_2", [nfilters2],initializer=tf.constant_initializer(0.05))
b3 = tf.get_variable("bias_3", [nfilters3],initializer=tf.constant_initializer(0.05))
b4 = tf.get_variable("bias_4", [nfilters4],initializer=tf.constant_initializer(0.05))
b5 = tf.get_variable("bias_5", [nfilters5],initializer=tf.constant_initializer(0.05))
b6 = tf.get_variable("bias_6", [nfilters6],initializer=tf.constant_initializer(0.05))


image_size = 400

def applyCNN(image):
	l1 = tf.nn.conv2d(input=image,filter=w1,strides=[1, 1, 1, 1],padding='SAME')
	l1 += b1
	l1 = tf.nn.max_pool(value=l1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
	l1 = tf.nn.relu(l1)
	l2 = tf.nn.conv2d(input=l1,filter=w2,strides=[1, 1, 1, 1],padding='SAME')
	l2 += b2
	l2 = tf.nn.max_pool(value=l2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
	l2 = tf.nn.relu(l2)
	l3 = tf.nn.conv2d(input=l2,filter=w3,strides=[1, 1, 1, 1],padding='SAME')
	l3 += b3
	l3 = tf.nn.max_pool(value=l3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
	l3 = tf.nn.relu(l3)
	lflat,numfeat = flatten_layer(l3)
	layer_fc1 = fcNode(lflat,w4,b4)
	layer_fc2 = fcNode(layer_fc1,w5,b5)
	layer_fc3 = fcNode(layer_fc1,w5,b5)
	return layer_fc3

image_size = 400
input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
# input_image_fempts = tf.placeholder(tf.float32,shape=[None,3,4],name="input_images_tagged_array")
x_image = tf.reshape(input_image, [-1, image_size, image_size, 1])
result = applyCNN(x_image)
# conf,bbox = tf.split(result,[3,12],1)
# error = tf.math.square(bbox - input_image_fempts)
# cost = tf.reduce_mean(error) 

# image_patches = tf.extract_image_patches(x_image,[1,window_size,window_size,1],[1,stride,stride,1],[1,1,1,1],padding='SAME')
# image_patches_tag = tf.extract_image_patches(x_image_tagged_guide,[1,window_size,window_size,1],[1,stride,stride,1],[1,1,1,1],padding='SAME')

# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
# sess.run(tf.variables_initializer(all_variables))

output = sess.run(list_files_neg)
print(output)


		