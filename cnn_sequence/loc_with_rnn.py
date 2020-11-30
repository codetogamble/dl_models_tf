import cv2
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow import keras
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import constraints
import copy

# exit()
global batch_size,image_size

image_size = 400
batch_size = 60

def flatten_layer(layer):
	layer_shape = layer.get_shape()		
	num_features = layer_shape[2:5].num_elements()
	print(num_features)
	layer_flat = tf.reshape(layer, [-1, 60,num_features])
	return layer_flat

def getActivDiff(layer):
	x = tf.expand_dims(layer, 1)
	return x
input_image = tf.placeholder(tf.float32,shape=[None,batch_size,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1,batch_size, image_size, image_size, 1])
target_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="target_image")
target_image_reshaped = tf.reshape(target_image, [-1, image_size, image_size, 1])

bias_constr = constraints.MinMaxNorm(min_value=-1.0, max_value=0.0, rate=1.0, axis=0)
kernel_constr = constraints.MinMaxNorm(min_value=0.0, max_value=2.0, rate=1.0, axis=[0,1,2])

conv1 = Layers.TimeDistributed(Layers.Conv2D(3,1,1,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr))(input_image_reshaped)

# conv1 = tf.layers.batch_normalization(conv1, training=True)
conv2 = Layers.TimeDistributed(Layers.Conv2D(5,3,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr))(conv1)
# conv2 = tf.layers.batch_normalization(conv2, training=True)
conv3 = Layers.TimeDistributed(Layers.Conv2D(7,3,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr))(conv2)
conv3 = tf.layers.batch_normalization(conv3, training=False)
conv4 = Layers.TimeDistributed(Layers.Conv2D(9,5,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr))(conv3)
# conv4 = tf.layers.batch_normalization(conv4, training=True)
conv5 = Layers.TimeDistributed(Layers.Conv2D(12,5,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr))(conv4)
# conv5 = tf.layers.batch_normalization(conv5, training=True)
conv6 = Layers.TimeDistributed(Layers.Conv2D(16,5,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr))(conv5)
conv6 = tf.layers.batch_normalization(conv6, training=False)
# mean_vec = tf.reduce_mean(conv6,axis=[1,2])
# std_vec = tf.math.reduce_std(conv6,axis=[1,2])
# lflat = tf.concat(mean_vec,std_vec)
# conv5 = tf.nn.max_pool(value=conv5,ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1],padding='SAME')
lflat = flatten_layer(conv6)

# print(num_features)
fc1 = Layers.TimeDistributed(Layers.Dense(512,activation='relu'))(lflat)
# fc1 = tf.layers.batch_normalization(fc1, training=True)
fc2 = Layers.TimeDistributed(Layers.Dense(64,activation='relu'))(fc1)
fc2 = tf.layers.batch_normalization(fc2, training=False)
# fc2 = tf.reshape(fc2, [1, batch_size, 32])
# fc2 = tf.reshape(fc2, [-1, batch_size, 64])
lstm_cell_1 = Layers.SimpleRNN(32,activation='relu',use_bias = True,return_sequences=False,go_backwards=True)(fc2)
# lstm_3 = Layers.RepeatVector(batch_size)(lstm_cell_2)
# lstm_3 = tf.reshape(lstm_cell_2,(1,batch_size,8))
# lstm_4 = Layers.LSTM(128,activation='relu',input_shape=(batch_size,32),return_sequences=False)(lstm_3)
# lstm_5 = Layers.LSTM(64, activation='relu', return_sequences=True)(lstm_4)
# Decoder
# fc3 = Layers.TimeDistributed(Layers.Dense(625,activation='relu'))(lstm_4)
fc3 = Layers.Dense(64,activation='relu')(lstm_cell_1)
# fc3 = tf.layers.batch_normalization(fc3, training=True)
fc4 = Layers.Dense(625,activation='relu')(fc3)

# fc5 = Layers.Dense(2048,activation='relu')(fc4)

reshape_fcout = tf.reshape(fc4, [-1,25,25,1])
layer1 = Layers.Conv2DTranspose(12,5,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr)
deconv1 = layer1(reshape_fcout)
# deconv1 = tf.layers.batch_normalization(deconv1, training=True)
deconv2 = Layers.Conv2DTranspose(9,5,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr)(deconv1)
# deconv2 = tf.layers.batch_normalization(deconv2, training=True)
deconv3 = Layers.Conv2DTranspose(7,3,1,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr)(deconv2)
deconv3 = tf.layers.batch_normalization(deconv3, training=False)
deconv4 = Layers.Conv2DTranspose(5,5,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr)(deconv3)
# deconv4 = tf.layers.batch_normalization(deconv4, training=True)
deconv5 = Layers.Conv2DTranspose(3,3,2,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr)(deconv4)
# deconv5 = tf.layers.batch_normalization(deconv5, training=True)
deconv6 = Layers.Conv2DTranspose(1,3,1,padding="same",activation='relu',use_bias=True,kernel_constraint=kernel_constr,bias_constraint=bias_constr)(deconv5)
# deconv5 = Layers.Conv2DTranspose(1,3,2,padding="same",activation='relu')(deconv4)
# xx = tf.constant([[1],[2],[3],[4]], dtype=tf.float32)
# xx = layer1.get_weights()
# yy = tf.constant([[0, 1],[2, 3],[4,5]], dtype=tf.float32)
# diffLayer = getActivDiff(xx)

# or more succinctly 
# zz = tf.reshape(xx[None] + yy[:, None], [-1, 2])

error = tf.math.square(tf.subtract(deconv6,target_image_reshaped))
cost = tf.reduce_mean(error)

# conv1_out = tf.concat([conv1_1,conv1_3,conv1_5],axis=3)
# tf.nn.max_pool(value=conv1_out,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
# fc1 = Layers.Dense(64,activation='relu')(lflat)
# fc2 = Layers.Dense(2,activation='relu')(fc1)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init)




model_name = "LOC_RNN"

try:
	saver.restore(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
except:
	sess.run(init)
	print("MODEL NOT FOUND")
	pass


img_array = []
img_array2 = []
img_batch_array = []
img_batch_array2 = []
img_batch_targets = []
img_batch_targets2 = []
test_folder = "./video_data/"

frames_no = 30
sample_n = 6

count = 0
for f in os.listdir(test_folder)[2:]:
	vidreader = cv2.VideoCapture(test_folder+f)
	# vidreader.set(cv2.CAP_PROP_POS_FRAMES,4000)
	tt = True
	while(tt):
		tt,ff = vidreader.read()
		frame_no = vidreader.get(cv2.CAP_PROP_POS_FRAMES)
		print(frame_no)
		exit()
		# gray = cv2.cvtColor(ff,cv2.COLOR_BGR2GRAY)
		# rsz = cv2.resize(gray,(400,400))
		# rsz = rsz/255
		if tt == False:
			break
		gray2 = cv2.cvtColor(ff[60:],cv2.COLOR_BGR2GRAY)
		main_image_shape = gray2.shape
		gray = cv2.resize(gray2,(int(gray2.shape[1]/16),int(gray2.shape[0]/16)))
		ret,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
		sum_mid_vert = np.sum(thresh[:,int(gray.shape[1]/2)])
		sum_mid_hori = np.sum(thresh[int(gray.shape[0]/2),:])
		# print(sum_mid_hori)
		wsz = 0
		splitimg_bool = False
		rightsplit_bool = False
		if sum_mid_vert < 2000 and sum_mid_hori > 5000:
			# print("split")
			splitimg_bool = True
			out2 = gray2[:,int(gray2.shape[1]/2):]
			out = gray2[:,:int(gray2.shape[1]/2)]
				# print(out.shape)
			wsz = out.shape[1]
			# print(out.shape)
			vert_gap = int((out.shape[0]-wsz)/2)
			hori_gap = int((out.shape[1]-wsz)/2)
			out = out[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
			out2 = out2[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
			out = cv2.resize(out,(400,400))/255
			out2 = cv2.resize(out2,(400,400))/255
			img_array.append(out)
			img_array2.append(out2)

		else:
			if len(img_array2)>0:
				img_array = []
				img_array2 = []
			# print("single")
			out = gray2
			# print(out.shape)
			wsz = 600
			vert_gap = int((out.shape[0]-wsz)/2)
			hori_gap = int((out.shape[1]-wsz)/2)
			out = out[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
			out = cv2.resize(out,(400,400))
			out = out/255
			img_array.append(out)
		
		if len(img_array) == batch_size+1 and len(img_array2) == batch_size+1:
			# c = sess.run(deconv6,feed_dict={input_image:img_array})
			# print(c.shape)
			# exit()
			img_batch_array.append(img_array[:-1])
			img_batch_array2.append(img_array2[:-1])
			img_batch_targets.append(img_array[-1])
			img_batch_targets2.append(img_array2[-1])
			img_array = img_array[10:]
			img_array2 = img_array2[10:]

		
			# break
		elif len(img_array) == batch_size+1:
			# c = sess.run(deconv6,feed_dict={input_image:img_array[:-1],target_image:img_array[-1:]})
			# print(c.shape)
			# print(c)
			# exit()
			img_batch_array.append(img_array[:-1])
			img_batch_targets.append(img_array[-1])
			img_array = img_array[10:]
		
		if len(img_batch_array) == sample_n and len(img_batch_array2) == sample_n:
			c = sess.run(cost,feed_dict={input_image:img_batch_array,target_image:img_batch_targets})
			print(c)
			c = sess.run(cost,feed_dict={input_image:img_batch_array2,target_image:img_batch_targets2})
			print(c)
			for ii in range(2):
				sess.run(optimizer,feed_dict={input_image:img_batch_array,target_image:img_batch_targets})
				sess.run(optimizer,feed_dict={input_image:img_batch_array2,target_image:img_batch_targets2})
			# saver.save(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
			img_batch_array = []
			img_batch_targets = []
			img_batch_array2 = []
			img_batch_targets2 = []
			count += 1
		
		elif len(img_batch_array) == sample_n:
			# c = sess.run(target_image,feed_dict={input_image:img_batch_array,target_image:img_batch_targets})
			# print(c.shape)
			# # print(c)
			# exit()
			c = sess.run(cost,feed_dict={input_image:img_batch_array,target_image:img_batch_targets})
			print(c)
			for ii in range(2):
				sess.run(optimizer,feed_dict={input_image:img_batch_array,target_image:img_batch_targets})
			# saver.save(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
			img_batch_array = []
			img_batch_targets = []
			count += 1

		# if count % 9 == 1:
		# 	saver.save(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
			# break
		# else:
		# 	pass
	vidreader.release()
	# exit()




