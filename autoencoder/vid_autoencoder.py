import cv2
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow import keras
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
import math

# exit()

image_size = 400

def flatten_layer(layer):
	layer_shape = layer.get_shape()		
	num_features = layer_shape[1:4].num_elements()
	print(num_features)
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat,num_features


beta = 0.003
beta2 = 0.003
steps_per_epoch = 1
input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])
# input_image_reshaped = tf.image.per_image_standardization(input_image_reshaped)

conv1 = Layers.Conv2D(4,1,1,padding="same",activation='relu',use_bias=True)(input_image_reshaped)
conv2 = Layers.Conv2D(8,3,2,padding="same",activation='relu',use_bias=True)(conv1)
conv3 = Layers.Conv2D(16,5,2,padding="same",activation='relu',use_bias=True)(conv2)
conv4 = Layers.Conv2D(32,5,2,padding="same",activation='relu',use_bias=True)(conv3)
conv5 = Layers.Conv2D(64,7,2,padding="same",activation='relu',use_bias=True)(conv4)
# conv5 = tf.nn.max_pool(value=conv5,ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1],padding='SAME')
# lflat,num_features = flatten_layer(conv5)
# print(num_features)
# fc1 = Layers.Dense(1024,activation='relu')(lflat)
# fc2 = Layers.Dense(256,activation='relu')(fc1)
# fc3 = Layers.Dense(32,activation='relu')(fc2)
# fc4 = Layers.Dense(512,activation='relu')(fc3)
# fc5 = Layers.Dense(2048,activation='relu')(fc4)

# reshape_fcout = tf.reshape(fc5, [-1, num_features])
deconv1 = Layers.Conv2DTranspose(32,7,2,padding="same",activation='relu',use_bias=True)(conv5)
deconv2 = Layers.Conv2DTranspose(16,5,2,padding="same",activation='relu',use_bias=True)(deconv1)
deconv3 = Layers.Conv2DTranspose(8,5,2,padding="same",activation='relu',use_bias=True)(deconv2)
deconv4 = Layers.Conv2DTranspose(4,3,2,padding="same",activation='relu',use_bias=True)(deconv3)
deconv5 = Layers.Conv2DTranspose(1,3,1,padding="same",activation='relu',use_bias=True)(deconv4)
# deconv5 = Layers.Conv2DTranspose(1,3,2,padding="same",activation='relu')(deconv4)


error = tf.math.square(tf.subtract(deconv5,input_image_reshaped))
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


model_name = "autoencoder_large3"

try:
	saver.restore(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
except:
	sess.run(init)
	print("MODEL NOT FOUND")
	pass
# output = sess.run(deconv5,feed_dict={input_image:img_array})


def getMapImage(conv_out):
	wsz = conv_out.shape[0]
	no_filters = conv_out.shape[2]
	print(wsz)
	print(no_filters)
	print("++++++++====")
	rows = int(no_filters/8)
	cols = int(no_filters/rows)
	print(rows*(wsz))
	print(cols*(wsz))
	blank_grid = np.zeros((cols*(wsz),rows*(wsz)),dtype=np.float32)
	for i in range(no_filters):
		fil = conv_out[:,:,i]
		rem = i%rows
		div = int(i/rows)
		print(rem)
		print(div)
		blank_grid[div*wsz:div*wsz+wsz,rem*wsz:rem*wsz+wsz] = fil
		# blank_grid[]
	return blank_grid

test_folder = "./test_vids/"
for f in os.listdir(test_folder):
	vidreader = cv2.VideoCapture(test_folder+f)
	vidreader.set(cv2.CAP_PROP_POS_FRAMES,4000)
	tt = True
	while(tt):
		tt,ff = vidreader.read()
		# gray = cv2.cvtColor(ff,cv2.COLOR_BGR2GRAY)
		# rsz = cv2.resize(gray,(400,400))
		# rsz = rsz/255
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
			print("split")
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
			out = cv2.resize(out,(400,400))
			out2 = cv2.resize(out2,(400,400))
			output = sess.run(conv5,feed_dict={input_image:[out/255,out2/255]})
			outleft = output[0]
			outright = output[1]
			outimg = getMapImage(outleft)
			outimg2 = getMapImage(outright)
			print(outimg.shape)
			cv2.imshow("img",out)
			cv2.imshow("img2",out2)
			cv2.imshow("outimg",outimg)
			cv2.imshow("outimg2",outimg2)
			# cv2.imshow("output2",out2)
			qq = cv2.waitKey(10)
		else:
			# print("single")
			out = gray2
			# print(out.shape)
			wsz = 600
			vert_gap = int((out.shape[0]-wsz)/2)
			hori_gap = int((out.shape[1]-wsz)/2)
			out = out[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
			out = cv2.resize(out,(400,400))
			output = sess.run(conv5,feed_dict={input_image:[out/255]})
			outleft = output[0]
			outimg = getMapImage(outleft)
			print(outimg.shape)
			cv2.imshow("img",out)
			cv2.imshow("outimg",outimg)
			cv2.imshow("img",out)
			qq = cv2.waitKey(10)
		if qq == ord('q'):
			vidreader.release()
			break
	# exit()




	
	# output = sess.run(deconv5,feed_dict={input_image:[rsz]})
	# cv2.imshow("img",gray)
	# cv2.imshow("img2",output[0])
	# cv2.waitKey(0)
	# output = sess.run(conv5,feed_dict={input_image:[rsz]})
	# print(output.shape)
	# conv5_filtermap = np.zeros((),dtype=np.float32)
	# for i in range(output.shape[3]):
	# 	actv = output[0,:,:,i]
	# 	cv2.imshow("img",gray)
	# 	cv2.imshow("img2",cv2.resize(actv,(400,400)))
	# 	cv2.waitKey(0)
	# cv2.destroyAllWindows()
	

