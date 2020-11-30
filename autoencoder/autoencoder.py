import cv2
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow import keras
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential

# exit()
global batch_size,image_size

pos_train_dir = "./training_data_3/roi_pos/"
neg_train_dir = "./training_data_3/roi_neg/"

neg_list = os.listdir(neg_train_dir)
pos_list = os.listdir(pos_train_dir)

if len(pos_list)<len(neg_list):
	neg_list = neg_list[:len(pos_list)]
else:
	pos_list = pos_list[:len(neg_list)]

image_size = 400
batch_size = 256

f = os.listdir(neg_train_dir)[2]
img = cv2.imread(neg_train_dir+f,0)




def flatten_layer(layer):
	layer_shape = layer.get_shape()		
	num_features = layer_shape[1:4].num_elements()
	print(num_features)
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat,num_features


input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])



conv1 = Layers.Conv2D(3,1,1,padding="same",activation='relu',use_bias=True)(input_image_reshaped)
# conv1 = tf.layers.batch_normalization(conv1, training=True)
conv2 = Layers.Conv2D(5,3,2,padding="same",activation='relu',use_bias=True)(conv1)
# conv2 = tf.layers.batch_normalization(conv2, training=True)
conv3 = Layers.Conv2D(7,3,2,padding="same",activation='relu',use_bias=True)(conv2)
conv3 = tf.layers.batch_normalization(conv3, training=True)
conv4 = Layers.Conv2D(9,5,2,padding="same",activation='relu',use_bias=True)(conv3)
# conv4 = tf.layers.batch_normalization(conv4, training=True)
conv5 = Layers.Conv2D(12,5,2,padding="same",activation='relu',use_bias=True)(conv4)
# conv5 = tf.layers.batch_normalization(conv5, training=True)
conv6 = Layers.Conv2D(16,5,2,padding="same",activation='relu',use_bias=True)(conv5)
conv6 = tf.layers.batch_normalization(conv6, training=True)
# mean_vec = tf.reduce_mean(conv6,axis=[1,2])
# std_vec = tf.math.reduce_std(conv6,axis=[1,2])
# lflat = tf.concat(mean_vec,std_vec)
# conv5 = tf.nn.max_pool(value=conv5,ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1],padding='SAME')
lflat,num_features = flatten_layer(conv6)
# print(num_features)
fc1 = Layers.Dense(128,activation='relu')(lflat)
fc1 = tf.layers.batch_normalization(fc1, training=True)
fc2 = Layers.Dense(32,activation='relu')(fc1)
# fc2 = tf.layers.batch_normalization(fc2, training=True)
fc3 = Layers.Dense(625,activation='relu')(fc2)
fc3 = tf.layers.batch_normalization(fc3, training=True)
# fc4 = Layers.Dense(625,activation='relu')(fc3)

# fc5 = Layers.Dense(2048,activation='relu')(fc4)

reshape_fcout = tf.reshape(fc3, [-1, 25,25])
deconv1 = Layers.Conv2DTranspose(12,5,2,padding="same",activation='relu',use_bias=True)(conv5)
# deconv1 = tf.layers.batch_normalization(deconv1, training=True)
deconv2 = Layers.Conv2DTranspose(9,5,2,padding="same",activation='relu',use_bias=True)(deconv1)
# deconv2 = tf.layers.batch_normalization(deconv2, training=True)
deconv3 = Layers.Conv2DTranspose(7,3,1,padding="same",activation='relu',use_bias=True)(deconv2)
deconv3 = tf.layers.batch_normalization(deconv3, training=True)
deconv4 = Layers.Conv2DTranspose(5,5,2,padding="same",activation='relu',use_bias=True)(deconv3)
# deconv4 = tf.layers.batch_normalization(deconv4, training=True)
deconv5 = Layers.Conv2DTranspose(3,3,2,padding="same",activation='relu',use_bias=True)(deconv4)
# deconv5 = tf.layers.batch_normalization(deconv5, training=True)
deconv6 = Layers.Conv2DTranspose(1,3,1,padding="same",activation='relu',use_bias=True)(deconv5)
# deconv5 = Layers.Conv2DTranspose(1,3,2,padding="same",activation='relu')(deconv4)


error = tf.math.square(tf.subtract(deconv6,input_image_reshaped))
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

main_list = list(zip(pos_list,neg_list))
random.shuffle(main_list)
print(len(main_list))
training_set = main_list[:]
# testing_set = main_list[4000:]


img_array = []
target_array = []
prev_acc = 0
count = 0
tag_dir_pathfem = "./training_data_3/tag_pos/roi_posfem/"
tag_dir_pathpel = "./training_data_3/tag_pos/roi_pospel/"
tag_dir_pathgen = "./training_data_3/tag_pos/roi_posgen/"
fem_list = os.listdir(tag_dir_pathfem)
pel_list = os.listdir(tag_dir_pathpel)
gen_list = os.listdir(tag_dir_pathgen)


def getFlippedZoomImages(img):
	img = cv2.resize(img,(image_size,image_size))
	img_flip = cv2.flip(img,1)
	return img,img_flip

def getTagFeature(fem_tag_img):
	if np.amax(fem_tag_img) > 0.0:
		fem_tag_img = fem_tag_img/np.amax(fem_tag_img)
		bool_row = np.where(fem_tag_img==1.0)
		# print(bool_row)
		first_pt = [np.amin(bool_row[0])/image_size,np.amin(bool_row[1])/image_size]
		second_pt = [np.amax(bool_row[0])/image_size,np.amax(bool_row[1])/image_size]
		# print(first_pt)
		# print(second_pt)
		feature_array_fem = first_pt+second_pt
		return feature_array_fem
	else:
		return [0.0,0.0,0.0,0.0]


model_name = "AUTO_FC_32_lownorm"

try:
	saver.restore(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
except:
	sess.run(init)
	print("MODEL NOT FOUND")
	pass

for epoch in range(40):
	for x in training_set[:]:
		pos_img = cv2.imread(pos_train_dir+x[0],0)
		neg_img = cv2.imread(neg_train_dir+x[1],0)
		ps,psf = getFlippedZoomImages(pos_img)
		ns,nsf = getFlippedZoomImages(neg_img)
		img_array = img_array + [ps/255,psf/255,ns/255,nsf/255]
		if len(img_array)>=batch_size or training_set.index(x) == (len(training_set)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			# output = sess.run(deconv4,feed_dict={input_image:img_array})
			# cv2.imshow("img",img_array[0])
			# cv2.imshow("img2",output[0])
			# cv2.imshow("img3",img_array[2])
			# cv2.imshow("img4",output[2])
			# cv2.waitKey(0)
			# exit()
			c = sess.run(cost,feed_dict={input_image:img_array})
			print(c)
			for xx in range(5):
				sess.run(optimizer,feed_dict={input_image:img_array})
			# exit()
			saver.save(sess,'./tmp/models/train_accurate_models/'+model_name+'.ckpt')
			img_array = []
		# break
	
	# break







