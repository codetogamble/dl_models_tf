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



# beta = 0.003
# beta2 = 0.003
# steps_per_epoch = 1
input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])
input_image_reshaped = tf.image.per_image_standardization(input_image_reshaped)
input_target = tf.placeholder(tf.float32,shape=[None,50,50,3],name="input_images_array")

conv1 = Layers.Conv2D(5,1,1,padding="same",activation='relu')(input_image_reshaped)
conv2 = Layers.Conv2D(3,3,2,padding="same",activation='relu')(conv1)
conv3 = Layers.Conv2D(5,5,2,padding="same",activation='relu')(conv2)
conv4 = Layers.Conv2D(7,5,1,padding="same",activation='relu')(conv3)
conv5 = Layers.Conv2D(3,7,2,padding="same",activation='relu')(conv4)
error = tf.math.square(tf.subtract(conv5,input_target))
cost = tf.reduce_mean(error)
# lflat,num_features = flatten_layer(conv4)
# print(num_features)
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
training_set = main_list[:4512]
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



model_name = "conv_loc"

try:
	saver.restore(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
except:
	sess.run(init)
	print("MODEL NOT FOUND")
	pass

for epoch in range(10):
	for x in training_set[:]:


		pos_img = cv2.imread(pos_train_dir+x[0],0)
		neg_img = cv2.imread(neg_train_dir+x[1],0)
		ps,psf = getFlippedZoomImages(pos_img)
		ns,nsf = getFlippedZoomImages(neg_img)

		fem_loc = np.zeros((50,50),dtype=np.float32)
		pel_loc = np.zeros((50,50),dtype=np.float32)
		gen_loc = np.zeros((50,50),dtype=np.float32)
		fem_locf = np.zeros((50,50),dtype=np.float32)
		pel_locf = np.zeros((50,50),dtype=np.float32)
		gen_locf = np.zeros((50,50),dtype=np.float32)
		# print(x[0])
		if x[0] in fem_list:
			# print("FOUND")
			fem_tag_img = cv2.imread(tag_dir_pathfem+x[0],0)
			ffpts,ffptsf = getFlippedZoomImages(fem_tag_img)
			fem_loc = cv2.resize(ffpts/255,(50,50))
			fem_locf = cv2.resize(ffptsf/255,(50,50))
			# cv2.imshow("img",fem_tag_img)
			# cv2.imshow("img2",pos_img)
			# cv2.waitKey(0)
			# exit()
			
		if x[0] in pel_list:
			# print("FOUND")
			pel_tag_img = cv2.imread(tag_dir_pathpel+x[0],0)
			pppts,ppptsf = getFlippedZoomImages(pel_tag_img)
			pel_loc = cv2.resize(pppts/255,(50,50))
			pel_locf = cv2.resize(ppptsf/255,(50,50))
			
		if x[0] in gen_list:
			# print("FOUND")
			gen_tag_img = cv2.imread(tag_dir_pathgen+x[0],0)
			ggpts,ggptsf = getFlippedZoomImages(gen_tag_img)
			gen_loc = cv2.resize(ggpts/255,(50,50))
			gen_locf = cv2.resize(ggptsf/255,(50,50))
		
		comb_feat = np.zeros((50,50,3),dtype=np.float32)
		comb_featf = np.zeros((50,50,3),dtype=np.float32)

		comb_feat[:,:,0]=fem_loc
		comb_feat[:,:,1]=pel_loc
		comb_feat[:,:,2]=gen_loc
		comb_featf[:,:,0]=fem_locf
		comb_featf[:,:,1]=pel_locf
		comb_featf[:,:,2]=gen_locf

		img_array = img_array + [ps,psf,ns,nsf]
		blank_target = np.zeros((50,50),dtype=np.float32)
		# comb_feat = np.zeros((50,50,3),dtype=np.float32)
		# comb_feat[:,:,0]=fem_loc
		# comb_feat[:,:,1]=pel_loc
		# comb_feat[:,:,2]=gen_loc	
		# pos_img_300_flip = cv2.flip(pos_img_300,1)
		target_array = target_array + [comb_feat,comb_featf,np.zeros((50,50,3),dtype=np.float32),np.zeros((50,50,3),dtype=np.float32)]
		if len(img_array)>=batch_size or training_set.index(x) == (len(training_set)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			c = sess.run(cost,feed_dict={input_image:img_array,input_target:target_array})
			print(c)
			sess.run(optimizer,feed_dict={input_image:img_array,input_target:target_array})
			saver.save(sess,'./tmp/models/train_accurate_models/'+model_name+'.ckpt')
			img_array = []
			target_array = []
		# break
	
	# break







