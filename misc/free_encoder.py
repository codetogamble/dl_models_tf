import tensorflow as tf
import cv2
import os
import numpy as np
import random
from tensorflow import keras
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential

def flatten_layer(layer):
	layer_shape = layer.get_shape()		
	num_features = layer_shape[1:4].num_elements()
	print(num_features)
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat,num_features

tf.reset_default_graph()
sess = tf.Session()
print(sess.graph.get_collection('trainable_variables'))
# exit()
model_name = "autoencoder"
model_dir = "/home/gawd/workspace/securesound/tmp/autoencoder/"
imported_meta = tf.train.import_meta_graph(model_dir+model_name+".ckpt.meta")
imported_meta.restore(sess,model_dir+model_name+".ckpt")
g = sess.graph
print(g.get_collection('trainable_variables'))
graphdef = g.as_graph_def()
# print(graphdef)
subgraph = tf.graph_util.extract_sub_graph(graphdef,['conv2d_transpose_3/Relu'])
# tf.graph_util.import_graph_def(subgraph)
# new_g = sess.graph.as_graph_def()
const_g = tf.graph_util.convert_variables_to_constants(sess,subgraph,['conv2d_transpose_3/Relu'])
# tf.io.write_graph(const_g,'./frozen_auto/','trained_auto.pbtxt')
# tf.reset_default_graph()
tf.import_graph_def(const_g)
g = sess.graph
print(g.as_graph_def())

exit()

input_image = g.get_tensor_by_name("input_images_array:0")
encoder_result = g.get_tensor_by_name("conv2d_4/Relu:0")
decoder_result = g.get_tensor_by_name("conv2d_transpose_3/Relu:0")
lflat,num_features = flatten_layer(encoder_result)
fc1 = Layers.Dense(64,activation='relu')
fc1_out = fc1(lflat)
fc2 = Layers.Dense(2,activation='relu')
class_output = fc2((fc1_out))
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
# init = tf.global_variables_initializer()
# sess.run(init)


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
batch_size = 1
main_list = list(zip(pos_list,neg_list))
random.shuffle(main_list)
print(len(main_list))
training_set = main_list[:4512]
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


for epoch in range(1):
	for x in training_set[:]:
		pos_img = cv2.imread(pos_train_dir+x[0],0)
		neg_img = cv2.imread(neg_train_dir+x[1],0)
		ps,psf = getFlippedZoomImages(pos_img)
		ns,nsf = getFlippedZoomImages(neg_img)
		img_array = img_array + [ps,psf,ns,nsf]
		if len(img_array)>=batch_size or training_set.index(x) == (len(training_set)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			output = sess.run(decoder_result,feed_dict={input_image:img_array})
			# print(output)
			cv2.imshow("img",img_array[0])
			cv2.imshow("img2",output[0])
			# cv2.imshow("img3",img_array[2])
			# cv2.imshow("img4",output[2])
			cv2.waitKey(0)
			exit()
			
			# exit()
		# break
	
	# break


