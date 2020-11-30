import cv2
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow import keras
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
import copy

# exit()
model_name = "AUTO_LSTM"
model_dir = "AUTO_LSTM"
image_size = 400
batch_size = 90

def flatten_layer(layer):
	layer_shape = layer.get_shape()		
	num_features = layer_shape[1:4].num_elements()
	print(num_features)
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat,num_features

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

imported_meta = tf.train.import_meta_graph("./tmp/"+model_dir+"/"+model_name+".ckpt.meta")
imported_meta.restore(sess,"./tmp/"+model_dir+"/"+model_name+".ckpt")
g = sess.graph
for i in g.get_operations():
	print(i.name)
	# if 'conv2d_4/Relu' in i.name:
# for i in g.get_variables():
# 	print(i)
# exit()
input_image = g.get_tensor_by_name('input_images_array:0')
reshapedd = g.get_tensor_by_name('Reshape:0')

deconv_out = g.get_tensor_by_name('conv2d_transpose_5/Relu:0')
conv_out = g.get_tensor_by_name('conv2d_3/Relu:0')
print(dir(conv_out))
print(conv_out.value_index)
# graphdef = g.as_graph_def()
# subgraph = tf.graph_util.extract_sub_graph(graphdef,['lstm_1/transpose_1'])

# const_g = tf.graph_util.convert_variables_to_constants(sess,graphdef,['conv2d_5/Relu'])
# tf.io.write_graph(const_g,'./frozen_auto/','trained_lstm_ENCODER.pbtxt')
# exit()
img_array = []
img_array2 = []
test_folder = "./test_vids/"
for f in os.listdir(test_folder):
	vidreader = cv2.VideoCapture(test_folder+f)
	# vidreader.set(cv2.CAP_PROP_POS_FRAMES,4000)
	tt = True
	while(tt):
		tt,ff = vidreader.read()
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
		
		if len(img_array) == batch_size and len(img_array2) == batch_size:
			c = sess.run(deconv_out,feed_dict={input_image:img_array})
			print(c)
			cv2.imshow("img",img_array[-1])
			cv2.imshow("img2",c[-1])
			cv2.waitKey(10)
			c2 = sess.run(deconv_out,feed_dict={input_image:img_array2})
			print(c2)
			cv2.imshow("img3",img_array2[-1])
			cv2.imshow("img4",c2[-1])
			cv2.waitKey(10)
			img_array = copy.deepcopy(img_array[:-30])
			img_array2 = copy.deepcopy(img_array2[:-30])
			# break
		elif len(img_array) == batch_size:
			c = sess.run(deconv_out,feed_dict={input_image:img_array})
			print(c)
			cv2.imshow("img",img_array[-1])
			cv2.imshow("img2",c[-1])
			cv2.waitKey(10)
			img_array = copy.deepcopy(img_array[:-30])
			# break
		# else:
		# 	pass
	vidreader.release()
	# exit()




