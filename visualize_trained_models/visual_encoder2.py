import cv2
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow import keras
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
import math
from google.protobuf import text_format

# exit()
# model_name = "autoencoder_large3"
# model_dir = "auto_large3"
model_name = "RNN_AUTO"
model_dir = "RNN_AUTO"
image_size = 400

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
# exit()
# shape1 = conv_out.get_shape()
# conv_out_act1 = conv_out[:0,:,:,:0]
# conv_out_edit = tf.zeros((1,shape1[1],shape1[2],shape1[3]),dtype=tf.float32)
# conv_out_edit = tf.concat((conv_out_act1,conv_out_edit),axis=3)
# assign_conv_out = tf.assign(conv_out,conv_out_edit)
# tf.assign(conv_out_edit[:,:,:,0],conv_out[:,:,:,0])
# tf.assign(conv_out,conv_out_edit)

# exit()
# conv2d_transpose_5/Relu

# graphdef = g.as_graph_def()
# subgraph = tf.graph_util.extract_sub_graph(graphdef,['conv2d_4/Relu'])
# const_g = tf.graph_util.convert_variables_to_constants(sess,graphdef,['conv2d_transpose_4/Relu'])
# tf.io.write_graph(const_g,'./frozen_auto/','trained_autol4.pbtxt')
# exit()

# output = sess.run(deconv5,feed_dict={input_image:img_array})

# f = open("./frozen_auto/trained_auto.pbtxt",'rb')
# data = f.read()
# print(type(data))
# gdef = tf.GraphDef()
# text_format.Merge(data,gdef)
# tf.graph_util.import_graph_def(gdef)
# # print(graphstr)
# g = sess.graph
# f.close()


def getMapImage(conv_out):
	# wsz = conv_out.shape[0]
	wsz = 100
	no_filters = conv_out.shape[2]
	# print(wsz)
	# print(no_filters)
	# print("++++++++====")
	if no_filters == 1:
		min_val = np.amin(conv_out[:,:,0])
		zeroes_val = conv_out[:,:,0] - min_val
		max_val = np.amax(zeroes_val)
		norm_val = zeroes_val/max_val
		return norm_val
	rows = int(no_filters/4)+1
	cols = int(no_filters/rows)+1
	# print(rows*(wsz))
	# print(cols*(wsz))
	blank_grid = np.zeros((cols*(wsz),rows*(wsz)),dtype=np.float32)
	for i in range(no_filters):
		fil = conv_out[:,:,i]
		rem = i%rows
		div = int(i/rows)
		# print(rem)
		# print(div)
		blank_grid[div*wsz:div*wsz+wsz,rem*wsz:rem*wsz+wsz] = cv2.resize(fil,(100,100))
		# blank_grid[]
	min_val = np.amin(blank_grid)
	zeroes_val = blank_grid - min_val
	max_val = np.amax(zeroes_val)
	norm_val = zeroes_val/max_val
	return norm_val

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
			# output = sess.run(conv_out,feed_dict={input_image:[out/255,out2/255]})
			output = sess.run(deconv_out,feed_dict={input_image:[out/255,out2/255]})
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
			output = sess.run(deconv_out,feed_dict={input_image:[out/255]})
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
	

