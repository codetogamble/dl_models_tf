import cv2
import numpy as np
import tensorflow as tf
import os
import random


global batch_size

pos_train_dir = "./training_data_2/roi_pos/"
neg_train_dir = "./training_data_2/roi_neg/"

neg_list = os.listdir(neg_train_dir)
pos_list = os.listdir(pos_train_dir)

if len(pos_list)<len(neg_list):
	neg_list = neg_list[:len(pos_list)]
else:
	pos_list = pos_list[:len(neg_list)]

image_size = 300
f = os.listdir(neg_train_dir)[2]
img = cv2.imread(neg_train_dir+f,0)




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

shape1 = [3,3,1,2]
shape2 = [4,4,2,4]
shape3 = [4,4,4,4]
fcshape1 = [5776,20]
fcshape2 = [20,2]
beta = 0.001

input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])
input_label = tf.placeholder(tf.float32,shape=[None,2],name="input_images_array")

# input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])

w1 = tf.get_variable("weightss_1", shape1,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
w2 = tf.get_variable("weightss_2", shape2,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
w3 = tf.get_variable("weightss_3", shape3,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
b1 = tf.get_variable("bias_1", [2],initializer=tf.constant_initializer(0.05))
b2 = tf.get_variable("bias_2", [4],initializer=tf.constant_initializer(0.05))
b3 = tf.get_variable("bias_3", [4],initializer=tf.constant_initializer(0.05))
fcw1 = tf.get_variable("weightss_4", fcshape1,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fcw2 = tf.get_variable("weightss_5", fcshape2,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fcb1 = tf.get_variable("bias_4", [20],initializer=tf.constant_initializer(0.05))
fcb2 = tf.get_variable("bias_5", [2],initializer=tf.constant_initializer(0.05))

l1 = tf.nn.conv2d(input=input_image_reshaped,filter=w1,strides=[1, 1, 1, 1],padding='SAME')
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
layer_fc1 = fcNode(lflat,fcw1,fcb1)
layer_fc2 = fcNode(layer_fc1,fcw2,fcb2)

layer_argmax = tf.math.argmax(layer_fc2,axis = 1)
label_argmax = tf.math.argmax(input_label,axis = 1)
equality = tf.equal(layer_argmax, label_argmax)
accuracy = tf.reduce_mean(tf.cast(equality,tf.float32))

# layer_argmax = tf.math.argmax(layer_fc2,axis = 1)
error = tf.math.square(layer_fc2 - input_label) + beta*tf.nn.l2_loss(fcw1)+ beta*tf.nn.l2_loss(fcw2)
cost = tf.reduce_mean(error)
cost_summary = tf.summary.scalar('training cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
# init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
# sess.run(init)
saver.restore(sess,'./tmp/models/test_accurate_models/model_400imgsize_2filters.ckpt')
print(pos_list[:2])
print(neg_list[:2])

batch_size = 128 #MULTIPLES OF 2


# for img in neg_list:
# 	print(img)
# 	img = cv2.imread(neg_train_dir+img,0)
# 	cv2.imshow("img",img)
# 	cv2.waitKey(0)
# 	output = sess.run(layer_fc2,feed_dict={input_image:[img,img]})
# 	print(output)
# 	break
	# if output[0][1]>output[0][0]:
	#     print("NEGATIVE")
	# else:
	#     print("POSITIVE")
	# break
# exit()

test_vid = "dvr_20190611_1531"

vidreader = cv2.VideoCapture("./video_data/"+test_vid+".mp4")

tt,ff = vidreader.read()
print(ff.shape)
ff = ff[60:]
prev_frame = np.zeros((int(ff.shape[0]/16),int(ff.shape[1]/16)),dtype=np.uint8)
count = 0
while(True):
	ret,frame = vidreader.read()
	if ret == True:
		gray2 = cv2.cvtColor(frame[60:],cv2.COLOR_BGR2GRAY)
		main_image_shape = gray2.shape
		gray = cv2.resize(gray2,(int(gray2.shape[1]/16),int(gray2.shape[0]/16)))
		change = np.abs(gray - prev_frame)
		prev_frame = gray
		change = change/np.amax(change)
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
			if np.sum(change[:,int(gray.shape[1]/2):]) > np.sum(change[:,:int(gray.shape[1]/2)]):
				# print("RIGHT")
				rightsplit_bool = True
				out = gray2[:,int(gray2.shape[1]/2):]
				# print(out.shape)
			else:
				# print("LEFT")
				out = gray2[:,:int(gray2.shape[1]/2)]
				# print(out.shape)
			wsz = out.shape[1]
			# print(out.shape)
		else:
			# print("single")
			out = gray2
			# print(out.shape)
			wsz = 600
		vert_gap = int((out.shape[0]-wsz)/2)
		hori_gap = int((out.shape[1]-wsz)/2)
		out = out[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
		out = cv2.resize(out,(400,400))
		output = sess.run(layer_fc2,feed_dict={input_image:[out]})
		print(output)
		if output[0][1] < output[0][0]:
			print("POSITIVE")
		else:
			print("NEGATIGVE")
		cv2.imshow("img",out)
		cv2.waitKey(10)
	else:
		vidreader.release()
		break




