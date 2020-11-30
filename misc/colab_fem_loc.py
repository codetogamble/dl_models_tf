import tensorflow as tf
import numpy as np
import cv2
import os
import random
import csv
import sys

pos_data_dir = "C:/Users/user1/Documents/securesound/training_data_2/roi_pos/"
pos_data = os.listdir(pos_data_dir)

neg_data_dir = "C:/Users/user1/Documents/securesound/training_data_2/roi_neg/"
neg_data = os.listdir(neg_data_dir)

pos_data = pos_data + neg_data

fem_dir = "C:/Users/user1/Documents/securesound/training_data_2/tag_pos/roi_posfem/"
fem_files = os.listdir(fem_dir)

pel_dir = "C:/Users/user1/Documents/securesound/training_data_2/tag_pos/roi_pospel/"
pel_files = os.listdir(pel_dir)


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
beta = 0.001

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
nfilters6 = 10

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

batch_size = 1
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
	layer_fc3 = fcNode(layer_fc2,w6,b6)
	reg_term = beta*tf.nn.l2_loss(w4)+ beta*tf.nn.l2_loss(w5)+ beta*tf.nn.l2_loss(w6)
	return layer_fc3,reg_term

image_size = 400

input_image = tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size],name="input_images_array")
input_image_fempts = tf.placeholder(tf.float32,shape=[batch_size,2,4],name="input_images_tagged_array")

x_image = tf.reshape(input_image, [batch_size, image_size, image_size, 1])

result,reg_term = applyCNN(x_image)
threshs,values = tf.split(result,[2,8],1)
values = tf.reshape(values,[batch_size,2,4])
cost_adjust = 10*tf.reduce_mean(tf.math.square(tf.subtract(tf.fill(tf.shape(values), 0.5),values)))

threshs = tf.reshape(threshs,[batch_size,2,1])
cond_thresh = tf.less(threshs,0.1)
# mult_mask = tf.where(cond_thresh,tf.constant(0.0,shape=[batch_size,2,1]),tf.constant(1.0,shape=[batch_size,2,1]))
mult_mask = tf.where(cond_thresh,tf.fill(tf.shape(threshs), 0.0),tf.fill(tf.shape(threshs), 0.0))
after_val = tf.multiply(values,mult_mask)
error = tf.math.square(after_val - input_image_fempts)
cost = tf.reduce_mean(error) + reg_term + tf.reduce_mean(error)*cost_adjust

# image_patches = tf.extract_image_patches(x_image,[1,window_size,window_size,1],[1,stride,stride,1],[1,1,1,1],padding='SAME')
# image_patches_tag = tf.extract_image_patches(x_image_tagged_guide,[1,window_size,window_size,1],[1,stride,stride,1],[1,1,1,1],padding='SAME')

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
try:
		saver.restore(sess,"C:/Users/user1/Documents/securesound/tmp/models/test_accurate_models/model_fem_loc3.ckpt")
except:
		print("MODEL NOT FOUND")
		pass


# sess.run(tf.variables_initializer(all_variables))
random.seed(101)
random.shuffle(pos_data)
training_data = pos_data[:]
# testing_data = pos_data[1700:]



def getFeatureArray(imgpath):
	feature_array_fem = [0.0,0.0,0.0,0.0]
	feature_array_pel = [0.0,0.0,0.0,0.0]
	if img_file in fem_files:
#       print(img_file)
		tmplist = img_file.split("_")
		frame_number = int(tmplist[3])
		tag_img = np.zeros((400,400),dtype=np.uint8)
		for x in range(frame_number-2,frame_number+2):
			tmplist[3] = str(x)
	#         print(tmplist)
			new_name = "_".join(tmplist)
			if new_name in fem_files:
				tag_img = cv2.add(tag_img,cv2.imread(fem_dir + new_name,0))
#         print(new_name)
		if np.amax(tag_img)>0.0:
			tag_img = tag_img/np.amax(tag_img)
			bool_row = np.where(tag_img==1.0)
	#         print(bool_row)
			first_pt = [np.amin(bool_row[0])/400,np.amin(bool_row[1])/400]
			second_pt = [np.amax(bool_row[0])/400,np.amax(bool_row[1])/400]
	#         print(first_pt)
	#         print(second_pt)
			feature_array_fem = first_pt+second_pt
	if img_file in pel_files:
	#       print(img_file)
			tmplist = img_file.split("_")
			frame_number = int(tmplist[3])
			tag_img = np.zeros((400,400),dtype=np.uint8)
			for x in range(frame_number-2,frame_number+2):
				tmplist[3] = str(x)
		#         print(tmplist)
				new_name = "_".join(tmplist)
				if new_name in pel_files:
					tag_img = cv2.add(tag_img,cv2.imread(pel_dir + new_name,0))
			#         print(new_name)
			if np.amax(tag_img)>0.0:
				tag_img = tag_img/np.amax(tag_img)
				bool_row = np.where(tag_img==1.0)
		#         print(bool_row)
				first_pt = [np.amin(bool_row[0])/400,np.amin(bool_row[1])/400]
				second_pt = [np.amax(bool_row[0])/400,np.amax(bool_row[1])/400]
		#         print(first_pt)
		#         print(second_pt)
				feature_array_pel = first_pt+second_pt
	# print(feature_array_fem)
	# print(feature_array_pel)
	return feature_array_fem,feature_array_pel


####################
# input_array_img = []
# input_array_features = []
# for img_file in training_data[:200]:
# #     print(img_file)
# 	img = cv2.imread(pos_data_dir+img_file,0)
# 	if img is None:
# 		img = cv2.imread(neg_data_dir+img_file,0)
# 	img = img/255
# 	femFA,pelFA = getFeatureArray(img_file)
# 	print(femFA)
# 	print(pelFA)
# 	# rect_box = [int(x*400) for x in femFA]
# 	# rect_box2 = [int(x*400) for x in pelFA]
# 	# rgb_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# 	# cv2.rectangle(rgb_img,(rect_box[1],rect_box[0]),(rect_box[3],rect_box[2]),(0,255,0),2)
# 	# cv2.rectangle(rgb_img,(rect_box2[1],rect_box2[0]),(rect_box2[3],rect_box2[2]),(0,0,255),2)
# 	# cv2.imshow("img",rgb_img)
# 	# cv2.waitKey(0)
# 	input_feature = [femFA,pelFA]
# 	input_array_img.append(img)
# 	input_array_features.append(input_feature)
# 	if len(input_array_img) == batch_size or training_data.index(img_file) == (len(training_data)-1):
# 		output1 = sess.run(cost,feed_dict={input_image:input_array_img,input_image_fempts:input_array_features})
# 		print(output1.shape)
# 		print(output1)
# 		o = sess.run(result,feed_dict={input_image:input_array_img,input_image_fempts:input_array_features})
# 		print(o)
# 		output1 = sess.run(cost,feed_dict={input_image:input_array_img,input_image_fempts:input_array_features})
# 		print(output1.shape)
# 		print(output1)
# 		# saver.save(sess,os.getcwd()+"/drive/My Drive/obs_diag_data/tmp/models/test_accurate_models/model_fem_loc2.ckpt")
# 		input_array_img = []
# 		input_array_features = []
# 		# break


test_vid = "dvr_20190610_1050"

vidreader = cv2.VideoCapture("C:/Users/user1/Documents/securesound/video_data/"+test_vid+".mp4")

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
		sum_mid_vert = np.sum(thresh[:,int(
			gray.shape[1]/2)])
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
		output = sess.run([after_val,result],feed_dict={input_image:[out/255]})
		print(output)
		rect_box1 = [int(x*400) for x in output[0][0][0]]
		rect_box2 = [int(x*400) for x in output[0][0][1]]
		print(output[1])
		rgb_img = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
		cv2.rectangle(rgb_img,(rect_box1[1],rect_box1[0]),(rect_box1[3],rect_box1[2]),(0,255,0),2)
		cv2.rectangle(rgb_img,(rect_box2[1],rect_box2[0]),(rect_box2[3],rect_box2[2]),(0,0,255),2)
		cv2.imshow("img",rgb_img)
		# cv2.waitKey(0)
		# cv2.imshow("img",out)
		cv2.waitKey(0)
	else:
		vidreader.release()
		break
