import cv2
import numpy as np
import tensorflow as tf
import os
import random


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
batch_size = 128

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
	layer = tf.nn.relu(layer)
	return layer

def getConvWeightsandBiases(shape,index):
	w = tf.get_variable("convweights"+str(index), shape,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.03))
	b = tf.get_variable("convbias"+str(index), [shape[3]],initializer=tf.constant_initializer(0.05))
	return w,b

# def var_summaries(var):

def incepLayer(prev_layer,w1_incep,w3_incep,w5_incep,w5_pool,b1_incep,b3_incep,b5_incep,b5_pool):
	l1 = tf.nn.conv2d(input=prev_layer,filter=w1_incep,strides=[1, 1, 1, 1],padding='SAME')
	l1 += b1_incep
	l3 = tf.nn.conv2d(input=prev_layer,filter=w3_incep,strides=[1, 1, 1, 1],padding='SAME')
	l3 += b3_incep
	l5 = tf.nn.conv2d(input=prev_layer,filter=w5_incep,strides=[1, 1, 1, 1],padding='SAME')
	l5 += b5_incep
	pl_mpool = tf.nn.max_pool(value=prev_layer,ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1],padding='SAME')
	l5_2 = tf.nn.conv2d(input=pl_mpool,filter=w5_pool,strides=[1, 1, 1, 1],padding='SAME')
	l5_2 += b5_pool
	out = tf.concat([l1,l3,l5,l5_2],axis=1)
	out = tf.nn.relu(out)
	return out




shape1 = [1,1,1,1]
shape3 = [3,3,1,1]
shape5 = [5,5,1,1]
shape4 = [9,9,8,8]
fcshape1 = [14250,200]
fcshape2 = [200,2]
fc2shape1 = [14250,400]
fc2shape2 = [400,100]
fc2shape3 = [100,4]
beta = 0.003
beta2 = 0.003

input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])
input_image_reshaped = tf.image.per_image_standardization(input_image_reshaped)
input_label = tf.placeholder(tf.float32,shape=[None,2],name="input_images_array")
input_tags = tf.placeholder(tf.float32,shape=[None,4],name="input_images_array")
# input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])

w1_incep_1,b1_incep_1 = getConvWeightsandBiases(shape1,1)
w3_incep_1,b3_incep_1 = getConvWeightsandBiases(shape3,2)
w5_incep_1,b5_incep_1 = getConvWeightsandBiases(shape5,3)
w5_pool_1,b5_pool_1 = getConvWeightsandBiases(shape5,4)
layer_incep_1 = incepLayer(input_image_reshaped,w1_incep_1,w3_incep_1,w5_incep_1,w5_pool_1,b1_incep_1,b3_incep_1,b5_incep_1,b5_pool_1)
layer_incep_1 = tf.nn.max_pool(value=layer_incep_1,ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1],padding='SAME')
w1_incep_2,b1_incep_2 = getConvWeightsandBiases(shape1,5)
w3_incep_2,b3_incep_2 = getConvWeightsandBiases(shape3,6)
w5_incep_2,b5_incep_2 = getConvWeightsandBiases(shape5,7)
w5_pool_2,b5_pool_2 = getConvWeightsandBiases(shape5,8)
layer_incep_2 = incepLayer(layer_incep_1,w1_incep_2,w3_incep_2,w5_incep_2,w5_pool_2,b1_incep_2,b3_incep_2,b5_incep_2,b5_pool_2)
layer_incep_2 = tf.nn.max_pool(value=layer_incep_2,ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1],padding='SAME')
w1_incep_3,b1_incep_3 = getConvWeightsandBiases(shape1,9)
w3_incep_3,b3_incep_3 = getConvWeightsandBiases(shape3,10)
w5_incep_3,b5_incep_3 = getConvWeightsandBiases(shape5,11)
w5_pool_3,b5_pool_3 = getConvWeightsandBiases(shape5,12)
layer_incep_3 = incepLayer(layer_incep_2,w1_incep_3,w3_incep_3,w5_incep_3,w5_pool_3,b1_incep_3,b3_incep_3,b5_incep_3,b5_pool_3)
layer_incep_3 = tf.nn.max_pool(value=layer_incep_3,ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1],padding='SAME')
w1_incep_4,b1_incep_4 = getConvWeightsandBiases(shape1,13)
w3_incep_4,b3_incep_4 = getConvWeightsandBiases(shape3,14)
w5_incep_4,b5_incep_4 = getConvWeightsandBiases(shape5,15)
w5_pool_4,b5_pool_4 = getConvWeightsandBiases(shape5,16)
layer_incep_4 = incepLayer(layer_incep_3,w1_incep_4,w3_incep_4,w5_incep_4,w5_pool_4,b1_incep_4,b3_incep_4,b5_incep_4,b5_pool_4)


fcw1 = tf.get_variable("fcweightss_4", fcshape1,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.03))
fcw2 = tf.get_variable("fcweightss_5", fcshape2,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.03))
fcb1 = tf.get_variable("fcbias_4", [fcshape1[1]],initializer=tf.constant_initializer(0.05))
fcb2 = tf.get_variable("fcbias_5", [fcshape2[1]],initializer=tf.constant_initializer(0.05))
fc2w1 = tf.get_variable("fc2weightss_1", fc2shape1,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.03))
fc2b1 = tf.get_variable("fc2bias_1", fc2shape1[1],initializer=tf.constant_initializer(0.05))
fc2w2 = tf.get_variable("fc2weightss_2", fc2shape2,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.03))
fc2b2 = tf.get_variable("fc2bias_2", fc2shape2[1],initializer=tf.constant_initializer(0.05))
fc2w3 = tf.get_variable("fc2weightss_3", fc2shape3,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.03))
fc2b3 = tf.get_variable("fc2bias_3", fc2shape3[1],initializer=tf.constant_initializer(0.05))

lflat,numfeat = flatten_layer(layer_incep_4)
layer_fc1 = fcNode(lflat,fcw1,fcb1)
layer_fc2 = fcNode(layer_fc1,fcw2,fcb2)
layer_fc2_1 = fcNode(lflat,fc2w1,fc2b1)
layer_fc2_2 = fcNode(layer_fc2_1,fc2w2,fc2b2)
layer_fc2_3 = fcNode(layer_fc2_2,fc2w3,fc2b3)
layer_fc2_max = tf.tile(tf.reshape(tf.math.reduce_max(layer_fc2,1),(-1,1)),[1,2])
layer_fc2_3_max = tf.tile(tf.reshape(tf.math.reduce_max(layer_fc2_3,1),(-1,1)),[1,4])
class_output = tf.div(layer_fc2,layer_fc2_max,name="class_output")
loc_output = tf.div(layer_fc2_3,layer_fc2_3_max,name="loc_output")
# fem_estimator = layer_fc2_2
layer_argmax = tf.math.argmax(layer_fc2,axis = 1)
label_argmax = tf.math.argmax(input_label,axis = 1)
equality = tf.equal(layer_argmax, label_argmax)
accuracy = tf.reduce_mean(tf.cast(equality,tf.float32))
# a_label = input_label[:,0]
# b_label = input_label[:,1]
a_label = layer_fc2[:,0]
b_label = layer_fc2[:,1]

activate_cond = tf.math.greater(a_label,b_label)

# layer_argmax = tf.math.argmax(layer_fc2,axis = 1)
error = tf.nn.softmax_cross_entropy_with_logits(labels = input_label,logits = class_output)
error_fc2 = tf.math.square(tf.subtract(input_tags,loc_output))
error_fc2_false = tf.constant(0.0,shape=(batch_size,4))
error_fc2_recalc = tf.where(activate_cond,error_fc2,error_fc2_false)
# error_fc2 = tf.nn.sigmoid_cross_entropy_with_logits(labels = input_tags,logits = layer_fc2_2)
reg_term = beta*tf.nn.l2_loss(fcw1)+ beta*tf.nn.l2_loss(fcw2) + beta2*tf.nn.l2_loss(fc2w1) + beta2*tf.nn.l2_loss(fc2w2)+beta*tf.nn.l2_loss(fcb1)+beta*tf.nn.l2_loss(fcb2) + beta2*tf.nn.l2_loss(fc2b1)+beta2*tf.nn.l2_loss(fc2b2)
# exit()

# cost = tf.reduce_mean(error)+ beta*tf.nn.l2_loss(fcw1)+ beta*tf.nn.l2_loss(fcw2)
cost = tf.reduce_mean(error)
cost_fc2 = tf.reduce_mean(error_fc2_recalc)
cost = cost + cost_fc2 + reg_term
cost_summary = tf.summary.scalar('training cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
init = tf.global_variables_initializer()
# tf.summary.scalar('Accuracy',accuracy)
# tf.summary.scalar('Cost',cost)
acc_summary = tf.summary.scalar('training accuracy',accuracy)
merged = tf.summary.merge_all()
summ_writer = tf.summary.FileWriter("./tmp/summaries_400")


model_name = 'incep99'
# model_name = 'trigger_model_8imgs'

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf.Session()



# saver.restore(sess,"C:\\Users\\user1\\Documents\\securesound\\tmp\\models\\train_accurate_models\\"+model_name+".ckpt")

try:
	saver.restore(sess,"./tmp/models/test_accurate_models/"+model_name+".ckpt")
except:
	sess.run(init)
	print("MODEL NOT FOUND")
	pass

# exit()

print(pos_list[:2])
print(neg_list[:2])


 #MULTIPLES OF 2

def testNN(testing_list):
	total = 0
	accurate = 0
	img_array = []
	label_array = []
	for x in testing_list[:]:
		pos_img = cv2.imread(pos_train_dir+x[0],0)
		neg_img = cv2.imread(neg_train_dir+x[1],0)
		img_array.append(pos_img/255)
		img_array.append(neg_img/255)
		label_array = label_array + [[1.0,0.0]] + [[0.0,1.0]]
		if len(img_array)>=batch_size or testing_list.index(x) == (len(testing_list)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			# output = sess.run(input_image,feed_dict={input_image:img_array})
			# print(output)
			output = sess.run(layer_fc2,feed_dict={input_image:img_array,input_label:label_array})
			for x in range(len(output)):
				if np.argmax(output[x]) == np.argmax(label_array[x]):
					accurate = accurate + 1
				else:
					pass
			img_array = []
			label_array = []    
			total = total + batch_size
	acc = accurate/total
	print("ACCURACY : " + str(accurate/total))
	return acc




main_list = list(zip(pos_list,neg_list))
random.shuffle(main_list)
print(len(main_list))
training_set = main_list[:4512]
# testing_set = main_list[4000:]


img_array = []
label_array = []
tag_array = []
prev_acc = 0
count = 0
tag_dir_path = "./training_data_3/tag_pos/roi_posfem/"
fem_list = os.listdir(tag_dir_path)


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

for epoch in range(20):
	for x in training_set[:]:
		pos_img = cv2.imread(pos_train_dir+x[0],0)
		neg_img = cv2.imread(neg_train_dir+x[1],0)
		ps,psf = getFlippedZoomImages(pos_img)
		ns,nsf = getFlippedZoomImages(neg_img)
		
		# print(x[0])
		if x[0] in fem_list:
			# print("FOUND")
			fem_tag_img = cv2.imread(tag_dir_path+x[0],0)
			pts,ptsf = getFlippedZoomImages(fem_tag_img)
			fpts = getTagFeature(pts)
			fptsf = getTagFeature(ptsf)
			# rgb_ps = cv2.cvtColor(ps,cv2.COLOR_GRAY2BGR)
			# rgb_pts = cv2.cvtColor(pts,cv2.COLOR_GRAY2BGR)
			# print(rgb_pts.shape)
			# cv2.rectangle(rgb_ps,(int(fpts[1]*image_size),int(fpts[0]*image_size)),(int(fpts[3]*image_size),int(fpts[2]*image_size)),(0,255,0),2)
			# cv2.rectangle(rgb_pts,(int(fpts[1]*image_size),int(fpts[0]*image_size)),(int(fpts[3]*image_size),int(fpts[2]*image_size)),(0,255,0),2)
			# cv2.imshow("img",rgb_ps)
			# cv2.imshow("imgt",rgb_pts)
			# cv2.waitKey(0)
			# continue
			# exit()
		else:
			fpts = [0.0,0.0,0.0,0.0]
			fptsf = [0.0,0.0,0.0,0.0]
		
		img_array = img_array + [ps,psf,ns,nsf]
		
		# pos_img_300_flip = cv2.flip(pos_img_300,1)
		label_array = label_array + [[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0]]
		tag_array = tag_array + [fpts,fptsf,[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
		if len(img_array)>=batch_size or training_set.index(x) == (len(training_set)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			# output = sess.run(input_image,feed_dict={input_image:img_array})
			# print(output)
			# print(np.array(label_array).shape)
			# print(np.array(tag_array).shape)
			# exit()
			output = sess.run(cost,feed_dict={input_image:img_array,input_label:label_array,input_tags:tag_array})
			# res_tr,res_loc = sess.run([class_output,loc_output],feed_dict={input_image:img_array})
			# print(res_tr)
			# print(res_loc)
			
			# exit()
			# print(output)
			for x in range(2):
				sess.run(optimizer,feed_dict={input_image:img_array,input_label:label_array,input_tags:tag_array})
			summary,training_acc = sess.run([merged,accuracy],feed_dict={input_image:img_array,input_label:label_array,input_tags:tag_array})
			summ_writer.add_summary(summary,count)
			# print(output)
			img_array = []
			label_array = []
			tag_array = []
			# saver.save(sess,'./tmp/models/train_accurate_models/'+model_name+'.ckpt',global_step=count)
			saver.save(sess,'./tmp/models/train_accurate_models/'+model_name+'.ckpt')
			print("training_accuracy : " + str(training_acc))
			count = count + 1
			# training_acc = testNN(training_set)








