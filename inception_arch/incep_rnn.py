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
f = os.listdir(neg_train_dir)[2]
img = cv2.imread(neg_train_dir+f,0)




def flatten_layer(layer):
	layer_shape = layer.get_shape()		
	num_features = layer_shape[1:4].num_elements()
	print(num_features)
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat,num_features
def fcNode(input_layer,weights,biases):
	layer = tf.matmul(input_layer, weights) + biases
	layer = tf.nn.relu(layer)
	return layer

def rnnCell(input_layer,prev_output,wa,wb,wout,bout):
    total_input = tf.add(tf.matmul(input_layer, wa),tf.matmul(prev_output, wb))
    layer = tf.matmul(total_input, wout) + bout
	layer = tf.nn.relu(layer)
    return layer

def getConvWeightsandBiases(shape,index):
	w = tf.get_variable("convweights"+str(index), shape,initializer=tf.random_normal_initializer(mean=0.5,stddev=0.07))
	b = tf.get_variable("convbias"+str(index), [shape[3]],initializer=tf.constant_initializer(0.05))
	return w,b

# def var_summaries(var):

# def incepLayer(prev_layer,weights,bias):
#     l = tf.nn.conv2d(input=prev_layer,filter=weights,strides=[1, 1, 1, 1],padding='VALID')
#     l += bias




shape1 = [3,3,1,4]
shape2 = [5,5,4,4]
shape3 = [7,7,4,8]
shape4 = [9,9,8,8]
fcshape1 = [20000,500]
fcshape2 = [500,200]
fc2shape1 = [200,16]
fc2shape2 = [16,4]
fcshape3 = [200,2]
beta = 0.001

input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])
input_image_reshaped = tf.image.per_image_standardization(input_image_reshaped)
input_label = tf.placeholder(tf.float32,shape=[None,2],name="input_images_array")
input_tags = tf.placeholder(tf.float32,shape=[None,4],name="input_images_array")
# input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])

w1,b1 = getConvWeightsandBiases(shape1,1)
w2,b2 = getConvWeightsandBiases(shape2,2)
w3,b3 = getConvWeightsandBiases(shape3,3)
w4,b4 = getConvWeightsandBiases(shape4,4)
fcw1 = tf.get_variable("fcweightss_4", fcshape1,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fcw2 = tf.get_variable("fcweightss_5", fcshape2,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fcw3 = tf.get_variable("fcweightss_6", fcshape3,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fcb1 = tf.get_variable("fcbias_4", [500],initializer=tf.constant_initializer(0.05))
fcb2 = tf.get_variable("fcbias_5", [200],initializer=tf.constant_initializer(0.05))
fcb3 = tf.get_variable("fcbias_6", [2],initializer=tf.constant_initializer(0.05))
fc2w1 = tf.get_variable("fc2weightss_1", fc2shape1,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fc2b1 = tf.get_variable("fc2bias_1", fc2shape1[1],initializer=tf.constant_initializer(0.05))
fc2w2 = tf.get_variable("fc2weightss_2", fc2shape2,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fc2b2 = tf.get_variable("fc2bias_2", fc2shape2[1],initializer=tf.constant_initializer(0.05))

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
l4 = tf.nn.conv2d(input=l3,filter=w4,strides=[1, 1, 1, 1],padding='SAME')
l4 += b4
l4 = tf.nn.relu(l4)
lflat,numfeat = flatten_layer(l4)
layer_fc1 = fcNode(lflat,fcw1,fcb1)
layer_fc2 = fcNode(layer_fc1,fcw2,fcb2)
layer_fc3 = fcNode(layer_fc2,fcw3,fcb3)
layer_fc2_1 = fcNode(layer_fc2,fc2w1,fc2b1)
layer_fc2_2 = fcNode(layer_fc2_1,fc2w2,fc2b2)
fem_estimator = tf.nn.softmax(layer_fc2_2,0)
# fem_estimator = layer_fc2_2
layer_argmax = tf.math.argmax(layer_fc3,axis = 1)
label_argmax = tf.math.argmax(input_label,axis = 1)
equality = tf.equal(layer_argmax, label_argmax)
accuracy = tf.reduce_mean(tf.cast(equality,tf.float32))
a_label = input_label[:,0]
b_label = input_label[:,1]

activate_cond = tf.math.greater(a_label,b_label)

# layer_argmax = tf.math.argmax(layer_fc2,axis = 1)
error = tf.nn.softmax_cross_entropy_with_logits(labels = input_label,logits = layer_fc3)
error_fc2 = tf.math.square(tf.subtract(input_tags,fem_estimator))
print(error_fc2.get_shape())
error_fc2_false = tf.constant(0.0,shape=(64,4))
# error_fc2_false.set_shape(tf.shape(error_fc2))
print(error_fc2.dtype)
print(error_fc2_false.dtype)

# exit()
error_fc2_recalc = tf.where(activate_cond,error_fc2,error_fc2_false)#tf.constant(0.0,dtype=tf.float32,shape=error_fc2.get_shape()))
# cost = tf.reduce_mean(error)+ beta*tf.nn.l2_loss(fcw1)+ beta*tf.nn.l2_loss(fcw2)
cost = tf.reduce_mean(error)
cost_fc2 = tf.reduce_mean(error_fc2_recalc)
cost = cost + cost_fc2
cost_summary = tf.summary.scalar('training cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
init = tf.global_variables_initializer()
# tf.summary.scalar('Accuracy',accuracy)
# tf.summary.scalar('Cost',cost)
acc_summary = tf.summary.scalar('training accuracy',accuracy)
merged = tf.summary.merge_all()
summ_writer = tf.summary.FileWriter("./tmp/summaries_400")


model_name = 'incep33'
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

batch_size = 64
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
training_set = main_list[:4000]
testing_set = main_list[4000:]


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
	img_300 = cv2.resize(img[50:350,50:350],(image_size,image_size))
	img_300_flip = cv2.flip(img_300,1)
	return img,img_flip,img_300,img_300_flip

def getTagFeature(fem_tag_img):
	if np.amax(fem_tag_img) > 0.0:
		fem_tag_img = fem_tag_img/np.amax(fem_tag_img)
		bool_row = np.where(fem_tag_img==1.0)
		# print(bool_row)
		first_pt = [np.amin(bool_row[0])/400,np.amin(bool_row[1])/400]
		second_pt = [np.amax(bool_row[0])/400,np.amax(bool_row[1])/400]
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
		ps,psf,ps3,ps3f = getFlippedZoomImages(pos_img)
		ns,nsf,ns3,ns3f = getFlippedZoomImages(neg_img)
		
		# print(x[0])
		if x[0] in fem_list:
			# print("FOUND")
			fem_tag_img = cv2.imread(tag_dir_path+x[0],0)
			pts,ptsf,pts3,pts3f = getFlippedZoomImages(fem_tag_img)
			fpts = getTagFeature(pts)
			fptsf = getTagFeature(ptsf)
			fpts3 = getTagFeature(pts3)
			fpts3f = getTagFeature(pts3f)
		else:
			fpts = [0.0,0.0,0.0,0.0]
			fptsf = [0.0,0.0,0.0,0.0]
			fpts3 = [0.0,0.0,0.0,0.0]
			fpts3f = [0.0,0.0,0.0,0.0]
		# exit()
		

		# cv2.imshow("img",pos_img)
		# cv2.imshow("img2",cv2.flip(pos_img,1))
		# # cv2.imshow("img3",pos_img[20:380,20:380])
		# cv2.imshow("img4",pos_img[50:350,50:350])
		# cv2.waitKey(0)
		# exit()
		img_array = img_array + [ps,psf,ps3,ps3f,ns,nsf,ns3,ns3f]
		
		# pos_img_300_flip = cv2.flip(pos_img_300,1)
		label_array = label_array + [[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]
		tag_array = tag_array + [fpts,fptsf,fpts3,fpts3f,[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
		if len(img_array)>=batch_size or training_set.index(x) == (len(training_set)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			# output = sess.run(input_image,feed_dict={input_image:img_array})
			# print(output)
			print(np.array(label_array).shape)
			# print(np.array(tag_array).shape)
			# exit()
			output = sess.run(cost,feed_dict={input_image:img_array,input_label:label_array,input_tags:tag_array})
			# print(output)
			for x in range(4):
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
	# curr_acc = testNN(testing_set)
	# if curr_acc > prev_acc:
	# 	saver.save(sess,'./tmp/models/test_accurate_models/'+model_name+'.ckpt')
	# 	# f = open("model_acc.txt",'a+')
	# 	# f.write("TESTING ACC : " + str(curr_acc)+"\n")
	# 	# f.close()
	# 	prev_acc = curr_acc
	f = open("model_acc.txt",'a+')
	f.write("TRAINING ACC : " + str(training_acc) + " , TESTING ACC : " + str(prev_acc)+"\n")
	f.write("EPOCH COMPLETE\n")
	f.close()








