import cv2
import numpy as np
import tensorflow as tf
import os
import random


global batch_size

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
def fcNode(input_layer,weights,biases,use_relu=True):
	layer = tf.matmul(input_layer, weights) + biases
	layer = tf.nn.relu(layer)
	return layer


# def var_summaries(var):


shape1 = [4,4,1,2]
shape2 = [8,8,2,4]
shape3 = [8,8,4,4]
shape4 = [8,8,4,4]
fcshape1 = [10000,20]
fcshape2 = [20,2]
beta = 0.001

input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])
input_label = tf.placeholder(tf.float32,shape=[None,2],name="input_images_array")

# input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])

w1 = tf.get_variable("weightss_1", shape1,initializer=tf.random_normal_initializer(mean=0.1,stddev=0.07))
w2 = tf.get_variable("weightss_2", shape2,initializer=tf.random_normal_initializer(mean=0.1,stddev=0.07))
w3 = tf.get_variable("weightss_3", shape3,initializer=tf.random_normal_initializer(mean=0.1,stddev=0.07))
w4 = tf.get_variable("weightss_4", shape4,initializer=tf.random_normal_initializer(mean=0.1,stddev=0.07))
b1 = tf.get_variable("bias_1", [2],initializer=tf.constant_initializer(0.05))
b2 = tf.get_variable("bias_2", [4],initializer=tf.constant_initializer(0.05))
b3 = tf.get_variable("bias_3", [4],initializer=tf.constant_initializer(0.05))
b4 = tf.get_variable("bias_4", [4],initializer=tf.constant_initializer(0.05))
fcw1 = tf.get_variable("fcweightss_4", fcshape1,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fcw2 = tf.get_variable("fcweightss_5", fcshape2,initializer=tf.random_normal_initializer(mean=0.02,stddev=0.07))
fcb1 = tf.get_variable("fcbias_4", [20],initializer=tf.constant_initializer(0.05))
fcb2 = tf.get_variable("fcbias_5", [2],initializer=tf.constant_initializer(0.05))

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

layer_argmax = tf.math.argmax(layer_fc2,axis = 1)
label_argmax = tf.math.argmax(input_label,axis = 1)
equality = tf.equal(layer_argmax, label_argmax)
accuracy = tf.reduce_mean(tf.cast(equality,tf.float32))

# layer_argmax = tf.math.argmax(layer_fc2,axis = 1)
error = tf.math.square(layer_fc2 - input_label) + beta*tf.nn.l2_loss(fcw1)+ beta*tf.nn.l2_loss(fcw2)
cost = tf.reduce_mean(error)
cost_summary = tf.summary.scalar('training cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
init = tf.global_variables_initializer()
# tf.summary.scalar('Accuracy',accuracy)
# tf.summary.scalar('Cost',cost)
acc_summary = tf.summary.scalar('training accuracy',accuracy)
merged = tf.summary.merge_all()
summ_writer = tf.summary.FileWriter("./tmp/summaries_400")


model_name = 'model_trigger_1'
# model_name = 'trigger_model_8imgs'

saver = tf.train.Saver()
sess = tf.Session()



# saver.restore(sess,"C:\\Users\\user1\\Documents\\securesound\\tmp\\models\\train_accurate_models\\"+model_name+".ckpt")

try:
	saver.restore(sess,"C:\\Users\\user1\\Documents\\securesound\\tmp\\models\\test_accurate_models\\"+model_name+".ckpt")
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

prev_acc = 0
count = 0
for epoch in range(20):
	for x in training_set[:]:
		pos_img = cv2.imread(pos_train_dir+x[0],0)
		neg_img = cv2.imread(neg_train_dir+x[1],0)
		pos_img = cv2.resize(pos_img,(image_size,image_size))
		neg_img = cv2.resize(neg_img,(image_size,image_size))
		pos_img_flip = cv2.flip(pos_img,1)
		neg_img_flip = cv2.flip(neg_img,1)
		# out = out[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
		pos_img_300 = cv2.resize(pos_img[50:350,50:350],(image_size,image_size))
		neg_img_300 = cv2.resize(neg_img[50:350,50:350],(image_size,image_size))
		pos_img_300_flip = cv2.flip(pos_img_300,1)
		neg_img_300_flip = cv2.flip(neg_img_300,1)

		# cv2.imshow("img",pos_img)
		# cv2.imshow("img2",cv2.flip(pos_img,1))
		# # cv2.imshow("img3",pos_img[20:380,20:380])
		# cv2.imshow("img4",pos_img[50:350,50:350])
		# cv2.waitKey(0)
		# exit()
		img_array.append(pos_img/255)
		img_array.append(neg_img/255)
		img_array.append(pos_img_flip/255)
		img_array.append(neg_img_flip/255)
		img_array.append(pos_img_300/255)
		img_array.append(neg_img_300/255)
		img_array.append(pos_img_300_flip/255)
		img_array.append(neg_img_300_flip/255)
		# pos_img_300_flip = cv2.flip(pos_img_300,1)
		label_array = label_array + [[1.0,0.0],[0.0,1.0],[1.0,0.0],[0.0,1.0],[1.0,0.0],[0.0,1.0],[1.0,0.0],[0.0,1.0]]
		if len(img_array)>=batch_size or training_set.index(x) == (len(training_set)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			# output = sess.run(input_image,feed_dict={input_image:img_array})
			# print(output)
			output = sess.run(cost,feed_dict={input_image:img_array,input_label:label_array})
			# print(output)
			for x in range(2):
				sess.run(optimizer,feed_dict={input_image:img_array,input_label:label_array})
			summary,training_acc = sess.run([merged,accuracy],feed_dict={input_image:img_array,input_label:label_array})
			summ_writer.add_summary(summary,count)
			# print(output)
			img_array = []
			label_array = []
			# saver.save(sess,'./tmp/models/train_accurate_models/'+model_name+'.ckpt',global_step=count)
			saver.save(sess,'./tmp/models/train_accurate_models/'+model_name+'.ckpt')
			print("training_accuracy : " + str(training_acc))
			count = count + 1
			# training_acc = testNN(training_set)
	curr_acc = testNN(testing_set)
	if curr_acc > prev_acc:
		saver.save(sess,'./tmp/models/test_accurate_models/'+model_name+'.ckpt')
		# f = open("model_acc.txt",'a+')
		# f.write("TESTING ACC : " + str(curr_acc)+"\n")
		# f.close()
		prev_acc = curr_acc
	f = open("model_acc.txt",'a+')
	f.write("TRAINING ACC : " + str(training_acc) + " , TESTING ACC : " + str(prev_acc)+"\n")
	f.write("EPOCH COMPLETE\n")
	f.close()








