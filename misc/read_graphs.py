import tensorflow as tf
import cv2
import numpy as np
from google.protobuf import text_format
import os
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
# graphstr = tf.io.read_file("./frozen_auto/trained_auto.pbtxt")
f = open("./frozen_auto/trained_auto.pbtxt",'rb')
data = f.read()
print(type(data))
gdef = tf.GraphDef()
text_format.Merge(data,gdef)
tf.graph_util.import_graph_def(gdef)
# print(graphstr)
g = sess.graph
f.close()

# for i in g.get_operations():
#     print(i.name)
# exit()
# input_image = tf.placeholder(tf.float32,shape=[None,image_size,image_size],name="input_images_array")
# input_image_reshaped = tf.reshape(input_image, [-1, image_size, image_size, 1])
# input_imge_reshaped = tf.image.per_image_standardization(input_image_reshaped)
input_image = g.get_tensor_by_name("import/input_images_array:0")
encoder_result = g.get_tensor_by_name("import/conv2d_4/Relu:0")
# decoder_result = g.get_tensor_by_name("import/conv2d_transpose_3/Relu:0")
input_target = tf.placeholder(tf.float32,shape=[None,12],name="input_images_array")
lflat,num_features = flatten_layer(encoder_result)
fc1 = Layers.Dense(256,activation='relu')
fc1_out = fc1(lflat)
fc2 = Layers.Dense(128,activation='relu')
fc2_out = fc2(fc1_out)
fc3 = Layers.Dense(12,activation='relu')
label_output = fc3((fc2_out))

error = tf.math.square(tf.subtract(label_output,input_target))
cost = tf.reduce_mean(error)

cost_summary = tf.summary.scalar('training cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
init = tf.global_variables_initializer()
# tf.summary.scalar('Accuracy',accuracy)
# tf.summary.scalar('Cost',cost)
merged = tf.summary.merge_all()
summ_writer = tf.summary.FileWriter("./tmp/summaries_400")
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()


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
batch_size = 512
main_list = list(zip(pos_list,neg_list))
random.shuffle(main_list)
print(len(main_list))
training_set = main_list[:4512]
model_name = "autoconv_loc"
try:
	saver.restore(sess,"./tmp/models/train_accurate_models/"+model_name+".ckpt")
except:
	sess.run(init)
	print("MODEL NOT FOUND")
	pass

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


for epoch in range(10):
	for x in training_set[:]:
		pos_img = cv2.imread(pos_train_dir+x[0],0)
		neg_img = cv2.imread(neg_train_dir+x[1],0)
		ps,psf = getFlippedZoomImages(pos_img)
		ns,nsf = getFlippedZoomImages(neg_img)
		img_array = img_array + [ps,psf,ns,nsf]
		if x[0] in fem_list:
			fem_tag_img = cv2.imread(tag_dir_pathfem+x[0],0)
			ffpts,ffptsf = getFlippedZoomImages(fem_tag_img)
			fempts = getTagFeature(ffpts)
			femptsf = getTagFeature(ffptsf)
		else:
			fempts = [0.0,0.0,0.0,0.0]
			femptsf = [0.0,0.0,0.0,0.0]
		if x[0] in pel_list:
			pel_tag_img = cv2.imread(tag_dir_pathpel+x[0],0)
			pppts,ppptsf = getFlippedZoomImages(pel_tag_img)
			pelpts = getTagFeature(pppts)
			pelptsf = getTagFeature(ppptsf)
		else:
			pelpts = [0.0,0.0,0.0,0.0]
			pelptsf = [0.0,0.0,0.0,0.0]
		if x[0] in gen_list:
			gen_tag_img = cv2.imread(tag_dir_pathgen+x[0],0)
			ggpts,ggptsf = getFlippedZoomImages(gen_tag_img)
			genpts = getTagFeature(ggpts)
			genptsf = getTagFeature(ggptsf)
		else:
			genpts = [0.0,0.0,0.0,0.0]
			genptsf = [0.0,0.0,0.0,0.0]
		target_array = target_array + [fempts+pelpts+genpts,femptsf+pelptsf+genptsf,[0.0,0.0,0.0,0.0]*3,[0.0,0.0,0.0,0.0]*3]
		if len(img_array)>=batch_size or training_set.index(x) == (len(training_set)-1):
			# print(label_array)
			img_array = np.asarray(img_array)
			c = sess.run(cost,feed_dict={input_image:img_array,input_target:target_array})
			print(c)
			sess.run(optimizer,feed_dict={input_image:img_array,input_target:target_array})
			# cv2.imshow("img",img_array[0])
			# cv2.imshow("img2",output[0])
			# cv2.imshow("img3",img_array[2])
			# cv2.imshow("img4",output[2])
			# cv2.waitKey(0)
			# exit()
			saver.save(sess,'./tmp/models/train_accurate_models/'+model_name+'.ckpt')
			img_array = []
			target_array = []
			
			# exit()
		# break
	
	# break


