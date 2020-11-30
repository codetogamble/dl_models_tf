import tensorflow as tf
import numpy as np
import os
import cv2

pos_train_dir = "C:/Users/user1/Documents/securesound/training_data_2/roi_pos/"
neg_train_dir = "C:/Users/user1/Documents/securesound/training_data_2/roi_neg/"

model_name = 'model_400_reg2'
sess = tf.Session()
new_saver = tf.train.import_meta_graph('./tmp/models/test_accurate_models/'+model_name+'.ckpt.meta')
new_saver.restore(sess, './tmp/models/test_accurate_models/'+model_name+'.ckpt')
test_file = open('./model_400_reg2.txt')
test_files = test_file.readlines()
test_file.close()
# print(os.path.exists(pos_train_dir+test_files[0]))
tp = 0
tn = 0
fp = 0
fn = 0
total = 0
for ind in range(len(test_files)):
    curr_file = test_files[ind].replace("\n","")
    tag = curr_file.split("_")[-1].replace(".jpg","")
    # print(curr_file)
    # print(tag)
    if "pos" in tag:
        test_img = cv2.imread(pos_train_dir+curr_file,0)
        print(pos_train_dir+curr_file)
        test_img = test_img/255
        output = sess.run("layer_fc2:0",feed_dict={"input_images_array:0":[test_img]})
        # print(output)
        if output[0][0]>output[0][1]:
            tp = tp + 1
        else:
            fn = fn + 1
    else:
        test_img = cv2.imread(neg_train_dir+curr_file,0)
        test_img = test_img/255
        output = sess.run("layer_fc2:0",feed_dict={"input_images_array:0":[test_img]})
        if output[0][0]<output[0][1]:
            tn = tn + 1
        else:
            fp = fp + 1
    total = total + 1

print(tp)
print(tn)
print(fp)
print(fn)
print(total)
