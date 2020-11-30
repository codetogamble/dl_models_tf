import cv2
import numpy as np
import tensorflow as tf
import os

pos_train_dir = "./training_data_2/roi_pos/"
neg_train_dir = "./training_data_2/roi_neg/"
posgen_dir = "./training_data_2/tag_pos/roi_posgen"
posfem_dir = "./training_data_2/tag_pos/roi_posfem"
pospel_dir = "./training_data_2/tag_pos/roi_pospel"

posgen_data = os.listdir(posgen_dir)
posfem_data = os.listdir(posfem_dir)
pospel_data = os.listdir(pospel_dir)

print(posgen_data)
# exit()

for f in os.listdir(pos_train_dir):
    print(f)
    img = cv2.imread(pos_train_dir+f)
    print(img.shape)
    # exit()
    img = cv2.resize(img,(400,400))
    tag_img = np.zeros(img.shape,dtype = np.uint8)
    print(tag_img.shape)
    # exit()
    if f in posgen_data:
        print("FOUND GENITAL")
        gen_img  = cv2.imread(posgen_dir+"/"+f,0)
        gen_img = gen_img/255
        if gen_img is not None:
            if np.amax(gen_img)>0.0:
                gen_img = gen_img/np.amax(gen_img)
                bool_row = np.where(gen_img==1.0)
            #         print(bool_row)
                first_pt = [int(np.amin(bool_row[0])),int(np.amin(bool_row[1]))]
                second_pt = [int(np.amax(bool_row[0])),int(np.amax(bool_row[1]))]
            #         print(first_pt)
            #         print(second_pt)
                feature_array = first_pt+second_pt
                print(first_pt)
                print(second_pt)
                print(type(img))
                colorr = (0,255,0)
                print(type(colorr))
                cv2.rectangle(img,(int(first_pt[0]),int(first_pt[1])),(int(second_pt[0]),int(second_pt[1])),colorr,2)
            else:
                feature_array = [0,0,0,0]
        else:
            feature_array = [0,0,0,0]
        
      # blank_img = np.zeros((400,400),dtype=np.uint8)
      # blank_img[int(feature_array[0]*400):int(feature_array[2]*400),int(feature_array[1]*400):int(feature_array[3]*400)] = 255
        # img[:,:,2] = (img[:,:,2] + tag_img[:,:,2])/2
    if f in posfem_data:
        print("FOUND FEMUR")
        fem_img  = cv2.imread(posfem_dir+"/"+f,0)
        fem_img = fem_img/255
        if fem_img is not None:
            if np.amax(fem_img)>0.0:
                fem_img = fem_img/np.amax(fem_img)
                bool_row = np.where(fem_img==1.0)
            #         print(bool_row)
                first_pt = [np.amin(bool_row[0]),np.amin(bool_row[1])]
                second_pt = [np.amax(bool_row[0]),np.amax(bool_row[1])]
            #         print(first_pt)
            #         print(second_pt)
                feature_array = first_pt+second_pt
                cv2.rectangle(img,(int(first_pt[0]),int(first_pt[1])),(int(second_pt[0]),int(second_pt[1])),(0,0,255),2)
            else:
                feature_array = [0,0,0,0]
        else:
            feature_array = [0,0,0,0]
        

    if f in pospel_data:
        print("FOUND PELVIC")
        pel_img  = cv2.imread(pospel_dir+"/"+f,0)
        pel_img = pel_img/255
        if pel_img is not None:
            if np.amax(pel_img)>0.0:
                pel_img = pel_img/np.amax(pel_img)
                bool_row = np.where(pel_img==1.0)
            #         print(bool_row)
                first_pt = [np.amin(bool_row[0]),np.amin(bool_row[1])]
                second_pt = [np.amax(bool_row[0]),np.amax(bool_row[1])]
            #         print(first_pt)
            #         print(second_pt)
                feature_array = first_pt+second_pt
                cv2.rectangle(img,(int(first_pt[0]),int(first_pt[1])),(int(second_pt[0]),int(second_pt[1])),(255,0,0),2)
            else:
                feature_array = [0,0,0,0]
        else:
            feature_array = [0,0,0,0]
    cv2.imshow("img",img)
    # cv2.imshow("tag",tag_img)
    cv2.waitKey(10)
    # break

