import os
import cv2
import time
import datetime
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

# vid_file = "dvr_20190610_1050"
# videos_list = ["dvr_20190610_1050","dvr_20190610_1142","dvr_20190610_1158","dvr_20190610_1245","dvr_20190611_1222"]
videos_list = ["dvr_20190610_1050","dvr_20190610_1142","dvr_20190610_1158","dvr_20190610_1245","dvr_20190611_1222","dvr_20190610_1328","dvr_20190610_1424","dvr_20190610_1457","dvr_20190610_1548","dvr_20190610_1555","dvr_20190611_1029","dvr_20190611_1043","dvr_20190611_1438","dvr_20190610_1216"]
train_dir = "./training_tags_3/"
tag_dir = "./home_tag_2/"
savetrain_dir = "./training_data_4/"
for vid_file in videos_list:
    f_neg = open(train_dir+vid_file+"/roi_neg.txt",'r')
    f_pos = open(train_dir+vid_file+"/roi_pos.txt",'r')
    f_posgen = open(train_dir+vid_file+ "/roi_posgen.txt",'r')
    f_posfem = open(train_dir+vid_file+ "/roi_posfem.txt",'r')
    f_pospel = open(train_dir+vid_file+ "/roi_pospel.txt",'r')

    neg_data = f_neg.read()
    neg_data = neg_data.split("\n")
    # print(len(neg_data))
    f_neg.close()

    pos_data = f_pos.read()
    pos_data = pos_data.split("\n")
    # print(len(pos_data))
    f_pos.close()
    
    posgen_data = f_posgen.read()
    posgen_data = posgen_data.split("\n")
    # print(len(neg_data))
    f_posgen.close()

    posfem_data = f_posfem.read()
    posfem_data = posfem_data.split("\n")
    # print(len(pos_data))
    f_posfem.close()

    pospel_data = f_pospel.read()
    pospel_data = pospel_data.split("\n")
    # print(len(neg_data))
    f_pospel.close()
    
    if not os.path.exists(savetrain_dir + "roi_neg/"):
        os.mkdir(savetrain_dir + "roi_neg/")
    if not os.path.exists(savetrain_dir + "roi_pos/"):
        os.mkdir(savetrain_dir + "roi_pos/")
    if not os.path.exists(savetrain_dir + "tag_pos/"):
        os.mkdir(savetrain_dir + "tag_pos/")
    if not os.path.exists(savetrain_dir + "tag_pos/roi_posgen/"):
        os.mkdir(savetrain_dir + "tag_pos/roi_posgen/")
    if not os.path.exists(savetrain_dir + "tag_pos/roi_posfem/"):
        os.mkdir(savetrain_dir + "tag_pos/roi_posfem/")
    if not os.path.exists(savetrain_dir + "tag_pos/roi_pospel/"):
        os.mkdir(savetrain_dir + "tag_pos/roi_pospel/")

    videoreader = cv2.VideoCapture("./video_data/"+ vid_file + ".mp4")
# CAP_PROP_POS_FRAMES
    tt,ff = videoreader.read()
    print(ff.shape)
    ff = ff[60:]
    prev_frame = np.zeros((int(ff.shape[0]/16),int(ff.shape[1]/16)),dtype=np.uint8)
    count = 0
    while(True):
        ret,frame = videoreader.read()
        if (str(count) in neg_data or str(count) in pos_data) and ret == True:
            # print("FOUND")
            gray2 = cv2.cvtColor(frame[60:],cv2.COLOR_BGR2GRAY)
            main_image_shape = gray2.shape
            # print(main_image_shape)
            # exit()
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
            if str(count) in neg_data:
                cv2.imwrite(savetrain_dir + "roi_neg/"+str(vid_file)+"_"+str(count)+"_neg.jpg",out)
            if str(count) in pos_data:
                cv2.imwrite(savetrain_dir + "roi_pos/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out)
                tagimage_dir = tag_dir + vid_file
                image_name = vid_file+"_"+str(count)+".jpg"
                if str(count) in posgen_data:
                    print(tagimage_dir+"/Genital/"+image_name,0)
                    if os.path.exists(tagimage_dir+"/Probable Genital/"+image_name):
                        tagged_img = cv2.imread(tagimage_dir+"/Probable Genital/"+image_name,0)
                        tagged_img = cv2.resize(tagged_img,(main_image_shape[1],main_image_shape[0]))
                    elif os.path.exists(tagimage_dir+"/Genital/"+image_name):
                        tagged_img = cv2.imread(tagimage_dir+"/Genital/"+image_name,0)
                        tagged_img = cv2.resize(tagged_img,(main_image_shape[1],main_image_shape[0]))
                    else:
                        tagged_img = None
                    if tagged_img is not None:
                        print("FOUND")
                        print(tagimage_dir+"/Genital/"+image_name,0)
                        tagged_img = cv2.resize(tagged_img,(main_image_shape[1],main_image_shape[0]))
                        if splitimg_bool:
                            left_split = tagged_img[:,:int(tagged_img.shape[1]/2)]
                            right_split = tagged_img[:,int(tagged_img.shape[1]/2):]
                            # print("SUM LEFT")
                            # print(np.sum(left_split))
                            # print("SUM RIGHT")
                            # print(np.sum(right_split))
                            # print(image_name)
                            if rightsplit_bool:
                                label_image = right_split
                            else:
                                label_image = left_split
                            if np.sum(left_split)>np.sum(right_split) and rightsplit_bool:
                                label_image = left_split
                                out2 = gray2[:,:int(gray2.shape[1]/2)]
                                out2 = out2[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                                out2 = cv2.resize(out2,(400,400))
                                cv2.imwrite(savetrain_dir + "roi_pos/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out2)
                            else:
                                label_image = right_split
                                out2 = gray2[:,int(gray2.shape[1]/2):]
                                out2 = out2[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                                out2 = cv2.resize(out2,(400,400))
                                cv2.imwrite(savetrain_dir + "roi_pos/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out2)
                        else:
                            label_image = tagged_img
                        label_image = label_image[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                        label_image = cv2.resize(label_image,(400,400))
                        cv2.imwrite(savetrain_dir + "/tag_pos/roi_posgen/"+str(vid_file)+"_"+str(count)+"_pos.jpg",label_image)
                if str(count) in posfem_data:
                    if os.path.exists(tagimage_dir+"/Femur Bone/"+image_name):
                        tagged_img = cv2.imread(tagimage_dir+"/Femur Bone/"+image_name,0)
                        tagged_img = cv2.resize(tagged_img,(main_image_shape[1],main_image_shape[0]))
                    else:
                        # print(count)
                        # print(vid_file)
                        # print(tagimage_dir+"/Probable Genital/"+image_name)
                        # print("NOT FOUND")
                        tagged_img = None
                    if tagged_img is not None:
                        print("FOUND")
                        print(tagimage_dir+"/Femur Bone/"+image_name,0)
                        if splitimg_bool:
                            left_split = tagged_img[:,:int(tagged_img.shape[1]/2)]
                            right_split = tagged_img[:,int(tagged_img.shape[1]/2):]
                            # if np.sum(left_split)>np.sum(right_split):
                            #     label_image = left_split
                            # else:
                            #     label_image = right_split
                            if rightsplit_bool:
                                label_image = right_split
                            else:
                                label_image = left_split
                            if np.sum(left_split)>np.sum(right_split) and rightsplit_bool:
                                label_image = left_split
                                out2 = gray2[:,:int(gray2.shape[1]/2)]
                                out2 = out2[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                                out2 = cv2.resize(out2,(400,400))
                                cv2.imwrite(savetrain_dir + "roi_pos/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out2)
                            else:
                                label_image = right_split
                                out2 = gray2[:,int(gray2.shape[1]/2):]
                                out2 = out2[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                                out2 = cv2.resize(out2,(400,400))
                                cv2.imwrite(savetrain_dir + "roi_pos/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out2)
                        else:
                            label_image = tagged_img
                        label_image = label_image[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                        label_image = cv2.resize(label_image,(400,400))
                        cv2.imwrite(savetrain_dir + "/tag_pos/roi_posfem/"+str(vid_file)+"_"+str(count)+"_pos.jpg",label_image)
                        
                        # cv2.imwrite(savetrain_dir + "/tag_pos/roi_posfem/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out)
                if str(count) in pospel_data:
                    if os.path.exists(tagimage_dir+"/Pelvic Girdle/"+image_name):
                        tagged_img = cv2.imread(tagimage_dir+"/Pelvic Girdle/"+image_name,0)
                    else:
                        # print(count)
                        # print(vid_file)
                        # print(tagimage_dir+"/Probable Genital/"+image_name)
                        # print("NOT FOUND")
                        tagged_img = None
                    if tagged_img is not None:
                        print("FOUND")
                        print(tagimage_dir+"/Pelvic Girdle/"+image_name,0)
                        tagged_img = cv2.resize(tagged_img,(main_image_shape[1],main_image_shape[0]))
                        if splitimg_bool:
                            left_split = tagged_img[:,:int(tagged_img.shape[1]/2)]
                            right_split = tagged_img[:,int(tagged_img.shape[1]/2):]
                            # if np.sum(left_split)>np.sum(right_split):
                            #     label_image = left_split
                            # else:
                            #     label_image = right_split
                            if rightsplit_bool:
                                label_image = right_split
                            else:
                                label_image = left_split
                            
                            if np.sum(left_split)>np.sum(right_split) and rightsplit_bool:
                                label_image = left_split
                                out2 = gray2[:,:int(gray2.shape[1]/2)]
                                out2 = out2[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                                out2 = cv2.resize(out2,(400,400))
                                cv2.imwrite(savetrain_dir + "roi_pos/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out2)
                            else:
                                label_image = right_split
                                out2 = gray2[:,int(gray2.shape[1]/2):]
                                out2 = out2[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                                out2 = cv2.resize(out2,(400,400))
                                cv2.imwrite(savetrain_dir + "roi_pos/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out2)
                        else:
                            label_image = tagged_img
                        label_image = label_image[vert_gap:vert_gap+wsz,hori_gap:hori_gap+wsz]
                        label_image = cv2.resize(label_image,(400,400))
                        cv2.imwrite(savetrain_dir + "/tag_pos/roi_pospel/"+str(vid_file)+"_"+str(count)+"_pos.jpg",label_image)
                    # cv2.imwrite(savetrain_dir + "/tag_pos/roi_pospel/"+str(vid_file)+"_"+str(count)+"_pos.jpg",out)
        elif ret == False:
            videoreader.release()
            break

        count = count + 1
    
    # break


# for x in neg_data:
#     if x!= '':
#         get_frame = int(x)
#         # print(get_frame)
#         videoreader.set(cv2.CAP_PROP_POS_FRAMES,get_frame)
#         ret,frame = videoreader.read()
        
        # gray[int(gray.shape[0]/2),:] = 255
        # print(out.shape)

        # cv2.imshow("frame",out)
        # cv2.imshow("frame2",gray)
        # cv2.imshow("img",thresh)
        # cv2.waitKey(10)
        # prev_frame = gray
        # break