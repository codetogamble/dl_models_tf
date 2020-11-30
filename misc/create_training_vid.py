import tensorflow as tf
import numpy as np
import cv2
import os
import csv
import sys

video_dir = "./video_data/"
tag_dir = "./home_tag_2/"
# csv_list = ["Ganna.csv","Shubham.csv"]#INPUT CSV FILES WITH LABELS
csv_list = ["ful.csv","shubham.csv"]#INPUT CSV FILES WITH LABELS
video_list = []
orig_image = None

vid_dict = {}
vid_ftpair = {}
for f in csv_list:
    filepath = "./tags_csv/" + f
    f = open(filepath)
    rows = csv.reader(f, delimiter=',', quotechar='"')
    for r in rows:
        print(r)
        imgpath = r[1]
        main_category = r[2]
        tag_type = r[8]
        if main_category == "Obstetric" and tag_type != "NULL":#ONLY FOR REGION TAGGED IMAGES
            scale = (100)/(100-float(r[7]))
            x = int(float(r[3])*scale)
            y = int(float(r[4])*scale)
            width = int(float(r[5])*scale)
            height = int(float(r[6])*scale)
            video_name = imgpath.split('/')[2]
            imgname = imgpath.split('/')[3]
            frame_number = int(imgname.split('_')[-1].replace(".jpg",""))
            vidpath = video_dir+video_name+".mp4"
            vidn_fr = video_name +"_"+ str(frame_number)
            if video_name not in vid_dict.keys():
                vid_dict[video_name] = [frame_number]
            else:
                vid_dict[video_name].append(frame_number)
            if vidn_fr not in vid_ftpair.keys():
                vid_ftpair[vidn_fr]=[[tag_type,[x,y,width,height]]]
            else:
                vid_ftpair[vidn_fr].append([tag_type,[x,y,width,height]])
            # vidreader = cv2.VideoCapture(vidpath)
            # vidreader.set(cv2.CAP_PROP_POS_FRAMES,frame_number)#SET CURRENT FRAME
            # ret,frame = vidreader.read()
            # orig_image = frame[60:]#COMPENSATION FOR ORIGINAL OFFSET TOP CROP FOR ANONYMIZATION
            # orig_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
            # totalFrames = vidreader.get(cv2.CAP_PROP_FRAME_COUNT)
            # vidreader.release()
            # break
    f.close()
    # break

def getFemPelGenTagImage(tgdata,main_shape):
    fem_img = np.zeros(main_shape,dtype=np.float32)
    pel_img = np.zeros(main_shape,dtype=np.float32)
    gen_img = np.zeros(main_shape,dtype=np.float32)
    if tgdata == None:
        return fem_img,pel_img,gen_img    
    for x in tgdata:
        tag_type = x[0]
        tag_loc = x[1]
        if tag_type == "Genital" or tag_type == "Probable Genital":
            gen_img[tag_loc[1]:tag_loc[1]+tag_loc[3],tag_loc[0]:tag_loc[0]+tag_loc[2]] = 1.0
        elif tag_type == "Femur Bone":
            fem_img[tag_loc[1]:tag_loc[1]+tag_loc[3],tag_loc[0]:tag_loc[0]+tag_loc[2]] = 1.0
        elif tag_type == "Pelvic Bone":
            pel_img[tag_loc[1]:tag_loc[1]+tag_loc[3],tag_loc[0]:tag_loc[0]+tag_loc[2]] = 1.0
    return fem_img,pel_img,gen_img

print(vid_dict.keys())
for k in vid_dict.keys():
    print(k)
    vidpath = video_dir+k+".mp4"
    vidreader = cv2.VideoCapture(vidpath)
    sorted_frames = sorted(set(vid_dict[k]))
    prev_frame = sorted_frames[0]
    start_frame = sorted_frames[0]
    frame_pairs = []
    for x in sorted_frames[1:]:
        if x - prev_frame >= 60:
            frame_pairs.append([start_frame,prev_frame])
            print(prev_frame-start_frame)
            start_frame = x
            prev_frame = x
        else:
            prev_frame = x
    frame_pairs.append([start_frame,prev_frame])
    print(prev_frame-start_frame)
    print(frame_pairs)
    for pair in frame_pairs:
        stf = pair[0]-120
        enf = pair[1]+60
        vidreader.set(cv2.CAP_PROP_POS_FRAMES,stf)#SET CURRENT FRAME
        curr_frame = stf
        while curr_frame <= enf:
            curr_frame = vidreader.get(cv2.CAP_PROP_POS_FRAMES)
            ret,frame = vidreader.read()
            orig_image = frame[60:]#COMPENSATION FOR ORIGINAL OFFSET TOP CROP FOR ANONYMIZATION
            orig_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
            main_shape = orig_image.shape
            keyname = k+"_"+str(int(curr_frame))
            keyname_pre = k+"_"+str(int(curr_frame)-1)
            keyname_post = k+"_"+str(int(curr_frame)+1)
            if ret:
                print(keyname)
                tgdata = vid_ftpair.get(keyname)
                tgdata_pre = vid_ftpair.get(keyname_pre)
                tgdata_post = vid_ftpair.get(keyname_post)
                if tgdata != None:
                    print(tgdata)
                    fem_img,pel_img,gen_img = getFemPelGenTagImage(tgdata,main_shape)
                    fem_preimg,pel_preimg,gen_preimg = getFemPelGenTagImage(tgdata_pre,main_shape)
                    fem_postimg,pel_postimg,gen_postimg = getFemPelGenTagImage(tgdata_post,main_shape)
                    tag_fem = cv2.add(fem_img,fem_preimg)
                    tag_fem = cv2.add(tag_fem,fem_postimg)
                    tag_pel = cv2.add(pel_img,pel_preimg)
                    tag_pel = cv2.add(tag_pel,pel_postimg)
                    tag_gen = cv2.add(gen_img,gen_preimg)
                    tag_gen = cv2.add(tag_gen,gen_postimg)
                    cv2.imshow("tag_fem",tag_fem)
                    cv2.imshow("tag_pel",tag_pel)
                    cv2.imshow("tag_gen",tag_gen)
                    cv2.imshow("img",orig_image)
                    cv2.waitKey(0)     
            else:
                break
    vidreader.release()
    break