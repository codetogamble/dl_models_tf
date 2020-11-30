import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from matplotlib import pyplot as plt
import sys

video_dir = "./video_data/"
tag_dir = "./home_tag_1/"
# csv_list = ["Ganna.csv","Shubham.csv"]#INPUT CSV FILES WITH LABELS
csv_list = ["ful.csv"]#INPUT CSV FILES WITH LABELS
video_list = []
orig_image = None

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
            if orig_image is None:
                vidreader = cv2.VideoCapture(vidpath)
                vidreader.set(cv2.CAP_PROP_POS_FRAMES,frame_number)#SET CURRENT FRAME
                ret,frame = vidreader.read()
                orig_image = frame[60:]#COMPENSATION FOR ORIGINAL OFFSET TOP CROP FOR ANONYMIZATION
                orig_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
                totalFrames = vidreader.get(cv2.CAP_PROP_FRAME_COUNT)
                vidreader.release()
            if not os.path.exists(tag_dir+"/"+video_name):#CREATE DIR IF NOT PRESENT
                os.mkdir(tag_dir+"/"+video_name) 
            if not os.path.exists(tag_dir+"/"+video_name+"/"+tag_type):
                os.mkdir(tag_dir+"/"+video_name+"/"+tag_type)
            # vidreader = cv2.VideoCapture(vidpath)
            # totalFrames = vidreader.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_number>=0 and frame_number <= totalFrames:
                # vidreader.set(cv2.CAP_PROP_POS_FRAMES,frame_number)#SET CURRENT FRAME
                # ret,frame = vidreader.read()
                # orig_image = frame[60:]#COMPENSATION FOR ORIGINAL OFFSET TOP CROP FOR ANONYMIZATION
                # orig_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
                #TRY FOR MORE ACCURATE LOCATION TAGGING
                # print(orig_image.dtype)
                # orig_image = orig_image/np.amax(orig_image)
                # orig_image = (orig_image*255).astype(np.uint8)
                # ret,thresh = cv2.threshold(orig_image,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                tag_image = np.zeros((orig_image.shape[0],orig_image.shape[1]),dtype=np.float32)
                tag_image[y:y+height,x:x+width] = 1        
                tag_image = cv2.resize(tag_image,(int(orig_image.shape[1]/8),int(orig_image.shape[0]/8)))
                resize_tag_image = cv2.resize(tag_image,(int(orig_image.shape[1]),int(orig_image.shape[0])))
                #FOLDER STRUCTURE videoname/tagtype/videoname_framenumber
                cv2.imwrite(tag_dir+video_name+"/"+tag_type+"/"+imgname,tag_image*255)
                # vidreader.release()
            # break
    f.close()
    # break