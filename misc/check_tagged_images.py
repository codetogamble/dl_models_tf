import cv2
import os
import csv
import numpy as np

tag_dir = "./home_tag_1/"
vid_file = "dvr_20190610_1050"
vid_dir = "./video_data/"

tag = "Femur Bone"

print(os.listdir(tag_dir+vid_file))

vidCapture = cv2.VideoCapture(vid_dir+vid_file+".mp4")

imagesList = os.listdir(tag_dir+vid_file+"/"+tag) #+ os.listdir(tag_dir+vid_file+"/"+tags[1])
# print(imagesList)

frame_index = vidCapture.get(cv2.CAP_PROP_POS_FRAMES)
print(frame_index)

# vidCapture.set(cv2.CAP_PROP_POS_FRAMES,2679)


# for imgfile in imagesList:
#     print(imgfile)
#     tagged_image = cv2.imread(tag_dir+vid_file+"/"+tag+"/"+imgfile,0)
#     frame_index = int(imgfile.split("_")[-1].replace(".jpg","")) - 2
#     vidCapture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
#     print(frame_index)
#     ret,frame = vidCapture.read()
#     gray = cv2.cvtColor(frame[60:],cv2.COLOR_BGR2GRAY)
#     resize_tagged = cv2.resize(tagged_image,(gray.shape[1],gray.shape[0]))
#     cv2.imshow("img",(gray/2+resize_tagged/2)/255)
#     cv2.waitKey(0)
#     # break
# vidCapture.release()

# exit()

csv_list = ["Ganna.csv","Shubham.csv"]#INPUT CSV FILES WITH LABELS
video_list = []

for f in csv_list[1:]:
    filepath = "./tags_csv/" + f
    f = open(filepath)
    rows = csv.reader(f, delimiter=',', quotechar='"')
    for r in rows:
        print(r)
        imgpath = r[0]
        main_category = r[1]
        tag_type = r[7]
        video_name = imgpath.split('/')[2]
        imgname = imgpath.split('/')[3]
        if main_category == "Obstetric" and tag_type == "Genital" and video_name == "dvr_20190610_1050":#ONLY FOR REGION TAGGED IMAGES
        # if main_category == "Obstetric" and tag_type == "Femur Bone" and video_name == "dvr_20190610_1050":#ONLY FOR REGION TAGGED IMAGES
        # if main_category == "Obstetric" and tag_type == "Pelvic Girdle" and video_name == "dvr_20190610_1050":#ONLY FOR REGION TAGGED IMAGES
            scale = (100)/(100-float(r[6]))
            x = int(float(r[2])*scale)
            y = int(float(r[3])*scale)
            width = int(float(r[4])*scale)
            height = int(float(r[5])*scale)
            frame_number = int(imgname.split('_')[-1].replace(".jpg",""))
            vidpath = vid_dir+video_name+".mp4"
            if not os.path.exists(tag_dir+"/"+video_name):#CREATE DIR IF NOT PRESENT
                os.mkdir(tag_dir+"/"+video_name) 
            if not os.path.exists(tag_dir+"/"+video_name+"/"+tag_type):
                os.mkdir(tag_dir+"/"+video_name+"/"+tag_type)
            vidreader = cv2.VideoCapture(vidpath)
            totalFrames = vidreader.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_number>=0 and frame_number <= totalFrames:
                vidreader.set(cv2.CAP_PROP_POS_FRAMES,frame_number)#SET CURRENT FRAME
                ret,frame = vidreader.read()
                orig_image = frame[60:]#COMPENSATION FOR ORIGINAL OFFSET TOP CROP FOR ANONYMIZATION
                orig_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
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
                # cv2.imshow("orig_image",orig_image)
                # cv2.imwrite(tag_dir+video_name+"/"+tag_type+"/"+imgname,tag_image*255)
                # cv2.imshow("tagged",orig_image/510+resize_tag_image/2)
                orig_rgb_image = cv2.cvtColor(orig_image,cv2.COLOR_GRAY2BGR)

                cv2.rectangle(orig_rgb_image,(x,y),(x+width,y+height),(0,255,0),2)
                cv2.imshow("tagged",orig_rgb_image)
                cv2.waitKey(0)
                vidreader.release()
            # break
    f.close()
    break