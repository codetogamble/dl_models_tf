import os
import cv2
import time
import datetime
import numpy as np
import sys


print(sys.argv)
file_path = sys.argv[1]
f = file_path.split("/")[-1][:-4]
cap = cv2.VideoCapture(file_path)
fps = round(cap.get(cv2.CAP_PROP_FPS))
print(fps)

#time conversion to frame count 
start_time = sys.argv[2]
end_time = sys.argv[3]

x = time.strptime(start_time,'%M:%S')
seconds_start = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
# print(seconds_start)
y = time.strptime(end_time,'%M:%S')
seconds_end = datetime.timedelta(hours=y.tm_hour,minutes=y.tm_min,seconds=y.tm_sec).total_seconds()
# print(seconds_start)
start_count = fps*seconds_start
end_count = fps*seconds_end
print(start_count)
print(end_count)

#START FOR PROCESSING VIDEO
ret,frame = cap.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
pixels_no = gray.shape[0]*gray.shape[1]
prev_frame = np.zeros(gray.shape)
count = 1
while(True):
    ret,frame = cap.read()
    if ret==True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not os.path.exists("./subimages/"+f):#FOLDER DESTINATION IS NAMED AFTER VIDEO NAME
            os.mkdir("./subimages/"+f)
        if count >=start_count and count <=end_count:#name of frames is vidname_framenumber.jpg
            cv2.imwrite("./subimages/"+f+"/"+f+"_"+str(count)+".jpg",gray[60:,:])#TOP CROP IS FOR ANONYMIZATION
        if count > end_count:
            break
        count = count + 1
    else:
        break
#Release video for efficient memory use
cap.release()

    

