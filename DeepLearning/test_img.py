#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2
import numpy as np   
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import ImageFont, ImageDraw, Image

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# 각 파일 path
protoFile = "./pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./pose_iter_160000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Face detection XML load and trained model loading
emotion_classifier = load_model('./model_r.h5', compile=False)
EMOTIONS = ["goddess","tree", "warrior2"]

# 이미지 읽어오기
image = cv2.imread("./goddess.jpg")


# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = image.shape

# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

# network에 넣어주기
net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()

# output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
H = output.shape[2]
W = output.shape[3]

# 키포인트 검출시 이미지에 그려줌
points = []
for i in range(0,15):
    # 해당 신체부위 신뢰도 얻음.
    probMap = output[0, i, :, :]
 
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
    if prob > 0.1 :    
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else :
        points.append(None)
        
# 이미지 복사
imageCopy = image

# 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
for pair in POSE_PAIRS:
    partA = pair[0]             # Head
    partA = BODY_PARTS[partA]   # 0
    partB = pair[1]             # Neck
    partB = BODY_PARTS[partB]   # 1
    
    #print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)

#cv2.imshow("Output-Keypoints",imageCopy)
#cv2.waitKey(0)

#관절 뽑은 사진 넘겨줌
frame = imageCopy
    
# Resize the image to 48x48 for neural network
roi = cv2.resize(frame, (71, 71), interpolation = cv2.INTER_AREA)
roi = img_to_array(roi)
roi = np.expand_dims(roi, axis=0)
roi = preprocess_input(roi)
        
# Emotion predict
prediction = emotion_classifier.predict(roi)[0]
emotion_probability = np.max(prediction)
predicted_class = EMOTIONS[prediction.argmax()]

if predicted_class == 0:
    pose = "goddess"
    
elif predicted_class == 1:
    pose = "tree"    
    
elif predicted_class == 2:
    pose = "warrior2"
    
elif predicted_class == 3:
    pose = "false"
    
fontpath = "font/gulim.ttc"
font1 = ImageFont.truetype(fontpath, 50)
frame_pil = Image.fromarray(frame)
draw = ImageDraw.Draw(frame_pil)
draw.text((30, 30), pose, font=font1, fill=(0, 0, 255, 3))
frame = np.array(frame_pil)

print(pose)
    
# Open two windows
## Display image ("Emotion Recognition")
## Display probabilities of emotion
cv2.imshow('your pose', frame)
cv2.waitKey(0)

# Clear program and close windows
cv2.destroyAllWindows()


# In[24]:


#키포인트 좌표값 출력
import cv2
import numpy as np   
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import ImageFont, ImageDraw, Image

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# 각 파일 path
protoFile = "./pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./pose_iter_160000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Face detection XML load and trained model loading
#emotion_classifier = load_model('./model_final.h5', compile=False)
EMOTIONS = ["goddess","tree", "warrior2"]

# 이미지 읽어오기
image = cv2.imread("./warrior2.jpg")


# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = image.shape

# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

# network에 넣어주기
net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()

# output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
H = output.shape[2]
W = output.shape[3]

# 키포인트 검출시 이미지에 그려줌
points = []
    
for i in range(0,15):
    # 해당 신체부위 신뢰도 얻음.
    probMap = output[0, i, :, :]
 
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H
    
    print(x,y)


# In[ ]:


#csv파일 만들기
from PIL import Image
import numpy as np 
import sys
import csv
import os
import cv2

f_output = open('Test.csv','w', newline = '')
csv_writer = csv.writer(f_output)
csv_writer.writerow([ 'video','frame_name', 'pixel'])
        
data_path = './data'

Frame_folder = os.listdir(data_path)
for folder in Frame_folder:
    sub_folder_list = os.listdir(os.path.join(data_path, folder))
    for sub_folder in sub_folder_list:
        person_video_list = os.listdir(os.path.join(data_path, folder,sub_folder))
        for video in person_video_list:
            frame_list = os.listdir(os.path.join(data_path, folder,sub_folder,video))
            for frame in frame_list:
                img = Image.open(os.path.join(data_path, folder,sub_folder,video,frame))
                print(img)

                img_grey = img.convert('L') 
                value = np.asarray(img_grey.getdata(),dtype=np.float64).reshape((img_grey.size[1],img_grey.size[0]))
                value = value.flatten()
                csv_writer.writerow([video,frame,value])       
 
f_output.close()


# In[12]:


#좌표값으로 맞추기
import cv2
import numpy as np   
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import ImageFont, ImageDraw, Image

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# 각 파일 path
protoFile = "./pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./pose_iter_160000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Face detection XML load and trained model loading
emotion_classifier = load_model('./model_xy.h5', compile=False)
EMOTIONS = ["goddess","tree","warrior2","false"]

# 이미지 읽어오기
image = cv2.imread("./warrior2.jpg")


# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = image.shape

# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

# network에 넣어주기
net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()

# output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
H = output.shape[2]
W = output.shape[3]

# 키포인트 검출시 이미지에 그려줌
points = []
for i in range(15):
    # 해당 신체부위 신뢰도 얻음.
    probMap = output[0, i, :, :]
 
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
    if prob > 0.1 :    
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.extend((int(x), int(y)))
    else :
        points.extend(int(0))
        
print(points)
        
# 이미지 복사

#cv2.imshow("Output-Keypoints",imageCopy)
#cv2.waitKey(0)

#관절 뽑은 사진 넘겨줌

    

# Emotion predict
prediction = emotion_classifier.predict(points)[0]

predicted_class = EMOTIONS[prediction.argmax()]

if predicted_class == [1,0,0,0]:
    pose = "goddess"
    
elif predicted_class == [0,1,0,0]:
    pose = "tree"    
    
elif predicted_class == 2:
    pose = "warrior2"
    
elif predicted_class == 3:
    pose = "false"

    
fontpath = "font/gulim.ttc"
font1 = ImageFont.truetype(fontpath, 50)
frame_pil = Image.fromarray(frame)
draw = ImageDraw.Draw(frame_pil)
draw.text((30, 30), pose, font=font1, fill=(0, 0, 255, 3))
frame = np.array(frame_pil)

print(pose)
    
# Open two windows
## Display image ("Emotion Recognition")
## Display probabilities of emotion
cv2.imshow('your pose', frame)
cv2.waitKey(0)

# Clear program and close windows
cv2.destroyAllWindows()


# In[ ]:




