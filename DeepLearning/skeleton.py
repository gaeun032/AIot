#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import subprocess
import cv2
import numpy as np

import tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

with tensorflow.device('/GPU:0'):
    
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

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    data_path = './me&&/' #사진데이터 담긴 폴더
    target = 'a2'  #여기로 오픈포즈 결과

    if not os.path.exists(target):
        os.makedirs(target)
        
    sub_data = os.listdir(data_path)

    for folder in sub_data:
        set_path = os.path.join(target,folder)
        if not os.path.exists(set_path):
            os.makedirs(set_path)
    person_folder_list = os.listdir(os.path.join(data_path,folder))

    Frame_folder = os.listdir(data_path)

    for folder in Frame_folder:
        sub_folder_list=os.listdir(os.path.join(data_path, folder))
        for sub_folder in sub_folder_list:
            image = cv2.imread(os.path.join(data_path, folder,sub_folder))
            print(os.path.join(data_path, folder,sub_folder))
            image = cv2.resize(image,(224, 224))
            imageHeight, imageWidth, _ = image.shape
            inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
        
            H = output.shape[2]
            W = output.shape[3]
            points = []
            for i in range(0,15):
                probMap = output[0, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                x = (imageWidth * point[0]) / W
                y = (imageHeight * point[1]) / H
                if prob > 0.1 :  
                    cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
                    cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))
                else :
                    points.append(None)
        
            imageCopy = image
            for pair in POSE_PAIRS:
                partA = pair[0]             # Head
                partA = BODY_PARTS[partA]   # 0
                partB = pair[1]             # Neck
                partB = BODY_PARTS[partB]   # 1
                if points[partA] and points[partB]:
                    cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)
        
            cv2.imwrite(os.path.join(target, folder, sub_folder), imageCopy)
        

print ("================================================================================\n")
print ("Extraction Successful")

