import socket
import cv2
import pickle
import struct 
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import ImageFont, ImageDraw, Image

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ip = '172.20.10.7'# ip 주소
port = 6002 # port 번호

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # 소켓 객체를 생성
s.bind((ip, port)) # 바인드(bind) : 소켓에 주소, 프로토콜, 포트를 할당
s.listen(True) # 연결 수신 대기 상태(리스닝 수(동시 접속) 설정)
print('클라이언트 연결 대기')

# 연결 수락(클라이언트 소켓 주소를 반환)
conn, addr = s.accept()
print(addr) # 클라이언트 주소 출력

data = b"" # 수신한 데이터를 넣을 변수
payload_size = struct.calcsize(">L")

UDP_IP = "172.20.10.7"
UDP_PORT = 5065

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

pose_classifier = load_model('./model_r.h5', compile=False)
POSE = ["goddess" ,"tree", "warrior2", "False"]

while True:
    with tensorflow.device('/GPU:0'):
        # 프레임 수신
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        print("Frame Size : {}".format(msg_size)) # 프레임 크기 출력

        # 역직렬화(de-serialization) : 직렬화된 파일이나 바이트를 원래의 객체로 복원하는 것
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes") # 직렬화되어 있는 binary file로 부터 객체로 역직렬화
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR) # 프레임 디코딩
    
        # frame.shape = 불러온 이미지에서 height, width, color 받아옴
        frame = cv2.resize(frame, (224,224))
        imageHeight, imageWidth, _ = frame.shape

        # network에 넣기위해 전처리
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

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
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else :
                points.append(None)
            
        # 이미지 복사
        imageCopy = frame

        # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
        for pair in POSE_PAIRS:
            partA = pair[0]             # Head
            partA = BODY_PARTS[partA]   # 0
            partB = pair[1]             # Neck
            partB = BODY_PARTS[partB]   # 1
    
            #print(partA," 와 ", partB, " 연결\n")
            if points[partA] and points[partB]:
                cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)
            
        #관절 뽑은 사진 넘겨줌
        frame = imageCopy
    
        # Resize the image to 48x48 for neural network
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(frame, (71, 71))
        roi = np.array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = preprocess_input(roi)
        
        # Emotion predict
        prediction = pose_classifier.predict(roi)
        predicted_class = np.argmax(prediction[0])
        
        if predicted_class == 0:
            pose = "goddess"
            sock.sendto( ("goddess").encode(), (UDP_IP, UDP_PORT))
        elif predicted_class == 1:
            pose = "tree"
            sock.sendto( ("tree").encode(), (UDP_IP, UDP_PORT))
        elif predicted_class == 2:
            pose = "warrior2"
            sock.sendto( ("warrior2").encode(), (UDP_IP, UDP_PORT))
        elif predicted_class == 3:
            pose = "False"
            sock.sendto( ("False").encode(), (UDP_IP, UDP_PORT))
        
    print(pose)

    # 영상 출력
    cv2.imshow('TCP_Frame_Socket',frame)
    
    #1초 마다 키 입력 상태를 받음
    
    if cv2.waitKey(1) == ord('q') : # q를 입력하면 종료
        break
            
        cv2.destroyAllWindows()