# -*- coding: utf-8 -*-
from mtcnn import MTCNN
from skimage import io
from cv2 import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
import numpy as np


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



from kkjh_test import depth_test


model = load_model('MobilenetV2_batch64.h5')



from mtcnn import MTCNN

def mtcnn_face_crop(raw_frame):
  # raw_frame 이미지 => face_detection => crop => return
  # raw_frame 형태는 array 형태여야 함 
  # face detector model
  
  detector = MTCNN()

  # 3차원인 이미지만 진행 // 나머지는 pass
  if raw_frame.shape[2] == 4 :
    raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2RGB)
    print("4차원을 3차원으로")

  if raw_frame.shape[2] == 3 :
    face_box = detector.detect_faces(raw_frame) # detector 모델에서 face info. return

    if face_box : # 얼굴을 찾은 경우에만 !

      # face_boundingbox 좌표
      x,y = face_box[0]["box"][0], face_box[0]["box"][1]
      w,h = face_box[0]["box"][2] , face_box[0]["box"][3]
      para = 40  # padding_para
      y_s, y_e, x_s, x_e = y-para, y+h+para, x-para, x+w+para 

      if y_s < 0 :
        y_s = 0
      if y_e > raw_frame.shape[0] : # raw_frame.shape[0] : 원본 이미지의 세로 길이
        y_e = raw_frame.shape[0]
      if x_s < 0 :
        x_s = 0
      if x_e > raw_frame.shape[1] : # raw_frame.shape[1] : 원본 이미지의 가로 길이
        x_e = raw_frame.shape[1]

      # Crop
      dst = raw_frame.copy() 
      dst = dst[y_s : y_e, x_s : x_e]
      
      # 자른 이미지 리턴
      #plt.imshow(dst)
      return dst
    else:
      return raw_frame
  else:
    return raw_frame


def depth_estimation(raw_frame,k):
    print('depth시작')
    purpose_shape=(480,640,3)#이 사이즈로 통일하려고 함
    origin=mtcnn_face_crop(raw_frame)
    savePath='./examples/%d.bmp' %k
  
    if origin.size!=purpose_shape:#규격이 다른 애들만 새로 저장하고
        print('if시작')
        resized = cv2.resize(origin,(purpose_shape[1],purpose_shape[0]), interpolation=cv2.INTER_AREA)
        
        if origin.shape[2]==4:
            resized = cv2.cvtColor(resized,cv2.COLOR_RGBA2RGB)
        cv2.imwrite(savePath,resized)
        
    else:#같은 애들은 건드릴필요없지
        print('else시작')
        resized = origin
        cv2.imwrite(savePath, resized)
    
    outputs = depth_test(resized)
    savePath2='./results/%d.bmp' %k
    cv2.imwrite(savePath2, outputs)

  

    center_point=(outputs.shape[1]//2,outputs.shape[0]//2)
    #region=np.zeros((40,40))
    region=outputs[center_point[1]-20:center_point[1]+20,center_point[0]-20:center_point[0]+20]
    min_intensity=region.min()
    # print(min_intensity)
    outputs[outputs<=min_intensity]=0


    # center_intensity=a[center_point[1],center_point[0]]
    region=outputs[center_point[1]-20:center_point[1]+20,center_point[0]-20:center_point[0]+20]
    min_intensity=region.min()
    max_intensity=region.max()

    diff = (max_intensity-min_intensity)

    if diff < 100:
        flag1 = True
    else:
        flag1 = False

    return flag1

def mask_put_bool(raw_frame):
  picture = raw_frame
  copy_picture = picture.copy()
  resized = cv2.resize(copy_picture, dsize=(224, 224), interpolation=cv2.INTER_AREA)
  preprocessed = np.array(resized) / 255
  xhat = preprocessed[None, ]
  yhat = (model.predict(xhat) > 0).astype("int32")
  if yhat==1:
    return False
  else:
    return True


def warning_putText(raw_frame,flag1,flag2,k):
    if (flag1 == True) & (flag2 == True): # 마스크는 썼는데 얼굴터치한 경우
        final_frame = cv2.putText(raw_frame, "Dont touch ur face!!", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    elif (flag1 == True) & (flag2 == False): # 마스크 미착용인데 얼굴도 만졌어(호로자식 나가뒤져!!!!!)
        final_frame = cv2.putText(raw_frame, "Put on mask &", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
        final_frame = cv2.putText(raw_frame, "dont touch ur face!!", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    elif (flag1 == False) & (flag2 == True): # 마스크 썼는데 터치도 안함
        final_frame = cv2.putText(raw_frame, "Well done", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    else: # 마스크 안쓰고 터치 안함
        final_frame = cv2.putText(raw_frame, "Put on mask!!", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    return final_frame

def kkjh_smooth(raw_frame,flag1,k):
    #flag1 = dont(raw_frame) # flag1 : 얼굴터치여부(만졌으면 True)
    if flag1: # dont 돌렸는데 얼굴 만졌다고 나옴 -> 깊이 추정 해봐야함
        
        touch_bool = depth_estimation(raw_frame,k) # 깊이추정해서 접촉 : True, 미접촉 : False
        if ~touch_bool: # dont에서 얼굴 터치했다고 했는데 깊이추정결과가 미접촉으로 나온 경우
            flag1 = touch_bool # dont에서 만졌다고 했지만 깊이추정결과 미접촉이라 False 할당
    
    flag2 = mask_put_bool(raw_frame) # flag2 : 마스크 착용 여부
    final_frame = warning_putText(raw_frame,flag1,flag2,k)
    
    return final_frame

#picture = io.imread('./examples/mine/train_00000008.jpg')
#flag1=True
#final_frame =bsmooth(picture,flag1)
#plt.imshow(final_frame)
#plt.show()