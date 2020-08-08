#/action_classifier_mobile
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from model.mobilenetv3 import mobilenetv3_large#
from torchvision import transforms as  T
import time
import cv2


class ActionClassifier:
    def __init__(self, model_path, temporal_batch_size=3, img_size=224):

        ### Binary-class
        self.classes = ['non-touching', 'touching']
        self.touching_actions = ['touching']

        # b, c, w, h
        self.model = mobilenetv3_large()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        if self.device is 'cuda':
            state_dict = torch.load(model_path, map_location='cuda:0')
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(state_dict)


        self.temporal_batch_size = temporal_batch_size
        self.temporal_batch = np.zeros((img_size, img_size, 3))
        self.transforms = T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.pred = None
        self.cnt = 0

        self.temporal_stride = 5
        self.temporal_cnt = 0

        self.tmp = 0

    def run(self, img):
        a=''
        with torch.no_grad():
            # input gray image every temporal_stride
            if self.cnt % self.temporal_stride == 0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (224, 224))

                self.temporal_batch[:, :, self.temporal_cnt] = gray
                # if temproal_batch is full
                if self.temporal_cnt == self.temporal_batch_size-1:
                    start_time = time.time()
                    self.temporal_batch = self.temporal_batch.astype(np.uint8)
                    temporal_batch_img = Image.fromarray(self.temporal_batch)
                    #temporal_batch_img.save('dbg/{}.png'.format(self.tmp))
                    img_tensor = self.transforms(temporal_batch_img)
                    img_tensor = torch.unsqueeze(img_tensor, 0)
                    img_tensor = img_tensor.to(self.device)
                     
                    outputs = self.model(img_tensor)
                    
                    _, preds = outputs.max(1)
                    
                    self.pred = self.classes[preds.item()]
                    if (self.pred in self.touching_actions):
                        a='DO'
                        self.pred = '얼굴을 만지지 마세요 !'
                        
                        print('돈터취===============================================')
                        a='DONTDO'
                    else:
                        self.pred = ''
                        a='DO'
                    
                    self.temporal_cnt = 0
                    end_time = time.time()
                    print('elapsed time: ', end_time - start_time)
                    #a='dontdooooo'
                    

                self.temporal_cnt += 1
                self.cnt = 0
                
            self.cnt += 1
            self.tmp += 1
        
        return self.pred,a