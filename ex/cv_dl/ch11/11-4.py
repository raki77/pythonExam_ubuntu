from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image

img=Image.open('BSDS_361010.jpg')

feature_extractor=DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
model=DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

inputs=feature_extractor(img,return_tensors='pt')
res=model(**inputs)

import numpy as np
import cv2 as cv

colors=np.random.uniform(0,255,size=(100,3))	# 100개 색으로 트랙 구분
im=cv.cvtColor(np.array(img),cv.COLOR_BGR2RGB)
for i in range(res.logits.shape[1]):
    predicted_label=res.logits[0,i].argmax(-1).item() 
    if predicted_label!=91:
        name=model.config.id2label[predicted_label]
        prob='{:.2f}'.format(float(res.logits[0,i].softmax(dim=0)[predicted_label]))
        cx,cy=int(481*res.pred_boxes[0,i,0]),int(321*res.pred_boxes[0,i,1])
        w,h=int(481*res.pred_boxes[0,i,2]),int(321*res.pred_boxes[0,i,3])
        cv.rectangle(im,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),colors[predicted_label],2)
        cv.putText(im,name+str(prob),(cx-w//2,cy-h//2-5),cv.FONT_HERSHEY_SIMPLEX,0.6,colors[predicted_label],1)

cv.imshow('DETR',im)
cv.waitKey()       
cv.destroyAllWindows()