import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')	 # 영상 읽기
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray,100,200) 

contour,hierarchy=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

lcontour=[]   
for i in range(len(contour)):
    if contour[i].shape[0]>100:	# 길이가 100보다 크면
        lcontour.append(contour[i])
    
cv.drawContours(img,lcontour,-1,(0,255,0),3)
             
cv.imshow('Original with contours',img)    
cv.imshow('Canny',canny)    

cv.waitKey()
cv.destroyAllWindows()