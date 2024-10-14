import numpy as np
import cv2 as cv
import sys

def draw_OpticalFlow(img,flow,step=16):
    for y in range(step//2,frame.shape[0],step):
        for x in range(step//2,frame.shape[1],step):
            dx,dy=flow[y,x].astype(np.int)
            if(dx*dx+dy*dy)>1:
                cv.line(img,(x,y),(x+dx,y+dy),(0,0,255),2) # 큰 모션 있는 곳은 빨간색
            else:
                cv.line(img,(x,y),(x+dx,y+dy),(0,255,0),2)            
    
cap=cv.VideoCapture(0,cv.CAP_DSHOW)	# 카메라와 연결 시도
if not cap.isOpened(): sys.exit('카메라 연결 실패')
    
prev=None

while(1):
    ret,frame=cap.read()	# 비디오를 구성하는 프레임 획득
    if not ret: sys('프레임 획득에 실패하여 루프를 나갑니다.')
    
    if prev is None:	# 첫 프레임이면 광류 계산 없이 prev만 설정
        prev=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        continue
    
    curr=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    flow=cv.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0)
    
    draw_OpticalFlow(frame,flow)
    cv.imshow('Optical flow',frame)

    prev=curr

    key=cv.waitKey(1)	# 1밀리초 동안 키보드 입력 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv.destroyAllWindows() 