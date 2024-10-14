import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통약자 보호')
        self.setGeometry(200,200,700,200)
       
        signButton=QPushButton('표지판 등록',self)
        roadButton=QPushButton('도로 영상 불러옴',self)
        recognitionButton=QPushButton('인식',self)
        quitButton=QPushButton('나가기',self)
        self.label=QLabel('환영합니다!',self)
        
        signButton.setGeometry(10,10,100,30)
        roadButton.setGeometry(110,10,100,30)
        recognitionButton.setGeometry(210,10,100,30)
        quitButton.setGeometry(510,10,100,30)
        self.label.setGeometry(10,40,600,170)
        
        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction) 
        recognitionButton.clicked.connect(self.recognitionFunction)        
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles=[['child.png','어린이'],['elder.png','노인'],['disabled.png','장애인']]	# 표지판 모델 영상
        self.signImgs=[]				# 표지판 모델 영상 저장
        
    def signFunction(self):
        self.label.clear()
        self.label.setText('교통약자 번호판을 등록합니다.')
        
        for fname,_ in self.signFiles:
            self.signImgs.append(cv.imread(fname))
            cv.imshow(fname,self.signImgs[-1])        

    def roadFunction(self):
        if self.signImgs==[]: 
            self.label.setText('먼저 번호판을 등록하세요.')
        else:
            fname=QFileDialog.getOpenFileName(self,'파일 읽기','./')
            self.roadImg=cv.imread(fname[0])
            if self.roadImg is None: sys.exit('파일을 찾을 수 없습니다.')  
    
            cv.imshow('Road scene',self.roadImg)  
        
    def recognitionFunction(self):
        if self.roadImg is None: 
            self.label.setText('먼저 도로 영상을 입력하세요.')
        else:
            sift=cv.SIFT_create()
        
            KD=[] # 여러 표지판 영상의 키포인트와 기술자 저장
            for img in self.signImgs: 
                gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray,None))
                
            grayRoad=cv.cvtColor(self.roadImg,cv.COLOR_BGR2GRAY) # 명암으로 변환
            road_kp,road_des=sift.detectAndCompute(grayRoad,None) # 키포인트와 기술자 추출
                
            matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            GM=[]			# 여러 표지판 영상의 good match를 저장
            for sign_kp,sign_des in KD:
                knn_match=matcher.knnMatch(sign_des,road_des,2)
                T=0.7
                good_match=[]
                for nearest1,nearest2 in knn_match:
                    if (nearest1.distance/nearest2.distance)<T:
                        good_match.append(nearest1)
                GM.append(good_match)        
            
            best=GM.index(max(GM,key=len)) # 매칭 쌍 개수가 최대인 번호판 찾기
            
            if len(GM[best])<4:	# 최선의 번호판이 매칭 쌍 4개 미만이면 실패
                self.label.setText('표지판이 없습니다.')  
            else:			# 성공(호모그래피 찾아 영상에 표시)
                sign_kp=KD[best][0]
                good_match=GM[best]
            
                points1=np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])
                points2=np.float32([road_kp[gm.trainIdx].pt for gm in good_match])
                
                H,_=cv.findHomography(points1,points2,cv.RANSAC)
                
                h1,w1=self.signImgs[best].shape[0],self.signImgs[best].shape[1] # 번호판 영상의 크기
                h2,w2=self.roadImg.shape[0],self.roadImg.shape[1] # 도로 영상의 크기
                
                box1=np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(4,1,2)
                box2=cv.perspectiveTransform(box1,H)
                
                self.roadImg=cv.polylines(self.roadImg,[np.int32(box2)],True,(0,255,0),4)
                
                img_match=np.empty((max(h1,h2),w1+w2,3),dtype=np.uint8)
                cv.drawMatches(self.signImgs[best],sign_kp,self.roadImg,road_kp,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imshow('Matches and Homography',img_match)
                
                self.label.setText(self.signFiles[best][1]+' 보호구역입니다. 30km로 서행하세요.')                 
                winsound.Beep(3000,500)         
                      
    def quitFunction(self):
        cv.destroyAllWindows()        
        self.close()
                
app=QApplication(sys.argv) 
win=TrafficWeak() 
win.show()
app.exec_()