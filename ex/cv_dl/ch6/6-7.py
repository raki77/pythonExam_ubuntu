import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
    
class VideoSpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오 특수 효과')
        self.setGeometry(200,200,400,100)
       
        videoButton=QPushButton('비디오 시작',self)
        self.pickCombo=QComboBox(self)     
        self.pickCombo.addItems(['엠보싱','카툰','연필 스케치(명암)','연필 스케치(컬러)','유화'])
        quitButton=QPushButton('나가기',self)        
        
        videoButton.setGeometry(10,10,140,30)
        self.pickCombo.setGeometry(150,10,110,30)                  
        quitButton.setGeometry(280,10,100,30)
        
        videoButton.clicked.connect(self.videoSpecialEffectFunction) 
        quitButton.clicked.connect(self.quitFunction)
        
    def videoSpecialEffectFunction(self):             
        self.cap=cv.VideoCapture(0,cv.CAP_DSHOW) 
        if not self.cap.isOpened(): sys.exit('카메라 연결 실패')
        
        while True:
            ret,frame=self.cap.read()  
            if not ret: break

            pick_effect=self.pickCombo.currentIndex()        
            if pick_effect==0:
                femboss=np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 1.0]])
                gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)    
                gray16=np.int16(gray)
                special_img=np.uint8(np.clip(cv.filter2D(gray16,-1,femboss)+128,0,255))
            elif pick_effect==1:
                special_img=cv.stylization(frame,sigma_s=60,sigma_r=0.45)
            elif pick_effect==2:    
                special_img,_=cv.pencilSketch(frame,sigma_s=60,sigma_r=0.07,shade_factor=0.02)
            elif pick_effect==3:    
                _,special_img=cv.pencilSketch(frame,sigma_s=60,sigma_r=0.07,shade_factor=0.02)
            elif pick_effect==4:    
                special_img=cv.xphoto.oilPainting(frame,10,1,cv.COLOR_BGR2Lab)
                
            cv.imshow('Special effect',special_img)              
            cv.waitKey(1) 
                                        
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()        
        self.close()
                
app=QApplication(sys.argv) 
win=VideoSpecialEffect() 
win.show()
app.exec_()