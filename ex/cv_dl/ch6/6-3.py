import cv2 as cv 
import numpy as np
import sys
from PyQt5.QtWidgets import *
      
class Orim(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('오림')
        self.setGeometry(200,200,700,200)
       
        fileButton=QPushButton('파일',self)
        paintButton=QPushButton('페인팅',self)
        cutButton=QPushButton('오림',self)
        incButton=QPushButton('+',self)
        decButton=QPushButton('-',self)
        saveButton=QPushButton('저장',self)
        quitButton=QPushButton('나가기',self)
        
        fileButton.setGeometry(10,10,100,30)
        paintButton.setGeometry(110,10,100,30)
        cutButton.setGeometry(210,10,100,30)
        incButton.setGeometry(310,10,50,30)
        decButton.setGeometry(360,10,50,30)
        saveButton.setGeometry(410,10,100,30)
        quitButton.setGeometry(510,10,100,30)
        
        fileButton.clicked.connect(self.fileOpenFunction)
        paintButton.clicked.connect(self.paintFunction) 
        cutButton.clicked.connect(self.cutFunction)    
        incButton.clicked.connect(self.incFunction)              
        decButton.clicked.connect(self.decFunction) 
        saveButton.clicked.connect(self.saveFunction)                         
        quitButton.clicked.connect(self.quitFunction)

        self.BrushSiz=5			# 페인팅 붓의 크기
        self.LColor,self.RColor=(255,0,0),(0,0,255) # 파란색 물체, 빨간색 배경
        
    def fileOpenFunction(self):
        fname=QFileDialog.getOpenFileName(self,'Open file','./')
        self.img=cv.imread(fname[0])
        if self.img is None: sys.exit('파일을 찾을 수 없습니다.')  
        
        self.img_show=np.copy(self.img)	# 표시용 영상 
        cv.imshow('Painting',self.img_show)
        
        self.mask=np.zeros((self.img.shape[0],self.img.shape[1]),np.uint8) 
        self.mask[:,:]=cv.GC_PR_BGD	# 모든 화소를 배경일 것 같음으로 초기화
            
    def paintFunction(self):
        cv.setMouseCallback('Painting',self.painting) 
        
    def painting(self,event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONDOWN:   
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1) # 왼쪽 버튼을 클릭하면 파란색
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event==cv.EVENT_RBUTTONDOWN: 
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1) # 오른쪽 버튼을 클릭하면 빨간색
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)
        elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1) # 왼쪽 버튼을 클릭하고 이동하면 파란색
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1) # 오른쪽 버튼을 클릭하고 이동하면 빨간색 
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)
    
        cv.imshow('Painting',self.img_show)        
        
    def cutFunction(self):
        background=np.zeros((1,65),np.float64) 
        foreground=np.zeros((1,65),np.float64) 
        cv.grabCut(self.img,self.mask,None,background,foreground,5,cv.GC_INIT_WITH_MASK)
        mask2=np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
        self.grabImg=self.img*mask2[:,:,np.newaxis]
        cv.imshow('Scissoring',self.grabImg) 
        
    def incFunction(self):
        self.BrushSiz=min(20,self.BrushSiz+1) 
        
    def decFunction(self):
        self.BrushSiz=max(1,self.BrushSiz-1) 
        
    def saveFunction(self):
        fname=QFileDialog.getSaveFileName(self,'파일 저장','./')
        cv.imwrite(fname[0],self.grabImg)
                
    def quitFunction(self):
        cv.destroyAllWindows()        
        self.close()
                
app=QApplication(sys.argv) 
win=Orim() 
win.show()
app.exec_()