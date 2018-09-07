import sys
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
import cv2    #3.4.0    cv2.__version__
import random
import subprocess as sp
import time
import threading
import v4l2capture
import select
import Image
def saveBufferedImg(picList,Alphabet):
    i=1;
    for pic in picList:
        pic.save('./dataProcess/Data/'+Alphabet+str(i), 'jpg')
        i += 1
    print "Finish save snapshot"   


class QtCapture(QtGui.QWidget):
    def __init__(self,videoSource=0,dataSource = None,nameList=['Plot1','Plot2','Plot3','Plot4','Plot5','Plot6']):
        self.app = QtGui.QApplication([])
        QtGui.QWidget.__init__(self)
        # self.mainWindow = QtGui.QMainWindow()
        # self.mainWindow.setWindowTitle('pyqtgraph example: PlotWidget')
        self.resize(1400,480)
        self.numofPlotWidget=3
        self.plotNum = 3
        # self.GuiWiget = QtGui.QWidget()
        # self.mainWindow.setCentralWidget(self.GuiWiget)
        self.firstDataTime=0.0
        self.lastDataTime=0.0
        self.fps = 100
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video = v4l2capture.Video_device("/dev/video0")
        self.size_x, self.size_y = self.video.set_format(640, 480)
        self.video.create_buffers(2)
        self.video.queue_all_buffers()

        self.video.start()
        # self.cap = cv2.VideoCapture(1)
        # self.port = open('/dev/video0', 'r')
        

        self.video_frame = QtGui.QLabel()
        self.fpsLabel = QtGui.QLabel()
        self.costTime = QtGui.QLabel()
        lay = QtGui.QGridLayout()
        lay.setMargin(0)
        
        # lay.addWidget(self.fpsLabel) #addWidget(QWidget, int row, int column, int rowspan, int columnspan)
        lay.addWidget(self.video_frame,0, 0,3,3) #lay.addWidget(self.video_frame,0, 0,3,3) for draw real time line
        lay.addWidget(self.costTime)
        VLayout1 = QtGui.QVBoxLayout()
        lay.addLayout(VLayout1,0, 4,3,8)

        self.numOfDataToPlot = 500
        self.XupperBound = 2.8
        self.plotWidgetList = []
        self.penStyleList= [[(0,0,200),(200,200,100),(195,46,212)],[(0,0,200),(200,200,100),(195,46,212)],[(0,0,200),(200,200,100),(195,46,212)]] 
        self.focusData = (256,0,0)
        self.index=0  
        self.curveList = []
        self.focusLineList = []
        self.curveXData =[i for i in range(0,self.numOfDataToPlot) ] 
        self.curveYDataList=[]

        for i,name in zip(range(0,self.numofPlotWidget),nameList):
            plotWidget = pg.PlotWidget(name=name)  ## giving the plots names allows us to link their axes together
            plotWidget.setXRange(0, self.numOfDataToPlot)
            if i == 0 :
                plotWidget.setYRange(-2, 2) 
            elif i == 1:
                plotWidget.setYRange(-2, 2) 
            else:
                plotWidget.setYRange(-180, 180)
            VLayout1.addWidget(plotWidget)
            self.plotWidgetList.append(plotWidget)
        # self.plotWidgetList[3].setXRange(0, self.XupperBound)


        # VLayout1.addWidget(self.costTime)
        self.setLayout(lay)
        self.show()

        #Draw Setting
        for plotWidget,penStyle in zip(self.plotWidgetList,self.penStyleList):  
            for i in range(0,self.plotNum):     
                curve = plotWidget.plot()       
                curve.setPen(penStyle[i])
                curveYData =[np.NAN for i in range(0,self.numOfDataToPlot) ] 
                self.curveList.append(curve)
                self.curveYDataList.append(curveYData)

        # for plotWidget in self.plotWidgetList:
        #     curve = plotWidget.plot()
        #     curve.setPen((255,0,0))            
        #     self.focusLineList.append(curve) #plot instance
            

        self.timeStamp =[0.0 for i in range(0,self.numOfDataToPlot) ]

        # ------ Modification ------ #
        self.isCapturing = False
        self.tempCapturing = False
        self.ith_frame = 1
        self.count=0
        self.tStart = time.time()
        self.tEnd = 0.0
        self.picList=[]
        self.Alphabet = 'A'
        # self.start()
        # self.app.instance().exec_()
        # self.fpsLabel.setText(str(self.fps))
        # ------ Modification ------ #

    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        # select.select((self.video,), (), ())
        # self.port.flush()
        # ret, frame = self.cap.read() 
        try: 
            image_data = self.video.read_and_queue()
            # print image_data
            image = Image.frombytes("RGB", (self.size_x, self.size_y), image_data)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)    #cv2.COLOR_RGB2GRAY

            frame = cv2.flip(np.array(image), 1)
            self.tEnd =  time.time()

            # My webcam yields frames in BGR format
            # if ret == True:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)#RGB888  Indexed8
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)
            # self.video_frame.setText("sss")
            # Save images if isCapturing
            if self.isCapturing :

                if self.tEnd - self.tStart >0.03:
                    p = QtGui.QPixmap.grabWidget(self)
                    self.picList.append(p)

                    # p.save('./dataProcess/Data/'+str(self.ith_frame), 'jpg')
                    # self.ith_frame = self.ith_frame + 1
                    self.tStart = time.time()

            elif self.isCapturing  == False and len(self.picList) !=0:
                # print "xxxxxxxxxxxxxxsaving"
                A = self.picList                
                t =threading.Thread(target = saveBufferedImg,args=(A ,self.Alphabet) ) #[data_chunk] make  data_chunk as a arguement
                t.start()
                self.picList = []
                self.Alphabet = self.Alphabet + 'X'

        except:
            pass



        self.costTime.setText(str(self.timeStamp[0])+"    "+str(self.timeStamp[-1])+"    "+ str(self.timeStamp[-1] - self.timeStamp[0]))
        # self.app.processEvents()
        # QtCore.QTimer.singleShot(1, self.nextFrameSlot)
        
    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    # ------ Modification ------ #
    def capture(self):
        if not self.isCapturing:
            self.isCapturing = True
        else:
            self.isCapturing = False
    # ------ Modification ------ #

    def ResetGraph(self):
        for i in range(0, len(self.curveYDataList) ):
            self.curveYDataList[i] =[np.NAN for j in range(0,self.numOfDataToPlot) ] 
        self.dataListLen = []
        self.dataTotolLen = 0

    def deleteLater(self):
        self.cap.release()
        super(QtGui.QWidget, self).deleteLater()

    def update(self):
        xd = self.curveXData
        for yData,curve in zip (self.curveYDataList,self.curveList):
            yData[1:]=yData[0:299]
            yData[0]=random.randint(-1,1)       
            yd = yData
            curve.setData(y=yd, x=xd)
        QtCore.QTimer.singleShot(20, self.update)

    def setMyData(self,dataList,isCapturing):

        if  len(dataList[0]) !=0 :
            for data,curve,yData in zip (dataList,self.curveList,self.curveYDataList): 
                if len(data) >= self.numOfDataToPlot:
                    curve.setData(y=data[:self.numOfDataToPlot], x=self.curveXData)
                else:
                    yData[0:self.numOfDataToPlot-len(data)]=yData[len(data):self.numOfDataToPlot]
                    yData[self.numOfDataToPlot-len(data):self.numOfDataToPlot]=data
                    curve.setData(y=yData, x=self.curveXData)
                    

            self.timeStamp[0:self.numOfDataToPlot-len(dataList[6])]=self.timeStamp[len(dataList[6]):self.numOfDataToPlot]
            self.timeStamp[self.numOfDataToPlot-len(dataList[6]):self.numOfDataToPlot]=dataList[6]
        # self.curveList[3].setData(y=self.curveYDataList[3], x=self.timeStamp)
        self.isCapturing = isCapturing
        # print self.isCapturing
        # if self.isCapturing == True:
        #     self.tempCapturing = self.isCapturing 
        self.nextFrameSlot()

        # exporter = pg.exporters.ImageExporter(self.plotWidgetList[0].plotItem)
        # exporter.export('fileName.png')
        self.app.processEvents()



if __name__ == '__main__':

    import sys
    # capture = QtCapture(0)
    # app = QtGui.QApplication([])
    # window = ControlWindow()
    # sys.exit(app.exec_())