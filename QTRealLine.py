from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import random

class MyRealTimePlot():
	def __init__(self,dataSource = None,nameList=['Plot1','Plot2','Plot3','Plot4','Plot5','Plot6']):   
		#Widget Setting
		#pg.setConfigOption('background', 'w')
		self.numOfDataToPlot = 500
		self.numofPlotWidget=3
		self.plotNum = 3
		self.plotWidgetList = []
		#[[(0,0,200),(200,200,100),(195,46,212)],[(237,177,32),(126,47,142),(43,128,200)],[(0,0,200),(200,200,100),(195,46,212)]] 
		self.penStyleList= [[(0,0,200),(200,200,100),(195,46,212)],[(0,0,200),(200,200,100),(195,46,212)],[(0,0,200),(200,200,100),(195,46,212)]] 
		self.index=0  
		self.dataListLen = []
		self.dataTotolLen = 0
		self.curveList = []
		self.curveXData =[i for i in range(0,self.numOfDataToPlot) ] 
		self.curveYDataList=[]
		self.app = QtGui.QApplication([])
		self.mainWindow = QtGui.QMainWindow()
		self.mainWindow.setWindowTitle('pyqtgraph example: PlotWidget')
		self.mainWindow.resize(800,800)
		self.GuiWiget = QtGui.QWidget()
		self.mainWindow.setCentralWidget(self.GuiWiget)
		layout = QtGui.QVBoxLayout()
		secondLayout = QtGui.QHBoxLayout()
		self.GuiWiget.setLayout(layout)
		layout.addLayout(secondLayout)
		pg.setConfigOption('background', 'w')
		
		for i,name in zip(range(0,self.numofPlotWidget),nameList):
			plotWidget = pg.PlotWidget(name=name)  ## giving the plots names allows us to link their axes together
			plotWidget.setXRange(0, self.numOfDataToPlot)
			if i == 0 :
				plotWidget.setYRange(-2, 2)	
			elif i == 1:
				plotWidget.setYRange(-2, 2)	
			else:
				plotWidget.setYRange(-180, 180)
			layout.addWidget(plotWidget)
			self.plotWidgetList.append(plotWidget)

		self.startLabel= QtGui.QLabel("Start:")
		self.startWindows = QtGui.QLineEdit()

		self.endLabel= QtGui.QLabel("End:")
		self.endWindows = QtGui.QLineEdit()

		self.button = QtGui.QPushButton('Split')
		self.button.clicked.connect(self.DrawPic)

		secondLayout.addWidget(self.startLabel)
		secondLayout.addWidget(self.startWindows)
		secondLayout.addWidget(self.endLabel)
		secondLayout.addWidget(self.endWindows)
		secondLayout.addWidget(self.button)

		# Display the widget as a new window
		self.mainWindow.show()



		#Draw Setting
		for plotWidget,penStyle in zip(self.plotWidgetList,self.penStyleList):	
			for i in range(0,self.plotNum):		
				curve = plotWidget.plot()		
				curve.setPen(penStyle[i])
				curveYData =[np.NAN for i in range(0,self.numOfDataToPlot) ] 
				self.curveList.append(curve)
				self.curveYDataList.append(curveYData)
		print "init ok"

	def close(self):
		self.app.closeAllWindows() 
		self.app.quit()

	def ResetGraph(self):
		for i in range(0, len(self.curveYDataList) ):
			self.curveYDataList[i] =[np.NAN for j in range(0,self.numOfDataToPlot) ] 
		self.dataListLen = []
		self.dataTotolLen = 0
	def setMyData(self,dataList,dataLength):
		# #reset Graph
		self.dataTotolLen = self.dataTotolLen +  len( dataList[0] )
		for data in dataLength:
			self.dataListLen.append(data)
		# print self.dataListLen,self.dataTotolLen
		for data,curve,yData,i in zip (dataList,self.curveList,self.curveYDataList ,range(0,7)): 
			if len(data) >= self.numOfDataToPlot:
				curve.setData(y=data[:self.numOfDataToPlot], x=self.curveXData)
			else:
				yData[0:self.numOfDataToPlot-len(data)]=yData[len(data):self.numOfDataToPlot]
				yData[self.numOfDataToPlot-len(data):self.numOfDataToPlot]=data
			# yData[1:]=yData[0:39]
			# yData[0]=data
				curve.setData(y=yData, x=self.curveXData)
		self.app.processEvents()
			# print yData
	def update(self):
		xd = self.curveXData
		for yData,curve in zip (self.curveYDataList,self.curveList):
			yData[1:]=yData[0:199]
			yData[0]=random.randint(-1,1)		
			yd = yData
			curve.setData(y=yd, x=xd)
		QtCore.QTimer.singleShot(1, self.update)
		# QtCore.QTimer.singleShot(1, self.update)

	def start(self):
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.update)
		self.timer.start(50)
		# start event
		self.app.instance().exec_()

	def  DrawPic(self):
		startWindowsIdx = int(self.startWindows.text())
		endWindowsIdx = int(self.endWindows.text())
		print startWindowsIdx - endWindowsIdx

		# for curve,yData,i in zip (self.curveList,self.curveYDataList ,range(0,7)): 
		# 	curveData =[np.NAN for i in range(0,self.numOfDataToPlot) ] 
		# 	curveData[self.numOfDataToPlot-len(yData[start:end]):self.numOfDataToPlot]=yData[start:end]
		# 	# yData[1:]=yData[0:39]
		# 	# yData[0]=data
		# 	curve.setData(y=curveData, x=self.curveXData)

		# self.app.processEvents()		

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #QtGui.QApplication.instance().exec_()
        A = MyRealTimePlot()
        # A.update()
        A.app.instance().exec_()

        # QtGui.QApplication.instance().exec_()
