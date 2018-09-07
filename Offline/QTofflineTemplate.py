from pyqtgraph.Qt import QtGui, QtCore
from sklearn import preprocessing
import numpy as np
import pyqtgraph as pg
import random
import pickle
import collections
from os import listdir
from os.path import isfile, join
import fastdtw
from scipy.spatial.distance import euclidean
import scipy.stats
from scipy import signal
import DTW
import time
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from scipy import stats 
import re

DirSetting ='./Jay/' #base gesture Dit
ComplexDirSetting ='./Jay/Complex/' #complex gesture Dir
Name = 'Jay'  #the prefix of filename

'''
	MyRealTimePlot: create UI and construct the DTW Model
	setMyData : a real-time ploter
	DrawPic : draw the specificed file data
	compare : compare two specificed file data by DTW and return the Distance

	

	JudgeAll : Recognize all the file include base gesture and complex gesture.
			   This function is for getting accuracy for test set.
			   *** If you need GUI only, you do not need this function     *** 	   

			    
'''

def absDist(A,B):

	return np.sum(np.abs(A-B))

class MyRealTimePlot():
	def __init__(self,dataSource = None,nameList=['Plot1','Plot2','Plot3','Plot4','Plot5','Plot6']):   
		'''
		construct GUI 
		'''
		self.numOfDataToPlot = 500 #nuber of point of x 
		self.ScalerNum = 2 # every self.ScalerNum  we sample once - not used 
		self.numofPlotWidget=3 
		self.plotNum = 3 # how many line to plot in a plot widget
		self.plotWidgetList = []
		self.penStyleList= [[(0,0,200),(200,200,100),(195,46,212)],[(237,177,32),(126,47,142),(43,128,200)],[(0,0,200),(200,200,100),(195,46,212)]] 
		self.index=0  
		self.dataListLen = []
		self.ROI1 = None # region of interest
		self.ROI2 = None
		self.dataTotolLen = 0
		self.curveList = []
		self.curveList2 = []
		self.curveXData =[i for i in range(0,self.numOfDataToPlot) ] #initial x value
		self.curveYDataList=[]
		self.curveYDataList2=[]
		self.app = QtGui.QApplication([])
		self.mainWindow = QtGui.QMainWindow()
		self.mainWindow.setWindowTitle('pyqtgraph example: PlotWidget')
		self.mainWindow.resize(720,640)
		self.GuiWiget = QtGui.QWidget()
		self.mainWindow.setCentralWidget(self.GuiWiget)
		layout = QtGui.QVBoxLayout()
		secondLayout = QtGui.QHBoxLayout()
		thirdLayout = QtGui.QHBoxLayout()
		self.GuiWiget.setLayout(layout)
		layout.addLayout(secondLayout)
		layout.addLayout(thirdLayout)
		pg.setConfigOption('background', 'w')

		# create plot widgets by  pg.PlotWidget(name=name) and we can draw multiple curve lines on it 
		for i,name in zip(range(0,self.numofPlotWidget),nameList):
			plotWidget = pg.PlotWidget(name=name) 

			# set X range
			plotWidget.setXRange(0, self.numOfDataToPlot)

			# set Y range
			if i == 0 :
				plotWidget.setYRange(-2, 2)	
			elif i == 1:
				plotWidget.setYRange(-180, 180)	
			else:
				plotWidget.setYRange(-2, 2)
			layout.addWidget(plotWidget)
			self.plotWidgetList.append(plotWidget)

		self.startLabel= QtGui.QLabel("Start:")
		self.startWindows = QtGui.QLineEdit()

		self.endLabel= QtGui.QLabel("End:")
		self.endWindows = QtGui.QLineEdit()

		self.button = QtGui.QPushButton('Split')
		self.button.clicked.connect(self.DrawPic)

		self.fileName= QtGui.QLabel("fileName:")
		self.fileInputName = QtGui.QComboBox() 
		#self.fileInputName.setText("UpStraight0.dat")

		self.Readbutton = QtGui.QPushButton('Read')
		
		self.Readbutton.clicked.connect(self.ReadFile)

		secondLayout.addWidget(self.startLabel)
		secondLayout.addWidget(self.startWindows)
		secondLayout.addWidget(self.endLabel)
		secondLayout.addWidget(self.endWindows)
		secondLayout.addWidget(self.button)
		secondLayout.addWidget(self.fileName)
		secondLayout.addWidget(self.fileInputName)
		secondLayout.addWidget(self.Readbutton)

		self.Aobj= QtGui.QLabel("A:")
		self.comboA = QtGui.QComboBox()
		self.AstartLabel= QtGui.QLabel("Start:")  
		self.AstartWindows = QtGui.QLineEdit()
		self.AendLabel= QtGui.QLabel("End:")
		self.AendWindows = QtGui.QLineEdit()		


		self.Comparebutton = QtGui.QPushButton('Compare')
		#register a callback funtion - when button is pressed, we execute it 
		self.Comparebutton.clicked.connect(self.compare)


		thirdLayout.addWidget(self.AstartLabel)
		thirdLayout.addWidget(self.AstartWindows)
		thirdLayout.addWidget(self.AendLabel)
		thirdLayout.addWidget(self.AendWindows)
		thirdLayout.addWidget(self.Aobj)
		thirdLayout.addWidget(self.comboA)

		thirdLayout.addWidget(self.Comparebutton)

		# read file from Directory
		self.readDir()
		# Display the whole GUI architecture
		self.mainWindow.show()



		#Create plot instance by plotWidget.plot() and initial the Y value
		for plotWidget,penStyle in zip(self.plotWidgetList,self.penStyleList):	
			for i in range(0,self.plotNum):		
				curve = plotWidget.plot()		
				curve.setPen(penStyle[i])
				curveYData =[np.NAN for i in range(0,self.numOfDataToPlot) ]   #initial y value
				self.curveList.append(curve)
				self.curveYDataList.append(curveYData)

		for i in range(0,self.plotNum):	
			curve = self.plotWidgetList[2].plot()
			curve.setPen(penStyle[i])
			curveYData =[np.NAN for i in range(0,self.numOfDataToPlot) ] 
			self.curveList2.append(curve)
			self.curveYDataList2.append(curveYData)			


		self.SettingModel()
		print "init ok"

		self.writeout = True
		self.logfp = open('log.txt', "w")
		self.timeLogfp = open('Timelog.txt', "w")

	def SettingModel(self):
		'''load model here'''
		pass

	def close(self):
		self.app.closeAllWindows() 
		self.app.quit()

	def ResetGraph(self):
		for i in range(0, len(self.curveYDataList) ):
			self.curveYDataList[i] =[np.NAN for j in range(0,self.numOfDataToPlot) ] 
		for i in range(0, len(self.curveYDataList2) ):
			self.curveYDataList2[i] =[np.NAN for j in range(0,self.numOfDataToPlot) ] 
		self.dataListLen = []
		self.dataTotolLen = 0
		try:
			self.plotWidgetList[0].removeItem(self.ROI1)
			self.plotWidgetList[1].removeItem(self.ROI2)
		except:
			pass 
		self.ROI1 = None
		self.ROI2 = None

	def RegionWindows(self,upBbound):
		axes = ['X','Y','Z']
		Dirs = ['P2N','N2P']

		ret = {}
		ret['X'] ={}
		ret['Y'] ={}
		ret['Z'] ={}
		ret['X']['N2P'] = []
		ret['X']['P2N'] = []
		ret['Y']['N2P'] = []
		ret['Y']['P2N'] = []
		ret['Z']['N2P'] = []
		ret['Z']['P2N'] = []

		for axis in axes:
			for Dir in Dirs:
				for boundry in self.windowsCrossDataIndex[axis][Dir]:

					if boundry[1] >  upBbound:
						break
					else:
						ret[axis][Dir].append(boundry)
		return ret

	def GetInfo(self,ret):
		axes = ['X','Y','Z']
		Dirs = ['P2N','N2P']
		pos = {}
		pos['X'] =[0,1]
		pos['Y'] =[1,2]
		pos['Z'] =[2,3]
		for axis in axes:
			for Dir in Dirs:
				# print  axis,Dir,"------------------------"
				for idx in ret[axis][Dir]:			
					if idx[1] - idx[0] < 40:
						# print idx[0],idx[1],"not enough"
						idx.append("not enough")
					else:
						# print  idx[0],idx[1],np.sqrt(np.mean(np.multiply(self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]],self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]]),axis=0))
						if np.sqrt(np.mean(np.multiply(self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]],self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]]),axis=0)) < 0.15:
							idx.append(None)
							# print np.sqrt(np.mean(np.multiply(self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]],self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]]),axis=0))
						else: 
							idx.append(np.sqrt(np.mean(np.multiply(self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]],self.Acc[idx[0]:idx[1],pos[axis][0]:pos[axis][1]]),axis=0)) )  
	def  DrawPic(self):
		self.ResetGraph()

		startWindowsIdx = int(self.startWindows.text())
		endWindowsIdx = int(self.endWindows.text())
		
		startIDX = self.workingIdx[startWindowsIdx][0] 
		endIDX = self.workingIdx[endWindowsIdx][1]

		#start:stop:step
		ret = self.RegionWindows(endIDX)
		print ret,endIDX

		dataList = np.concatenate((self.Acc[startIDX:endIDX,:],self.Angle[startIDX:endIDX,:]),axis=1)
		# print "scipy.stats.skewtest:",scipy.stats.skewtest(self.Acc[startIDX:endIDX,:],axis=0)
		# print "mean:",np.mean(  self.Acc[self.workingIdx[endWindowsIdx][0]:endIDX,:] ,axis=0)
		# print "start angle:",self.Angle[startIDX,:],"Last angle:",self.Angle[endIDX-1,:]
		# print "local Min X:",scipy.signal.argrelmin(self.Acc[startIDX:endIDX,0:1] ,axis=0)[0],"local Min Y:",scipy.signal.argrelmin(self.Acc[startIDX:endIDX,1:2] ,axis=0)[0],"local Min Z:",scipy.signal.argrelmin(self.Acc[startIDX:endIDX,2:3] ,axis=0)[0]
		# print "local Max X:",scipy.signal.argrelmax(self.Acc[startIDX:endIDX,0:1] ,axis=0)[0],"local Max Y:",scipy.signal.argrelmax(self.Acc[startIDX:endIDX,1:2] ,axis=0)[0],"local Max Z:",scipy.signal.argrelmax(self.Acc[startIDX:endIDX,2:3] ,axis=0)[0]
		dataList =  dataList[::self.ScalerNum,:]
		self.GetInfo(ret)
		dataList = dataList.transpose()


		
		self.ROI1 = pg.LinearRegionItem([startIDX,endIDX])
		self.ROI2 = pg.LinearRegionItem([startIDX,endIDX])

		self.plotWidgetList[0].addItem(self.ROI1)
		self.plotWidgetList[1].addItem(self.ROI2)
		# print endIDX-startIDX,dataList.shape
		for data,curve,yData,i in zip (dataList,self.curveList2,self.curveYDataList2 ,range(0,7)): 
			# print len(yData)
			yData[0:dataList.shape[1]] = dataList[i,:].tolist()[0]
			# print len(dataList[i,:].tolist())
			curve.setData(y=yData, x=self.curveXData)

		self.app.processEvents()	
	def diffAngle(self,data):
		ret = np.mat([0.0,0.0,0.0])
		for i,j in zip(range(0,data.shape[0]-1), range(1,data.shape[0]) ):
			ret =np.concatenate ( (ret,data[j,:]-data[i,:]),axis=0 ) 
		return ret
	def ReadFile(self):
		self.ResetGraph()
		self.filename = str(self.fileInputName.currentText())
		self.FileJudge(self.filename,paint=True)

	def FileJudge(self,filename,paint=False,writeout=False):		
		print filename,DirSetting,ComplexDirSetting
		if writeout==True:
			self.logfp.write(filename+ '  '+DirSetting+'\n')
		try:
			fp = open(ComplexDirSetting+filename, "rb")
		except:
			fp = open(DirSetting+filename, "rb")		           
		tempDict = pickle.load(fp)
			# print tempDict
		self.filteredIdx = tempDict['filteredIdx']
		self.Acc = tempDict['Acc']
		self.Gyo = tempDict['Gyo']
		
		self.Mag = tempDict['Mag'] 
		self.Angle = tempDict['Angle']
		# self.Angle = self.Angle - np.mean(self.Angle,axis=1)
		self.windowsCrossDataIndex = tempDict['windowsCrossDataIndex']
		self.AccRawState = tempDict['AccRawState']
		self.GyoRawState = tempDict['GyoRawStata']
		self.workingIdx = tempDict['workingIdx']

		self.MayBeValid = []
		self.VarDataIdx = tempDict['VarDataIdx']
		self.seq = tempDict['seq']
		self.timestamp = tempDict['timestamp']
		
		offset = self.workingIdx[0][0]
		for i in range (0,len(self.workingIdx)):
			self.workingIdx[i][0] = self.workingIdx[i][0] - offset
			self.workingIdx[i][1] = self.workingIdx[i][1] - offset

		for axis in ['X','Y','Z']:
			for Dir in ['N2P','P2N']:
				for i in range(0,len(self.windowsCrossDataIndex[axis][Dir])):					
					self.windowsCrossDataIndex[axis][Dir][i][0]  =	self.windowsCrossDataIndex[axis][Dir][i][0] - offset
					self.windowsCrossDataIndex[axis][Dir][i][1]  =	self.windowsCrossDataIndex[axis][Dir][i][1] - offset	
		# self.windowsCrossDataIndex =  self.getCrossingWindowsIDX(self.Acc)
		else:			
			self.logfp.write("workingIdx:"+str(len(self.workingIdx))+'\n')

		startIDX = self.workingIdx[0][0] 
		endIDX = self.workingIdx[-1][1]
		dataList = np.concatenate((self.Acc,self.Angle),axis=1)
		dataList = dataList.transpose()

		if paint == True:
			# Plot Data
			for data,curve,yData,i in zip (dataList,self.curveList,self.curveYDataList ,range(0,7)): 
				# print len(yData)
				yData[0:endIDX-startIDX] = dataList[i,:].tolist()[0]
				# print len(dataList[i,:].tolist())
				curve.setData(y=yData, x=self.curveXData)
			self.app.processEvents()
		
		# classify or somewhat you can implement in Judge function
		self.Judge()

	def readDir(self):
		''' read Dir and catch all file contained the keywords which defined in complexG & keywords'''
		global DirSetting,ComplexDirSetting,Name
		self.fileList = []
		complexG =['ForwardBackward','BackwardForward','DownUp','UpDown','RightLeft','LeftRight','LeftForward','RightForward','UpRight','UpLeft','RightUpForward','ForwardRight','V','VII','LeftbackForward','RightbackForward','ForwardLeft','LeftLeftForward','RightRightForward','ForwardUp','DownBackward','DownLeft','DownRight','ForwardBackwardForward','LeftRightUp','BackwardRightLeftforward','DownLeftRightforward','RightUpForward']
		keywords = ['GoStraight','BackStraight','DownStraight','UpStraight','LeftStraight','RightStraight','RightUpStraight','LeftGoStraight','RightGoStraight','LeftBackStraight','RightBackStraight']
		for i in range (0,len(complexG)):
			complexG[i] = Name + complexG[i]		
		for i in range (0,len(keywords)):
			keywords[i] = Name + keywords[i]

		for keyword in complexG:
			for fileName in listdir(ComplexDirSetting):	
				# if keyword == 'Circle':s
				# 	print fileName,fileName[0:len(keyword)]
				if keyword  in fileName[0:len(keyword)]:
					if fileName not in self.fileList:
						self.fileList.append(fileName)

		for keyword in keywords:
			for fileName in listdir(DirSetting):	
				# if keyword == 'Circle':s
				# 	print fileName,fileName[0:len(keyword)]
				if keyword  in fileName[0:len(keyword)]:
					if fileName not in self.fileList:
						self.fileList.append(fileName)
					
		self.fileInputName.addItems(self.fileList)	
		self.comboA.addItems(self.fileList)	

	def Judge(self,writeout=False):
		pass
		# print "\033[1;31m",data[3],"len:",data[2] - data[1] + 1 ,data[2],data[1],"\033[1;m"

		# print "Dis:",disOrder
		# print "Angle:",angleOrder

	def compare(self):
		startWindowsIdx = int(self.startWindows.text())
		endWindowsIdx = int(self.endWindows.text())
		startIDX = self.workingIdx[startWindowsIdx][0] 
		endIDX = self.workingIdx[endWindowsIdx][1]		

		AstartWindowsIdx = None
		AendWindowsIdx = None
		AstartIDX = None
		AendIDX = None
		try:
			AstartWindowsIdx = int(self.AstartWindows.text())
			AendWindowsIdx = int(self.AendWindows.text())
	
		except:
			pass

		filename = str(self.comboA.currentText())
		
		try:
			fp = open(DirSetting+filename, "rb") 
		except:
			fp = open(ComplexDirSetting+filename, "rb")

		tempDict = pickle.load(fp)


		Acc = tempDict['Acc']
		Gyo = tempDict['Gyo']
		Mag = tempDict['Mag'] 
		Angle = tempDict['Angle']
		windowsCrossDataIndex = tempDict['windowsCrossDataIndex']
		AccRawState = tempDict['AccRawState']
		GyoRawState = tempDict['GyoRawStata']
		workingIdx = tempDict['workingIdx']
		VarDataIdx = tempDict['VarDataIdx']
		seq = tempDict['seq']
		timestamp = tempDict['timestamp']       

		AstartIDX = workingIdx[AstartWindowsIdx][0] 
		AendIDX = workingIdx[AendWindowsIdx][1]

		print startWindowsIdx,endWindowsIdx,AstartWindowsIdx,AendWindowsIdx


	

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #QtGui.QApplication.instance().exec_()
        A = MyRealTimePlot()
        # A.update()
        A.app.instance().exec_()

        # QtGui.QApplication.instance().exec_()


# def cross_correlation_using_fft(x, y):
	# 	f1 = fft(x)
	# 	f2 = fft(np.flipud(y))
	# 	cc = np.real(ifft(f1 * f2))
	# 	return fftshift(cc)			
	# 	# print self.windowsCrossDataIndex
	# 	# for gesture in FinalS:
	# 	# 	print "\033[1;31m",gesture,"\033[1;m"
	# def compute_shift(x, y):
	    
	# 	assert len(x) == len(y)
	# 	c = cross_correlation_using_fft(x, y)
	# 	assert len(c) == len(x)
	# 	zero_index = int(len(x) / 2) - 1
	# 	shift = zero_index - np.argmax(c)
	# 	return shift