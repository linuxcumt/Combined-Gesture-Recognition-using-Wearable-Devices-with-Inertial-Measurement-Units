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

		#Load Model as DTW pattern
		self.SettingModel()
		print "init ok"

		self.writeout = True
		self.logfp = open('log.txt', "w")
		self.timeLogfp = open('Timelog.txt', "w")
		self.JudgeAll()

	def SettingModel(self):
		'''we load 10 based gesture here'''
		global DirSetting,Name	
		self.DTWModel = {}
		self.DTWModel['0'] = {}
		self.DTWModel['1'] = {}
		self.DTWModel['2'] = {}
		self.DTWModel['0']['N'] = {}
		self.DTWModel['0']['P'] = {}
		self.DTWModel['1']['N'] = {}
		self.DTWModel['1']['P'] = {}
		self.DTWModel['2']['N'] = {}
		self.DTWModel['2']['P'] = {}

		self.AnglePosePoint = {}
		self.AnglePosePoint['GoStraight'] = [1,0]
		self.AnglePosePoint['BackStraight'] = [1,0]
		self.AnglePosePoint['RightGoStraight'] = [0,1]
		self.AnglePosePoint['LeftGoStraight'] = [0,1]
		self.AnglePosePoint['LeftBackStraight'] = [0,1]
		self.AnglePosePoint['RightBackStraight'] = [0,1]
		self.AnglePosePoint['RightUpStraight'] = [1,0]

		self.AnglePosePoint['Right'] = [0]		
		self.AnglePosePoint['Left'] = [0]			
		self.AnglePosePoint['Circle1'] = [0]
		self.AnglePosePoint['Circle'] = [0]
		self.AnglePosePoint['Circle2'] = [0]

		self.AnglePosePoint['UpStraight'] = [1]
		self.AnglePosePoint['DownStraight'] = [1]





		self.AnglePose = {}

		# '0' for Yaw , '1' for pitch , '2' for roll 
		self.DTWModel['0']['N']['Right'] = None
		self.DTWModel['0']['P']['Left'] = None
		self.DTWModel['1']['N']['GoStraight'] = None
		self.DTWModel['1']['N']['DownStraight'] = None
		self.DTWModel['1']['P']['UpStraight'] = None
		self.DTWModel['1']['P']['BackStraight'] = None

		self.DTWModel['1']['N']['GoStraight'] = self.LoadData(DirSetting+Name+"GoStraight0.dat","GoStraight")

		# self.DTWModel['1']['N']['RollingDown']  = self.LoadData(DirSetting+Name+"RollingDown2.dat","RollingDown")
		self.DTWModel['0']['N']['Right'] = self.LoadData(DirSetting+Name+"RightStraight1.dat","Right")
		self.DTWModel['0']['N']['RightGoStraight'] = self.LoadData(DirSetting+Name+"RightGoStraight0.dat","RightGoStraight")
		self.DTWModel['0']['N']['LeftBackStraight'] = self.LoadData(DirSetting+Name+"LeftBackStraight0.dat","LeftBackStraight")
	
		self.DTWModel['1']['P']['UpStraight']= self.LoadData(DirSetting+Name+"UpStraight0.dat","UpStraight")
		self.DTWModel['1']['P']['BackStraight'] = self.LoadData(DirSetting+Name+"BackStraight0.dat","BackStraight")
		self.DTWModel['0']['P']['Left'] = self.LoadData(DirSetting+Name+"LeftStraight0.dat","Left")
		self.DTWModel['0']['P']['LeftGoStraight'] = self.LoadData(DirSetting+Name+"LeftGoStraight1.dat","LeftGoStraight")
		self.DTWModel['0']['P']['RightBackStraight'] = self.LoadData(DirSetting+Name+"RightBackStraight2.dat","RightBackStraight")
		
		
		self.DTWModel['1']['N']['DownStraight'] = self.LoadData(DirSetting+Name+"DownStraight0.dat","DownStraight")

		self.PassItem = []
	
		pickle.dump(self.DTWModel,open('JayDTW.dat', "wrb"))
		pickle.dump(self.AnglePose,open('JayAnglePose.dat', "wrb"))

	def LoadData(self,fileName,key):
		''' load data
			we catch gyro here but we do not use it to compare 
		'''
		fp = open(fileName, "rb")		           
		tempDict = pickle.load(fp)	
		self.AnglePose[key] = []
		for idx in self.AnglePosePoint[key]:
			self.AnglePose[key].append( [tempDict['Angle'][0,idx] , tempDict['Angle'][-1,idx] ]  )
		print fileName,tempDict['Angle'][0,] - tempDict['Angle'][tempDict['Angle'].shape[0]-1,],tempDict['Angle'][0,]
		ret=np.zeros((0,6))
		offset = tempDict['workingIdx'][0][0]
		tempDict['Acc'] = preprocessing.normalize(tempDict['Acc'] , norm='l2')
		tempDict['Gyo'] = preprocessing.normalize(tempDict['Gyo'] , norm='l2')
		for i,idx in enumerate(tempDict['workingIdx']):
			acc = np.mean(tempDict['Acc'][idx[0]-offset:idx[1]-offset,:],axis=0)
			gyo = np.mean(tempDict['Gyo'][idx[0]-offset:idx[1]-offset,:],axis=0)
			mix = np.concatenate ((np.asmatrix(acc),np.asmatrix(gyo)) ,axis=1 )
			ret = np.concatenate( (ret,mix) ,axis=0)
		ret =  np.asmatrix(ret)
		return ret.tolist()
		# return tempDict['Acc'].tolist()
		
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
		if writeout==False:	
			print self.workingIdx,len(self.workingIdx)
		else:			
			self.logfp.write("workingIdx:"+str(len(self.workingIdx))+'\n')

		startIDX = self.workingIdx[0][0] 
		endIDX = self.workingIdx[-1][1]

		dataList = np.concatenate((self.Acc,self.Angle),axis=1)
		dataList = dataList.transpose()
		# print endIDX-startIDX,dataList.shape
		JudgeStartTime = time.time()
		FinalS,MayGesture = self.Judge(writeout)
		JudgeEndTime = time.time()
		if writeout == False:
			print "\033[0;32mIt cost %f sec" % (JudgeEndTime - JudgeStartTime),"\033[0;0m"
		else:
			costTime = float(JudgeEndTime - JudgeStartTime)
			self.timeLogfp.write(str(costTime)+'\n')
		if paint == True:
			# print self.Angle
			for data,curve,yData,i in zip (dataList,self.curveList,self.curveYDataList ,range(0,7)): 
				# print len(yData)
				yData[0:endIDX-startIDX] = dataList[i,:].tolist()[0]
				# print len(dataList[i,:].tolist())
				curve.setData(y=yData, x=self.curveXData)
			self.app.processEvents()
		# print "-----------------------------"
		return FinalS,MayGesture
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
		'''
			sort all element of VarDataIdx by end windows index 
			sort result : OverlapList
		'''
		startTime = time.time()

		''' A element of self.VarDataIdx['0'] : ['P', 0, 28, 73.506342260781992, '0']
			A element of ['P', 0, 28, 73.506342260781992, '0']:
				1: Angle Direction --   P : the angle increasing  N : the angle decreasing
				2: Start windows index
				3: End windows index
				4: the magnitude of angle
				5: which axis (yaw(0), pitch(1) or roll(2))

		'''
		var0 = self.VarDataIdx['0']
		var1 = self.VarDataIdx['1']
		var2 = self.VarDataIdx['2']
		yawList = []
		yawIndex = 0
		pitchList = []
		pitchIndex = 0
		rollList = []
		rollIndex = 0
		self.Judge_Append(var0,yawList,'0')
		self.Judge_Append(var1,pitchList,'1')
		self.Judge_Append(var2,rollList,'2')
		
		print yawList
		print pitchList
		print rollList
		OverlapList = [ [data for data in yawList] ]

		'''insert pitch information to overlapList'''
		index = []
		for pitchData in pitchList:
			for data,i in zip(OverlapList[0],range(0,2000)):
				if pitchData[2] <= data[2] :
					index.append(i)
					break
		offset = 0
		print index
		for pitchData,i in zip(pitchList,index):			
			OverlapList[0].insert(i+offset,pitchData)		
			offset = offset +1 

		'''insert roll information to overlapList by end windows index'''
		index = []
		for rollData in rollList:
			for data,i in zip(OverlapList[0],range(0,2000)):
				if rollData[2] <= data[2] :
					index.append(i)
					break
		offset = 0
		print index
		for rollData,i in zip(rollList,index):			
			OverlapList[0].insert(i+offset,rollData)		
			offset = offset +1 

		'''
		After sort: 
			OverlapList:
				[[['P', 0, 1, 0.78270572569437502, '2'], ['P', 0, 1, 2.9600994785584298, '1'], 
				['N', 0, 4, -8.5205035486508294, '0'], ['N', 2, 9, -18.527294551724459, '1'], 
				['P', 10, 10, 0.0, '1'], ['N', 11, 11, 0.0, '1'], ['N', 2, 15, -7.4750121338771427, '2'], 
				['P', 12, 20, 3.0272416349145912, '1'], ['P', 16, 21, 2.3813215120928319, '2'], 
				['N', 21, 21, 0.0, '1'], ['P', 5, 21, 32.249938788706338, '0']]]
		'''

		#judge the each element of OverlapList
		FinalS = self.JudgeGesture(OverlapList,writeout)
		FinalS,MayGesture = self.peakProcess(FinalS,writeout)
		if writeout == False:
			print "\033[0;32m(Judge)MayBeValid:",self.MayBeValid,"\033[0;0m"
			print OverlapList
			print "--------------------------------"
		else:
			self.logfp.write( "\033[0;32m(Judge)MayBeValid:"+str(self.MayBeValid)+"\033[0;0m\n")

		return FinalS,MayGesture

	def Judge_Append(self,var0,ListData,index):
		for dataA,dataB in zip(var0['N'],var0['P']):
			if dataA[0] < dataB[0]:
				ListData.append(['N',dataA[0],dataA[1],dataA[2][0,0],index])
				ListData.append(['P',dataB[0],dataB[1],dataB[2][0,0],index])
			else:
				ListData.append(['P',dataB[0],dataB[1],dataB[2][0,0],index])
				ListData.append(['N',dataA[0],dataA[1],dataA[2][0,0],index])
		if len(var0['N']) > len(var0['P']):
			data = var0['N'][-1]
			ListData.append(['N',data[0],data[1],data[2][0,0],index])
		elif len(var0['P']) > len(var0['N']):
			data = var0['P'][-1]
			ListData.append(['P',data[0],data[1],data[2][0,0],index])

	def ThresholdTune(self,startIDX,endIDX,index):
		'''filter out the     '''
		# print "ThresholdTune"
		thres = 1.0
		if index == '0':
			thres = 2.0
		List = []
		for i in range(0,endIDX-startIDX):
			startWDS = self.workingIdx[startIDX+i]
			endWDS = self.workingIdx[startIDX+i+1]
			List.append(self.Angle[endWDS[1]-1,int(index)]-self.Angle[startWDS[0],int(index)])
		print "\033[1;31m" ,index,startIDX,endIDX,np.mean(List),"\033[0;0m"
		# print List
		for i in range(0,len(List)):
			if np.abs(List[i]) > thres:
				startIDX = startIDX + i
				break
		count = 0
		for i in range(len(List)-1,-1,-1):
			if np.abs(List[i]) > thres:
				endIDX = endIDX - count
				break
			count = count + 1
		print "\033[1;36m" ,startIDX,endIDX,"\033[0;0m"
		return startIDX,endIDX

	def JudgeGesture(self,OverlapList,writeout=False):
		
		FinalS = []		
		MixList= []
		disOrder = []
		angleOrder = []
		Angleidx = [ [0,1,2] , [1,0,2], [2,0,1] ]

		for data in OverlapList[0]:
			if data[2] - data[1] <5:
				continue
			if (np.abs(data[3]) > 6 and data[4] == '1') or (np.abs(data[3]) > 14 and data[4] == '0') or (np.abs(data[3]) > 14 and data[4] == '2'):
				self.PassItem.append(data)

				data[1],data[2] = self.ThresholdTune(data[1],data[2],data[4])
				startIDX = self.workingIdx[data[1]][0] 
				endIDX = self.workingIdx[data[2]][1]
				

				''' data preprocessing'''
				A = np.zeros((0,6))
				self.Acc[startIDX:endIDX,:] = preprocessing.normalize(self.Acc[startIDX:endIDX,:] , norm='l2')
				self.Gyo[startIDX:endIDX,:] = preprocessing.normalize(self.Gyo[startIDX:endIDX,:] , norm='l2')
				for idx in self.workingIdx[data[1]:data[2]+1]:
					acc = np.mean(self.Acc[idx[0]:idx[1],:],axis=0)
					gyo = np.mean(self.Gyo[idx[0]:idx[1],:],axis=0)
					mix = np.concatenate ((acc,gyo) ,axis=1 )
					A = np.concatenate( (A, mix ) ,axis=0)
				A = np.asmatrix(A)
				A = A.tolist()
				lowestDis = 2000
				lowestAngle = 2000
				gesture = None
				disOrderChildList = [[],[]]
				angleOrderChildList = [[],[]]

				if data[3] < 0:

					for key in self.DTWModel[data[4]]['N'].keys():						

						if self.DTWModel[data[4]]['N'][key] != None:
							# distance, path = DTW.distance(A, self.DTWModel[data[4]]['N'][key], dist= absDist)
							distance = DTW.distance(A, self.DTWModel[data[4]]['N'][key])
							# constraint_distance
							disOrderChildList[0].append(distance)
							disOrderChildList[1].append(key)
							gesture = disOrderChildList[1][np.argsort (disOrderChildList[0])[0]]
							lowestDis = disOrderChildList[0][np.argsort (disOrderChildList[0])[0]]


							diff = np.abs( np.abs(self.AnglePose[key][0][1]-self.AnglePose[key][0][0]) - np.abs(self.Angle[startIDX,:] - self.Angle[endIDX-1,:])[0, int(data[4]) ] )
							angleOrderChildList[0].append(diff)	
							angleOrderChildList[1].append(key)						
				else:

					for key in self.DTWModel[data[4]]['P'].keys():						

						if self.DTWModel[data[4]]['P'][key] != None:								
							# distance, path = DTW.distance(A, self.DTWModel[data[4]]['P'][key], dist=absDist)

							distance = DTW.distance(A, self.DTWModel[data[4]]['P'][key])

							disOrderChildList[0].append(distance)
							disOrderChildList[1].append(key)
							gesture = disOrderChildList[1][np.argsort (disOrderChildList[0])[0]]
							lowestDis = disOrderChildList[0][np.argsort (disOrderChildList[0])[0]]
							
							diff = np.abs( np.abs(self.AnglePose[key][0][1]-self.AnglePose[key][0][0]) - np.abs(self.Angle[startIDX,:] - self.Angle[endIDX-1,:])[0, int(data[4]) ] )
							angleOrderChildList[0].append(diff)	
							angleOrderChildList[1].append(key)

				FinalS = self.MergeWindows(FinalS,gesture,lowestDis,disOrderChildList,angleOrderChildList,disOrder,angleOrder,startIDX,endIDX,data)			
		return FinalS
	def MergeWindows(self,FinalS,gesture,lowestDis,disOrderChildList,angleOrderChildList,disOrder,angleOrder,startIDX,endIDX,data,writeout=False):
		''' judge if new gesture is an valid gesture 
			Args:
				FinalS : valid gesture list
				gesture : new gesture 
				lowestDis : new gesture distance

		'''
		if gesture != None:

			sortDis = np.argsort (disOrderChildList[0])
			Dis = []
			for idx in sortDis:
				Dis.append([disOrderChildList[0][idx],disOrderChildList[1][idx]])

			sortArg = np.argsort (angleOrderChildList[0])
			Arg = []
			for idx in sortArg:
				Arg.append([angleOrderChildList[0][idx],angleOrderChildList[1][idx]])
			
			disOrder.append(Dis)
			angleOrder.append(Arg)

			
			self.MayBeValid.append([gesture,lowestDis,data[1],data[2],data[4]])	
			#first gesture 
			if len(FinalS) == 0:	
				FinalS.append([gesture,lowestDis,data[1],data[2],data[4]])
			else:
				for i in range(len(FinalS)-1,-1,-1):
					item = FinalS[i]
					# print "item:",item,gesture,data[1],data[2],lowestDis
					#non overlap
					if data[1] >= item[3] and i == len(FinalS) -1:
						FinalS.append([gesture,lowestDis,data[1],data[2],data[4]])
						# print "xxxxx",gesture,data[1] ,item[3]
						break
					
					idx = [0,0]
					Nidx = [0,0]
					if item[2] >=  data[1]:
						Next = item
						Prev = data
						Pidx = [1,2]
						Nidx = [2,3]
					else:
						Next = data
						Prev = item
						Pidx = [2,3]
						Nidx = [1,2]

					Datalength = Prev[Pidx[1]] - Prev[Pidx[0]]
					thres = float(Datalength/2)
					print "(MergeWindows)",gesture,thres,Prev[ Pidx[1] ],Next[ Nidx[0] ]
					if Prev[ Pidx[1] ] - Next[ Nidx[0] ] >= thres or (  ( Next[ Nidx[0] ] <  Prev[ Pidx[0] ] or np.abs(Next[ Nidx[0] ] -  Prev[ Pidx[0] ])<=2   )  and (Next[ Nidx[1] - Prev[ Pidx[1]]] <=2) ) or (  ( Prev[ Pidx[0] ] < Next[ Nidx[0] ]) and np.abs(Next[ Nidx[1] ] - Prev[ Pidx[1] ])<=2  ):
						# print "(MergeWindows)overlap:" ,gesture,lowestDis,item[0],"be replaced!!"
						if lowestDis < item[1]:
							print "(MergeWindows)Replaced:" ,gesture,lowestDis,item[0],"be replaced!!"
							FinalS[i] = [gesture,lowestDis,data[1],data[2],data[4]]

						break
					elif i == 0:
						FinalS.append([gesture,lowestDis,data[1],data[2],data[4]])
						break
		else:
			pass

		return FinalS
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
		print filename

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

		AstartIDX = self.workingIdx[AstartWindowsIdx][0] 
		AendIDX = self.workingIdx[AendWindowsIdx][1]


		A = self.Acc[startIDX:endIDX,:].tolist()[::self.ScalerNum]
		# print np.divide(A,[1,1,1])
		if AstartIDX== None or AendIDX == None:
			B = Acc[:,:].tolist()[::self.ScalerNum]
		else:
			B = Acc[AstartIDX:AendIDX,:].tolist()[::self.ScalerNum]

		distance = DTW.distance(A, B)
		# W1 = []
		# W2 = []
		# for data in path:
		# 	W1.append(data[0])
		# 	W2.append(data[1])
		# plt.plot(W1,W2)
		# plt.title('DTW', fontsize=20)
		# plt.show()
		print startWindowsIdx,endWindowsIdx,AstartWindowsIdx,AendWindowsIdx
		print(distance)



	def peakProcess(self,FinalS,writeout=False):
		# print "(peakProcess)",self.windowsCrossDataIndex
		Axis = {}
		for gesture in FinalS:
			start = gesture[2]
			start = self.workingIdx[start][0]
			end = gesture[3]
			end = self.workingIdx[end][1]


		
		#using angle to judge which gesture 	
		for gesture in FinalS:
			start = self.workingIdx[gesture[2]][0]
			end = self.workingIdx[gesture[3]][1]

			if gesture[0] == 'GoStraight':
				LeftRightAngle = np.abs(np.abs(self.Angle[end-1,0]) - 45)
				GoAngle = np.abs(np.abs(self.Angle[end-1,0]) - 0)
				if LeftRightAngle < GoAngle:
					if self.Angle[end-1,0] < 0:
						gesture[0] = 'RightGoStraight'
					else:
						gesture[0] = 'LeftGoStraight'
				else:
					gesture[0] = 'GoStraight'
					
			# 	print gesture,self.Angle[end-1,:]
			elif gesture[0] == 'BackStraight': 
				LeftRightAngle = np.abs(np.abs(self.Angle[start,0]) - 45)
				BackAngle = np.abs(np.abs(self.Angle[start,0]) - 0)
				if LeftRightAngle < BackAngle:
					if self.Angle[start,0] < 0:
						gesture[0] = 'RightBackStraight'
					else:
						gesture[0] = 'LeftBackStraight'
				else:
					gesture[0] = 'BackStraight'

			elif gesture[0] == 'RightGoStraight': 
				RightAngle = np.abs(np.abs(self.Angle[end-1,0]) - 45)
				GoStraightAngle = np.abs(np.abs(self.Angle[end-1,0]) - 0)
				if RightAngle > GoStraightAngle:
					gesture[0] = 'GoStraight'

		Unused = []
		for unfined in self.MayBeValid:
			jax = False
			for validGesture in FinalS:
				# print "valid",validGesture
				if unfined[4] ==  validGesture[4] and unfined[2] == validGesture[2] and unfined[3] == validGesture[3]:
					jax = True
					break
			if jax == False:
				Unused.append(unfined)


		startIdx = 0

		Ret = self.RMSCompare()
		countState ={}
		countState['X'] = []
		countState['Y'] = []
		countState['Z'] = []

		currentState = None
		lastWindows = 0
		for i,ret in zip(range(0,2000),Ret):
			if currentState == None:
				currentState = ret
			else:
				if ret != currentState:
					countState[currentState].append([lastWindows,i-1])
					lastWindows = i 
			currentState = ret 

		print "\033[1;36m(peakProcess)FinalS:",FinalS,"\033[0;0m"
		if writeout == False:
			print "(peakProcess)Ret:",Ret
			print "(peakProcess)countState:",countState

		if writeout == True:
			self.logfp.write('\x1B[31;1m'+"Result:"+str(FinalS) + "\x1B[0m"+'\n')

		return FinalS,Unused
	 
		
	def RMSCompare(self):

		MaxList = []
		axis = ['X','Y','Z']
		for window in self.workingIdx:
			order = np.argsort( np.sum( np.abs(self.Acc[window[0]:window[1],]),axis=0) )[0,2]
			MaxList.append(axis[order])
		return MaxList
	def JudgeAll(self):
		''' do recognition for every test case '''
		global Name
		
		singleGesture = ['LeftStraight','RightStraight','DownStraight','UpStraight','GoStraight','BackStraight','LeftGoStraight','LeftBackStraight','RightGoStraight','RightBackStraight']
		singleGestureDict = collections.OrderedDict()
		for key in singleGesture:
			key = Name + key
			singleGestureDict[key] =[]

		MultiGesture = ['ForwardBackward','BackwardForward','DownUp','UpDown','RightLeft','LeftRight','LeftForward','RightForward','UpRight','UpLeft','V','VII','LeftbackForward','RightbackForward','ForwardRight','ForwardLeft','LeftLeftForward','RightRightForward','ForwardUp','DownBackward','DownLeft','DownRight','ForwardBackwardForward','LeftRightUp','BackwardRightLeftforward','DownLeftRightforward','RightUpForward']
		MultiGestureDict = collections.OrderedDict()
		for key in MultiGesture:
			key = Name + key
			MultiGestureDict[key] =[]

		for file in self.fileList:
			gestureName = re.split(r'(\d+)', file)[0]
			# print gestureName
			if singleGestureDict.get(gestureName)!= None:
				# print gestureName
				fileInfo = {}
				fileInfo['name'] = file
				fileInfo['logit'] = gestureName # true lable
				fileInfo['pred'] = None # the prediction gesture
				fileInfo['MaybeGesture'] = None # some gesture that 
				singleGestureDict[gestureName].append(fileInfo) 

			if MultiGestureDict.get(gestureName)!= None:
				# print gestureName
				fileInfo = {}
				fileInfo['name'] = file
				fileInfo['logit'] = gestureName
				fileInfo['pred'] = None
				fileInfo['MaybeGesture'] = None
				MultiGestureDict[gestureName].append(fileInfo)				
		


		for key in 	['JayDownLeft']:
			for FileItem in MultiGestureDict[key]:
				''' recognize the gesture for file '''
				FileItem['pred'],FileItem['MaybeGesture'] = self.FileJudge(FileItem['name'],writeout=True)


		for data in MultiGestureDict[Name + 'DownLeft']:
			used = []
			print data['name'],"logit:", data['logit']
			# print "\033[1;31m predict:", data['pred'],"\033[0;0m"
			print "\033[1;36m MaybeGesture:", data['MaybeGesture'],"\033[0;0m"

			correlated = []
			for mayGesture in data['MaybeGesture']:
				correlated.append([mayGesture])
				for preGesture in data['pred']:
					if np.abs( mayGesture[2] - preGesture[2]) < 4 or np.abs( mayGesture[3] - preGesture[3]) < 4:
						correlated[-1].append(preGesture)

			# print correlated
			# [['N', 0, 11, -57.723844525539761, '0']
			self.loadFile(data['name'])
			for Alist in correlated:

				if len(Alist) == 1:
					continue
				mayGesture = Alist[0]
				minimumWin = [2000,-2]
				maximumWin = [0,-2]
				replaced = False
				for i in range(1,len(Alist)):
					preGesture =  Alist[i]
					startWDS = preGesture[2]
					endWDS = preGesture[3]
					I = np.mean(self.Angle[self.workingIdx[startWDS][0]:self.workingIdx[startWDS][1],int(mayGesture[4])],axis=0)
					J = np.mean(self.Angle[self.workingIdx[endWDS][0]:self.workingIdx[endWDS][1],int(mayGesture[4])],axis=0)					
					result = self.judgeonce(['X',startWDS,endWDS,(J-I),mayGesture[4]])
					if preGesture[2] < minimumWin[0] :
						minimumWin[0] = preGesture[2] 
						minimumWin[1] = i
					if preGesture[3] > maximumWin[0]:
						maximumWin[0] = preGesture[3]
						maximumWin[1] = i

					if result != None and result[1] < preGesture[1] :
						replaced = True
						for i in range(0,len(data['pred'])):
							dis = data['pred'][i][1]
							Swin = data['pred'][i][2]
							Ewin = data['pred'][i][3]
							if dis == preGesture[1] and Swin == preGesture[2] and Ewin ==  preGesture[3]:
								mayGesture.append(True)	
								data['pred'][i] = result  
								break
						
				''' exception process here'''
				mayGesture =  Alist[0]
				preGesture = Alist[ maximumWin[1] ]
				if np.abs( mayGesture[3] - preGesture[3]) >= 4 :
					startWDS = preGesture[3]
					endWDS = mayGesture[3]
					I = np.mean(self.Angle[self.workingIdx[startWDS][0]:self.workingIdx[startWDS][1],int(mayGesture[4])],axis=0)
					J = np.mean(self.Angle[self.workingIdx[endWDS][0]:self.workingIdx[endWDS][1],int(mayGesture[4])],axis=0)						
					result = self.judgeonce(['X',startWDS,endWDS,(J-I),mayGesture[4]])
					# print J-I,mayGesture[4]	,startWDS,endWDS	
					if 	result!= None:
						mayGesture.append(True)											
						data['pred'].append(result)

				mayGesture =  Alist[0]
				preGesture = Alist[ minimumWin[1] ]
				if np.abs( mayGesture[2] - preGesture[2]) >= 4 :
					startWDS = mayGesture[2]
					endWDS = preGesture[2]
					I = np.mean(self.Angle[self.workingIdx[startWDS][0]:self.workingIdx[startWDS][1],int(mayGesture[4])],axis=0)
					J = np.mean(self.Angle[self.workingIdx[endWDS][0]:self.workingIdx[endWDS][1],int(mayGesture[4])],axis=0)						
					result = self.judgeonce(['X',startWDS,endWDS,(J-I),mayGesture[4]])	
					# print J-I,mayGesture[4]	,startWDS,endWDS											
					if 	result!= None:	
						mayGesture.append(True)									
						data['pred'].append(result)


			print "\033[1;31m predict:", data['pred'],"\033[0;0m"
			gesture = []
			for pred in data['pred'] :
				for X in data['pred']:
					if pred != X:
						if (pred[2] <= X[2] or np.abs(pred[2] - X[2])<=2 ) and (pred[3] > X[3]): 
							gesture.append(pred)
							gesture.append(X)
			
			print gesture

	def loadFile(self,fileName):
		'''load file '''
		try:
			fp = open(ComplexDirSetting+fileName, "rb")
		except:
			fp = open(DirSetting+fileName, "rb")
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

	def judgeonce(self,data):

		if data[2] - data[1] <5:
			return None
		if np.abs(data[3]) > 10 :
			
			startIDX = self.workingIdx[data[1]][0] 
			endIDX = self.workingIdx[data[2]][1]
			#A = stats.zscore(np.concatenate( (self.Acc[startIDX:endIDX,:],self.Gyo[startIDX:endIDX,:]),axis=1), axis=1, ddof=0).tolist()[::self.ScalerNum]
			# A = self.Acc[startIDX:endIDX,:].tolist()[::self.ScalerNum]
			A = np.zeros((0,6))
			self.Acc[startIDX:endIDX,:] = preprocessing.normalize(self.Acc[startIDX:endIDX,:] , norm='l2')
			self.Gyo[startIDX:endIDX,:] = preprocessing.normalize(self.Gyo[startIDX:endIDX,:] , norm='l2')
			for idx in self.workingIdx[data[1]:data[2]+1]:
				acc = np.mean(self.Acc[idx[0]:idx[1],:],axis=0)
				gyo = np.mean(self.Gyo[idx[0]:idx[1],:],axis=0)
				mix = np.concatenate ((acc,gyo) ,axis=1 )
				A = np.concatenate( (A, mix ) ,axis=0)
			A = A.tolist()
			lowestDis = 2000
			lowestAngle = 2000
			gesture = None
			disOrderChildList = [[],[]]
			angleOrderChildList = [[],[]]

			if data[3] < 0:
				# AngleDiff = self.Angle[endIDX-1,:] - self.Angle[startIDX,:]
				for key in self.DTWModel[data[4]]['N'].keys():
					if self.DTWModel[data[4]]['N'][key] != None:
						# distance, path = DTW.distance(A, self.DTWModel[data[4]]['N'][key], dist= absDist)
						# distance = DTW.distance(A, self.DTWModel[data[4]]['N'][key])
						distance = DTW.distance(A, self.DTWModel[data[4]]['N'][key])
						if distance < lowestDis:
							lowestDis = distance
							gesture = key
						# constraint_distance
					
			else:
				# AngleDiff = self.Angle[endIDX-1,:] - self.Angle[startIDX,:]
				for key in self.DTWModel[data[4]]['P'].keys():
					if self.DTWModel[data[4]]['P'][key] != None:								
						# distance, path = DTW.distance(A, self.DTWModel[data[4]]['P'][key], dist=absDist)						
						distance = DTW.distance(A, self.DTWModel[data[4]]['P'][key])
						if distance < lowestDis:
							lowestDis = distance
							gesture = key						
			# ['Right', 8.652159129005181, 0, 11, '0']			
			return [gesture,lowestDis,data[1],data[2],data[4]]

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