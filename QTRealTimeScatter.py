from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import random

class MyRealTimeScatterPlot():
	def __init__(self,dataSource = None,nameList=['Plot1','Plot2','Plot3']):   
		#Widget Setting
		#pg.setConfigOption('background', 'w')
		self.numOfDataToPlot = 200
		self.plotViewList = []
		self.penStyleList= [(255,5,5),(200,200,100),(195,46,212),(0,0,200),(237,177,32)] 
		self.count=0  
		self.curveList = []
		self.curveXData =[i for i in range(0,self.numOfDataToPlot) ] 
		self.curveYDataList=[]
		self.app = QtGui.QApplication([])
		self.mainWindow = QtGui.QMainWindow()
		self.mainWindow.setWindowTitle('pyqtgraph example: PlotWidget')
		self.mainWindow.resize(800,800/len(nameList))
		self.view = pg.GraphicsLayoutWidget() ## GraphicsView with GraphicsLayout inserted by default
		self.mainWindow.setCentralWidget(self.view)
		# layout = QtGui.QVBoxLayout()
		# self.GuiWiget.setLayout(layout)

		for i,name in zip(range(0,3),nameList):
			plotView = self.view.addPlot(name=name)
			plotView.setRange(xRange=[-600, 600], yRange=[-600, 600])
			self.plotViewList.append(plotView)
			# plotWidget = pg.PlotWidget(name=name)  ## giving the plots names allows us to link their axes together
			# plotWidget.setXRange(0, self.numOfDataToPlot)
			# plotWidget.setYRange(-1, 1)		
			# layout.addWidget(plotWidget)
			# self.plotWidgetList.append(plotWidget)

		# Display the widget as a new window
		self.mainWindow.show()



		#Draw Setting
		for plotView,penStyle in zip(self.plotViewList,self.penStyleList):
			curve=pg.ScatterPlotItem(size=1)
			curve.setPen(penStyle)
			self.curveList.append(curve)
			plotView.addItem(curve)

			# curve = plotWidget.plot()
			# curve.setPen(penStyle)
			# curveYData =[np.NAN for i in range(0,self.numOfDataToPlot) ] 
			
			# self.curveYDataList.append(curveYData)


		

	def close(self):
		self.app.closeAllWindows() 
		self.app.quit()

	# dataList= [[[x],[y]] ,[[x],[y]] ,[[x],[y]] ] 
	def setMyData(self,dataList):
		# for data,curve,yData in zip (dataList,self.curveList,self.curveYDataList): 
		# 	if len(data) >= self.numOfDataToPlot:
		# 		curve.setData(y=data[:self.numOfDataToPlot], x=self.curveXData)
		# 	else:
		# 		yData[0:self.numOfDataToPlot-len(data)]=yData[len(data):self.numOfDataToPlot]
		# 		yData[self.numOfDataToPlot-len(data):self.numOfDataToPlot]=data
		# 	# yData[1:]=yData[0:39]
		# 	# yData[0]=data
		# 		curve.setData(y=yData, x=self.curveXData)
		self.count=self.count + 1
		if self.count == 51:
			for i,plotView,penStyle in zip (range(0,100),self.plotViewList,self.penStyleList):
				plotView.clear()
				self.curveList[i] = pg.ScatterPlotItem( size=1)
				self.curveList[i].setPen(penStyle)
				plotView.addItem(self.curveList[i])
				self.count=0

		for i,curve,plotView in zip (range(0,100),self.curveList,self.plotViewList):
			curve.addPoints(x=dataList[i], y=dataList[(i+1)%3])
		self.app.processEvents()
			# print yData
	def update(self):
		self.count=self.count + 1
		if self.count == 51:
			for i,plotView,penStyle in zip (range(0,100),self.plotViewList,self.penStyleList):
				plotView.clear()
				self.curveList[i] = pg.ScatterPlotItem( size=5)
				self.curveList[i].setPen(penStyle)
				plotView.addItem(self.curveList[i])
				self.count=0

		for curve,plotView in zip (self.curveList,self.plotViewList):
			curve.addPoints(x=[random.randint(-99,99)], y=[random.randint(-99,99)])
		# QtCore.QTimer.singleShot(1, self.update)
		# QtCore.QTimer.singleShot(1, self.update)

	def start(self):
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.update)
		self.timer.start(50)
		# start event
		self.app.instance().exec_()
        

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #QtGui.QApplication.instance().exec_()
        A = MyRealTimeScatterPlot()
        A.start()
        # A.app.instance().exec_()

        # QtGui.QApplication.instance().exec_()
