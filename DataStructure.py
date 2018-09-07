from scipy import signal
from scipy.stats import kurtosis, skew
import scipy
import collections
import pickle
import numpy as np
import math



def getRMS(data,axis=0):
    
    return np.sqrt(np.mean(np.multiply(data,data),axis=axis))

    # print np.sqrt()
    # return np.sqrt(np.mean(data**2))
#0:split in  row, 1 split in column 
def getEntropy(data,axis=0):
    idx = 0 
    entMatrix= np.zeros( (1,data.shape[1] ))
    entMatrix= np.asmatrix(entMatrix)
    Hdata = np.split(data,data.shape[1],axis=not axis)

    for datagram in Hdata:
        datagram = np.around(datagram,2)
        dataSet = collections.OrderedDict()
        freq_list = []
        for singleData in datagram:
            if dataSet.get(str(singleData)):
                dataSet[str(singleData)] +=  1
            else:
                dataSet[str(singleData)] = 1.

        for value in  dataSet.values():
            freq_list.append(float(value) / data.shape[0])
        
        # Shannon entropy
        ent = 0.0
        for freq in freq_list:
            ent =ent - freq * np.log2(freq)
        entMatrix[0,idx] = ent
        idx = idx + 1

    return entMatrix
def getDiff(A,B):
    return A - B 

    
timeDomainDict = {
    'max': np.amax,   #np.amax(a, axis=0)  Maxima along the first axis
    'min':np.amin,
    'mean':np.mean,        #np.mean(a, axis=0)
    'std':np.std,   #np.std(a)
    'maxdiffmin':getDiff,
    'rms':getRMS,
    'entropy':getEntropy, 
    'skewness':scipy.stats.skew,          #Skewness is a measure of symmetry, or more precisely, the lack of symmetry
    'correlationcoefficientXY': scipy.stats.pearsonr,   #correlation coefficient 
    'correlationcoefficientXZ': scipy.stats.pearsonr,
    'correlationcoefficientYZ': scipy.stats.pearsonr
}
frenquenceDomainDict = {
    
}
#self.value : [ [ ['-',xxx,xxx,xx]   ,['+',xxx,xxx,xx]  ] , []  ,[]       ]
class DataVar():
    def __init__(self):
        self.orderList = []
        self.VarList = [[],[],[]]
        self.VarListValue = [[],[],[]]
        self.value = [[],[],[]]
        self.transition ={}
        self.transition['0'] = []
        self.transition['1'] = []
        self.transition['2'] = []

        self.currentState = {}
        self.currentState['0'] = None
        self.currentState['1'] = None
        self.currentState['2'] = None

        #It describe the angle last windows (ex:self.dataIndex['0']['N']=[ [0,8],[19,25])
        self.dataIndex = {}
        self.dataIndex['0'] = {}
        self.dataIndex['1'] = {}
        self.dataIndex['2'] = {}
        self.dataIndex['0']['lastIndex'] = 0
        self.dataIndex['1']['lastIndex'] = 0
        self.dataIndex['2']['lastIndex'] = 0
        self.dataIndex['0']['N'] = []
        self.dataIndex['0']['P'] = []
        self.dataIndex['1']['N'] = []
        self.dataIndex['1']['P'] = []
        self.dataIndex['2']['N'] = []
        self.dataIndex['2']['P'] = []

        #ValidWindows: The gesture have been merged
        self.ValidWindows = []
        #AllGestureWindows: All of gesture pass the threshold
        self.AllGestureWindows = []
        self.VarGestureWindows = []
    def getLastState(self,i):
        return self.VarList[i][-1]
    def getLastInfo (self,i):
        return self.value[i]

    def Judge(self,Inst,windowsCrossDataIndex):
        for idx in ['0','1','2']:
            RestoreList = [[],[],[]]
            print idx
            for direction in ['N','P']:
                print direction
                for data in  self.dataIndex[idx][direction]:
                    if np.abs(data[2]) > 8 :
                        print data[2],"len:",data[0],data[1]

                        for i in range(data[0],data[1]+1):
                            startIdx = Inst.Windows.workingIDX[i][0]
                            end = Inst.Windows.workingIDX[i][1]
                            RestoreList[int(idx)].append([data[0],data[1],data[2]] )
                            # print np.argsort( np.abs( np.mean(Inst.dataBlockSet['Acc'].filteredXYZ[startIdx:end,:],axis=0 ) ))
                        # StateList[-1].append(endData - firstData)

                    else:
                        print "\033[1;31m",data[2],"len:",data[1] - data[0] + 1 ,data[0],data[1],"\033[1;m"
        self.Intersection(RestoreList[0],RestoreList[1],RestoreList[2])
          

    def transitionDir(self,axis,direction,windows,Angle):
        # idx = len(self.VarList[0]) - 3                    
        idx = len(self.VarList[0])
        thres = 1.0
        if axis == '0':
            thres = 2.0
        List = []
        for i in range(self.dataIndex[axis]['lastIndex'],idx):
            startWDS = windows[i]
            endWDS = windows[i + 1]
            List.append(Angle.filteredXYZ[endWDS[1]-1,int(axis)]-Angle.filteredXYZ[startWDS[0],int(axis)])

        for i in range(0,len(List)):
            if np.abs(List[i]) > thres:
                self.dataIndex[axis]['lastIndex'] = self.dataIndex[axis]['lastIndex'] + i
                break
        count = 0
        for i in range(len(List)-1,-1,-1):
            if np.abs(List[i]) > thres:
                idx = idx - count
                break
            count = count + 1

        self.dataIndex[axis][direction].append( [self.dataIndex[axis]['lastIndex'],idx ] )       
        self.dataIndex[axis]['lastIndex'] = idx + 1
        self.currentState[axis] = direction
        self.transition[axis] = []

    def reset(self):
        self.orderList = []
        self.VarList = [[],[],[]]
        self.value = [[],[],[]] 
        self.transition ={}
        self.transition['0'] = []
        self.transition['1'] = []
        self.transition['2'] = []

        self.currentState = {}
        self.currentState['0'] = None
        self.currentState['1'] = None
        self.currentState['2'] = None

        self.dataIndex = {}
        self.dataIndex['0'] = {}
        self.dataIndex['1'] = {}
        self.dataIndex['2'] = {}
        self.dataIndex['0']['lastIndex'] = 0
        self.dataIndex['1']['lastIndex'] = 0
        self.dataIndex['2']['lastIndex'] = 0
        self.dataIndex['0']['N'] = []
        self.dataIndex['0']['P'] = []
        self.dataIndex['1']['N'] = []
        self.dataIndex['1']['P'] = []
        self.dataIndex['2']['N'] = []
        self.dataIndex['2']['P'] = []

        self.ValidWindows = []
        self.AllGestureWindows = []
        self.VarGestureWindows = []
class AxisVar():
    def __init__(self):
        self.DataVar = {}
        self.DataVar['Acc']= DataVar()
        self.DataVar['Gyo']= DataVar()
        self.DataVar['Angle']= DataVar()

    def reset(self):
        self.DataVar['Acc'].reset()
        self.DataVar['Gyo'].reset()
        self.DataVar['Angle'].reset()



class RealTimeWindow():
    def __init__(self,timeInterval,OverLapInterval):
        self.windowsIdx = []
        self.filterdofRawIdx = []
        self.timeInterval = timeInterval
        self.overLapInterval = OverLapInterval
        self.realTimeStartIdx = 0
        self.overLapIdx = 0
        self.saveOverLapIdx = True

        self.state = {}
        self.state['Acc'] ={}
        self.state['Acc']['TransitionState'] = []
        self.state['Acc']['CurrState']= []
        self.state['Acc']['currSingleState'] = None
        self.state['Gyo'] ={}
        self.state['Gyo']['TransitionState'] = []
        self.state['Gyo']['CurrState'] = []
        self.state['Gyo']['currSingleState'] = None


        self.AccStraight = 0
        self.GyoRoll = 0
        self.RawAccStata = []     
        self.RawGyoStata = [] 

        self.IdxforFilterd = []
        self.workingData = np.zeros( (0,3) )
        self.workingIDX = []
        self.workingFilteredData = []
        self.plotNum = 0
        self.startPlotPoint = 0


        self.XIndex = 0
        self.YIndex = 0
        self.ZIndex = 0


    def RealTimeIdx(self,endIdx):
        self.windowsIdx.append([self.realTimeStartIdx,endIdx] )
    def SetfilterdIdx(self):
        self.filterdofRawIdx.append( self.windowsIdx[-1] )
    def RestartWorking(self):
        self.workingData = np.zeros( (0,3) )
        self.workingIDX = []
        self.IdxforFilterd = []
        self.workingFilteredData = []
    def SetWorkingIdx(self,idxList):
        self.workingIDX.append (idxList) 
    def SetcurrSingleState(self,accState,gyoState):
        self.state['Acc']['currSingleState'] = accState
        self.state['Gyo']['currSingleState'] = gyoState
    def ResetState(self):
        self.state = {}
        self.state['Acc'] ={}
        self.state['Acc']['TransitionState'] = []
        self.state['Acc']['CurrState']= []
        self.state['Acc']['currSingleState'] = None
        self.state['Gyo'] ={}
        self.state['Gyo']['TransitionState'] = []
        self.state['Gyo']['CurrState'] = []
        self.state['Gyo']['currSingleState'] = None   
        self.RawAccStata = []     
        self.RawGyoStata = [] 
        self.AccStraight = 0
        self.GyoStraight = 0


    def ResetZeroCrossData(self,idx):
        self.XIndex = idx
        self.YIndex = idx
        self.ZIndex = idx

        self.XN2PWindowsIndex = 0
        self.YN2PWindowsIndex = 0
        self.ZN2PWindowsIndex = 0       
        self.XP2NWindowsIndex = 0
        self.YP2NWindowsIndex = 0
        self.ZP2NWindowsIndex = 0 

        self.windowsCrossDataIndex = {}
        self.windowsCrossDataIndex['X'] = {}
        self.windowsCrossDataIndex['Y'] = {}
        self.windowsCrossDataIndex['Z'] = {}
        self.windowsCrossDataIndex['X']['N2P'] = []
        self.windowsCrossDataIndex['X']['P2N'] = []
        self.windowsCrossDataIndex['Y']['N2P'] = []
        self.windowsCrossDataIndex['Y']['P2N'] = []
        self.windowsCrossDataIndex['Z']['N2P'] = []
        self.windowsCrossDataIndex['Z']['P2N'] = []

class WindowsFeature():
    def __init__(self,data,timeInterval,OverLapInterval,windowsIdx,timeDomainKey):
        self.timeInterval = timeInterval
        self.overLapInterval = OverLapInterval
        self.windowsIdx = windowsIdx
        self.name = "interval"+str(self.timeInterval) + "overlap"+str(OverLapInterval)
        ##############################"max","min","mean","std","maxdiffmin","rms","correlationcoefficientXY","correlationcoefficientXZ","correlationcoefficientYZ"
        self.timeDomainKey=timeDomainKey  #ZRC" self.timeDomainKey=["max","min","mean","std","maxdiffmin","entropy","rms","skewness"]  ,"correlationcoefficientXY","correlationcoefficientXZ","correlationcoefficientYZ"
        self.featureNum = 0
        self.frequceDomainKey=[]
        self.timeDomainFeature=collections.OrderedDict()
        self.frequenceDomainFeature=collections.OrderedDict()

        self.windowsAndTimeFeature(data,self.windowsIdx)

    def windowsAndTimeFeature(self,data,windowsIdx):
        # print np.std(data[idx[0]:idx[1],0:1])
        WindowsTimeFeature = collections.OrderedDict()
        for key in self.timeDomainKey:
            if key == "correlationcoefficientXY" or key == "correlationcoefficientXZ" or key == "correlationcoefficientYZ":
                WindowsTimeFeature[key] = np.zeros((0,1))
                self.featureNum += 1
            else:
                WindowsTimeFeature[key] = np.zeros((0,data.shape[1]))
                self.featureNum += data.shape[1]
        # self.extractTimeDomainFeature(self.)    
        # print windowsIdx
        for idx in windowsIdx:
            # print data[idx[0]:idx[1],0:1] , data[idx[0]:idx[1],1:2]
            for key,value in WindowsTimeFeature.items():

                # print np.asmatrix(timeDomainDict["correlationcoefficientXY"](data[:,0:1] , data[:,1:2] )[0])
                if key == "correlationcoefficientXY" :
                    WindowsTimeFeature[key] = np.concatenate((WindowsTimeFeature[key],np.asmatrix(timeDomainDict[key](data[idx[0]:idx[1],0:1] , data[idx[0]:idx[1],1:2] )[0])),axis=0)
                elif key == "correlationcoefficientXZ" : 
                    WindowsTimeFeature[key] = np.concatenate((WindowsTimeFeature[key],np.asmatrix(timeDomainDict[key](data[idx[0]:idx[1],0:1] , data[idx[0]:idx[1],2:3] )[0])),axis=0)
                elif key == "correlationcoefficientYZ" :
                    WindowsTimeFeature[key] = np.concatenate((WindowsTimeFeature[key],np.asmatrix(timeDomainDict[key](data[idx[0]:idx[1],1:2] , data[idx[0]:idx[1],2:3] )[0])),axis=0)
                elif key == "maxdiffmin" :
                    WindowsTimeFeature[key] = timeDomainDict[key](WindowsTimeFeature['max'] ,WindowsTimeFeature['min'])
                    # print WindowsTimeFeature['max'] ,WindowsTimeFeature['min'],WindowsTimeFeature[key]
                else:
                    WindowsTimeFeature[key] = np.concatenate((WindowsTimeFeature[key],timeDomainDict[key](data[idx[0]:idx[1],:],axis=0)),axis=0)


                # WindowsTimeFeature[key] = np.asmatrix(self.timeDomainFeature[key])
        # print WindowsTimeFeature["correlationcoefficientXY"]
        # return WindowsTimeFeature
        self.timeDomainFeature = WindowsTimeFeature

        # # print timeDomainFeatureMatrix
        # return timeDomainFeatureMatrix

class DataBlock():
    def __init__(self):
        self.timestamp = []

        self.rawXYZ = np.zeros((0,3))
        self.rawX = np.zeros((0,1))
        self.rawY = np.zeros((0,1))
        self.rawZ = np.zeros((0,1))

        self.filteredXYZ = np.zeros((0,3))
        self.filteredX = np.zeros((0,1))
        self.filteredY = np.zeros((0,1))
        self.filteredZ = np.zeros((0,1))

        self.normalizedXYZ = np.zeros((0,3))
        self.normalizedX = np.zeros((0,1))
        self.normalizedY = np.zeros((0,1))
        self.normalizedZ = np.zeros((0,1))

        self.magnitude =np.zeros((0,1))
        self.windowsInst=None  #save windows instance

    def lowPassFilter(self, xData, N, Wn):
        b, a = signal.butter(N, Wn)
        yData = signal.filtfilt(b, a, xData)
        return yData  

    def lowpass(self):

        data = self.rawXYZ
        data = data.transpose()
        # print "low pass",self.rawXYZ.shape,data.shape
        ret = np.concatenate(( self.lowPassFilter(data[0,:] , 1, 0.08) , self.lowPassFilter(data[1,:] , 1, 0.08), self.lowPassFilter(data[2,:] , 1, 0.08)),axis=0)  
        self.filteredXYZ = np.asmatrix( ret.transpose() ) 

        return self.filteredXYZ

    def specifiedLowpass(self,startIdx,endIdx):

        data = self.rawXYZ[startIdx:endIdx,]
        if data.shape[0] >6:
            data = data.transpose()
            # print data.shape
            # print "low pass",self.rawXYZ.shape,data.shape
            ret = np.concatenate(( self.lowPassFilter(data[0,:] , 1, 0.08) , self.lowPassFilter(data[1,:] , 1, 0.08), self.lowPassFilter(data[2,:] , 1, 0.08)),axis=0)  
            self.filteredXYZ = np.concatenate(  (self.filteredXYZ , np.asmatrix( ret.transpose() ) ) ,axis=0 ) 
        else:
            self.filteredXYZ = np.concatenate(  (self.filteredXYZ , data ) ,axis=0 )    
        # print self.filteredXYZ .shape
        return self.filteredXYZ

    def getMagnitude(self):
        squareData = np.multiply(self.filteredXYZ,self.filteredXYZ) #square        
        SplitData=np.hsplit(squareData,squareData.shape[1])
        whichData = []
        for data in SplitData:
            if not "matrix" in  str(type(whichData)) :
                whichData=data
            else:
                whichData=whichData+data
        # whichData=whichData*(float(1)/float(3))
        whichData = np.sqrt(whichData)
        self.magnitude = whichData

    def createWindows(self,timeInterval,OverLapInterval,windowsIdx,feature):
        self.feature = feature
        windows = WindowsFeature( self.filteredXYZ ,timeInterval ,OverLapInterval,windowsIdx,feature)
        self.windowsInst.append(windows)

    def getWindowsInst(self):
        return self.windowsInst

    def GetRealTimeFeature(self,featureSet,AxisData,Naxis):
        ret = np.zeros( (1,0) ) 
        for feature in featureSet:
            if feature != 'maxdiffmin':
                ret = np.concatenate( ( ret,timeDomainDict[feature]( AxisData ,axis=0) ),axis=1 )
            else:
                ret = np.concatenate( ( ret,timeDomainDict[feature]( timeDomainDict['max']( AxisData ,axis=0) , timeDomainDict['min']( AxisData ,axis=0)) ),axis=1 ) 

        return ret

class DataController():
    def __init__(self,timeInterval,OverLap):
        for clazz in self.__class__.__bases__: 
            clazz.__init__(self, filename)
        # super(LabelDataSet,self).__init__(filename)
        # self.label = []
        # self.label.append(label)
        # self.fileName = filename
        # self.DataBlockList = [ "accerameter", "gyroscope","magnetometer","angle"]
        self.DataBlockList=['Acc','Gyo','Mag','Angle']#you should know how many dataSet
        self.choiceAxisParam=[0x7,0x7,0x7,0x7]#you should know  whick axis you want
        self.dataBlockSet ={}
        self.RecognizeFlag = 1

        self.dataBlockSet=collections.OrderedDict()
        for blockName in self.DataBlockList:
            self.dataBlockSet[blockName] = DataBlock() 
        self.timestamp = []

        self.linearACC=[]
        self.gravity = -0.964141136218

        self.Windows = RealTimeWindow (timeInterval,OverLap)
        self.static = True
        self.staticCount = 1
        self.sequence = []

    def readPickle(self):
        '''read file and get rawdata'''
        fp = open(self.fileName, "rb")

        while 1:
            try:
                tempDict = pickle.load(fp)
                if tempDict['Angle'][0, 0] < -180:
                    tempDict['Angle'][0, 0]=tempDict['Angle'][0, 0]+360
                tempDict['Mag'] = np.abs(tempDict['Mag'])

                for key,value in self.dataBlockSet.items():   

                    self.dataBlockSet[key].rawXYZ = np.concatenate( (self.dataBlockSet[key].rawXYZ , tempDict[key]) ,axis=0  )  
                    self.dataBlockSet[key].timestamp.append(tempDict['timestamp'])


            except:
                # print "readPickle last process procedure"
                for key,value in self.dataBlockSet.items(): 
                    self.dataBlockSet[key].rawXYZ = np.asmatrix( self.dataBlockSet[key].rawXYZ )          
                    self.dataBlockSet[key].rawX = self.dataBlockSet[key].rawXYZ[:,0:1]  
                    self.dataBlockSet[key].rawY = self.dataBlockSet[key].rawXYZ[:,1:2]
                    self.dataBlockSet[key].rawZ = self.dataBlockSet[key].rawXYZ[:,2:3] 
                    self.timestamp = self.dataBlockSet[key].timestamp
                    # if we do not do low pass  => rawXYZ = filteredXYZ
                    self.dataBlockSet[key].filteredXYZ = self.dataBlockSet[key].rawXYZ
                break 


    def SetChoicePara(self,datapara,axispara):
        self.choiceDataParam=datapara
        self.choiceAxisParam=axispara

    def lowpass(self):
        for key,value in self.dataBlockSet.items(): 
            if key == "Acc" or key == "Gyo" :
                lowPassResult = self.dataBlockSet[key].lowpass()
                self.dataBlockSet[key].filteredX = lowPassResult[:,0:1]
                self.dataBlockSet[key].filteredY = lowPassResult[:,1:2]
                self.dataBlockSet[key].filteredZ = lowPassResult[:,2:3]

    def specifiedLowpass(self,startIdx,endIdx):
        temp = self.dataBlockSet["Acc"].filteredXYZ.shape[0]
        for key,value in self.dataBlockSet.items(): 
            if key == "Acc" or key == "Gyo" :
                lowPassResult = self.dataBlockSet[key].specifiedLowpass(startIdx,endIdx)
                self.dataBlockSet[key].filteredX = np.concatenate(  (self.dataBlockSet[key].filteredX,lowPassResult[startIdx:endIdx,0:1]),axis=0  )
                self.dataBlockSet[key].filteredY = np.concatenate(  (self.dataBlockSet[key].filteredY,lowPassResult[startIdx:endIdx,1:2]),axis=0  )
                self.dataBlockSet[key].filteredZ = np.concatenate(  (self.dataBlockSet[key].filteredZ,lowPassResult[startIdx:endIdx,2:3]),axis=0  )
        idx = self.dataBlockSet["Acc"].filteredXYZ.shape[0]    
         
        self.Windows.IdxforFilterd.append( [temp   ,idx ] )       
        # print self.Windows.IdxforFilterd
    def getMagnitude(self):
        self.dataBlockSet["Acc"].getMagnitude()
        self.dataBlockSet["Gyo"].getMagnitude()

    def createWindows(self,timeInterval,OverLapInterval):
        if OverLapInterval >= timeInterval:
            raise "OverLap Interval must be smaller than timeInterval" 
        windowsIdx = []
        dataLen = len(self.timestamp)
        startIdx = 0
        temp = 0
        startTime = self.timestamp[0] 
        endTime = startTime + timeInterval
        run = 0
        for i,time in zip ( range(0,dataLen),self.timestamp ):

            if  time >= endTime - OverLapInterval and run==0 :            
                startTime = time
                temp = i
                run = 1

            if time >= endTime or i == dataLen -1:
                if i - startIdx < 3:
                    continue
                windowsIdx.append([startIdx,i])
                startIdx = temp
                endTime = startTime + timeInterval
                run = 0
                # print startTime , endTime,time
        # print windowsIdx

        for key,value in self.dataBlockSet.items(): 

            if key == "Acc" or key == "Gyo":
                self.dataBlockSet[key].createWindows(timeInterval,OverLapInterval,windowsIdx,["max","min","mean","std","maxdiffmin","rms","correlationcoefficientXY","correlationcoefficientXZ","correlationcoefficientYZ"])
            else:
                self.dataBlockSet[key].createWindows(timeInterval,OverLapInterval,windowsIdx,["max","min","mean","std","maxdiffmin","rms"])
    def fillLabelList(self,axis=0):
        mylabel=self.label[0]
        for i in range (1,self.dataBlockSet["Acc"].filteredXYZ.shape[axis]):
            self.label.append(mylabel)

    def mergeRawData(self,usedElement):
        #format : [[acc,gyo,mag,angle]]
        # ret = np.zeros( (self.dataBlockSet[ self.DataBlockList[0] ].rawXYZ.shape[0],0) )
        # for element in usedElement:
            # ret =np.concatenate( (ret,self.dataBlockSet[element].rawXYZ),axis=1 )   
        return np.asmatrix(self.dataBlockSet[ usedElement[0] ].rawXYZ)

    def mergeFilteredData(self,usedElement):
        #format : [[acc,gyo,mag,angle]]
        ret = np.zeros( (self.dataBlockSet[ self.DataBlockList[0] ].filteredXYZ.shape[0],0) )
        for element in usedElement:
            ret =np.concatenate( (ret,self.dataBlockSet[element].filteredXYZ),axis=1 )   
        return np.asmatrix(ret)

    def mergeNormalizedData(self,usedElement):
        #format : [[acc,gyo,mag,angle]]
        ret = np.zeros( (self.dataBlockSet[ self.DataBlockList[0] ].normalizedXYZ.shape[0],0) )
        for element in usedElement:
            ret =np.concatenate( (ret,self.dataBlockSet[element].normalizedXYZ),axis=1 )   
        return np.asmatrix(ret)

    #usedfeature :a dict for every element
    def mergeWindowsFeatrue(self,usedElement,usedfeature,idx):
        Inst = self.dataBlockSet[ usedElement[0] ].windowsInst[idx]
        ret = np.zeros( ( Inst.timeDomainFeature[ Inst.timeDomainFeature.keys()[0] ].shape[0],0) )
        # timeDomainFeatureMatrix = np.asmatrix(timeDomainFeatureMatrix)
        #format :[[ acc feature, gyro feature,mag feature, Angle feature]]
        for element in usedElement:
            winInstDict = self.dataBlockSet[ element ].windowsInst[idx].timeDomainFeature
            for feature in usedfeature[element]:
                ret = np.concatenate( (ret,winInstDict[feature]),axis=1  ) 
 
        return np.asmatrix(ret)

    def getWindowsIdx(self,idx):
        return self.dataBlockSet[ self.DataBlockList[0] ].getWindowsInst()[idx].windowsIdx

        # print self.linearACC[1,2:3],self.mergeRawData[1,3:6][0,2:3]
    def toGlobalFrameData(self,usedElement):
        
        dataBlock = self.mergeRawData(usedElement)

        Angle = self.mergeFilteredData(["Angle"])
        for i,angle in zip(range(0,dataBlock.shape[0]),Angle):
            inverseMatrix = self.EularRotateMatrixInverse(angle)
            dataBlock[i,:] = dataBlock[i,:] * inverseMatrix
        dataBlock = self.mergeRawData(usedElement)
        
    def EularRotateMatrixInverse(self,angles):
        rotmat = [[0,0,0],[0,0,0],[0,0,0]]
        Yaw =  angles[0,0] * 0.01745329251
        Pitch =  angles[0,1] * 0.01745329251
        Roll =  angles[0,2] * 0.01745329251

        cosYaw = math.cos(Yaw)
        cosPitch = math.cos(Pitch)
        cosRoll = math.cos(Roll)
        sinYaw = math.sin(Yaw)
        sinPitch = math.sin(Pitch)
        sinRoll = math.sin(Roll)


        rotmat[0][0] = cosYaw * cosPitch 
        rotmat[0][1] = sinYaw * cosPitch
        rotmat[0][2] = -sinPitch
        rotmat[1][0] = -sinYaw*cosRoll+cosYaw*sinPitch*sinRoll
        rotmat[1][1] = cosYaw * cosRoll + sinYaw*sinPitch*sinRoll
        rotmat[1][2] = cosPitch  * sinRoll 
        rotmat[2][0] = sinYaw * sinRoll + cosYaw*sinPitch* cosRoll  
        rotmat[2][1] = -cosYaw * sinRoll + sinYaw * sinPitch * cosRoll      
        rotmat[2][2] = cosPitch  * cosRoll

        return np.asmatrix(rotmat)
    def getTimeStamp(self,idx):
        return self.timestamp[idx]

    #GetRealTimeFeature(self,featureSet,AxisData,Naxis)
    def GetRealTimeFeature(self,Element,Feature,data):
        ret = self.dataBlockSet[Element].GetRealTimeFeature(Feature,data,1)
        return ret
    def SetRawXYZData(self,AccData,GyroData,MagData,AngleData):
        self.dataBlockSet["Acc"].rawXYZ = np.concatenate((self.dataBlockSet["Acc"].rawXYZ,AccData) , axis=0)
        self.dataBlockSet["Gyo"].rawXYZ = np.concatenate((self.dataBlockSet["Gyo"].rawXYZ,GyroData) , axis=0)
        self.dataBlockSet["Mag"].rawXYZ = np.concatenate((self.dataBlockSet["Mag"].rawXYZ, MagData ) , axis=0)
        self.dataBlockSet["Angle"].rawXYZ = np.concatenate((self.dataBlockSet["Angle"].rawXYZ, AngleData ) , axis=0)

    def SetFilteredXYZData(self,AccData,GyroData,MagData,AngleData):
        self.dataBlockSet["Acc"].filteredXYZ = np.concatenate((self.dataBlockSet["Acc"].filteredXYZ,AccData) , axis=0)
        self.dataBlockSet["Gyo"].filteredXYZ = np.concatenate((self.dataBlockSet["Gyo"].filteredXYZ,GyroData) , axis=0)
        self.dataBlockSet["Mag"].filteredXYZ = np.concatenate((self.dataBlockSet["Mag"].filteredXYZ, MagData ) , axis=0)
        self.dataBlockSet["Angle"].filteredXYZ = np.concatenate((self.dataBlockSet["Angle"].filteredXYZ, AngleData ) , axis=0)



if __name__ == '__main__':
    w = np.mat([[1, 8],[6,2],[3,5],[7,10],[4,8]])
    # print w[0:,0:1]
    print scipy.stats.pearsonr(w[0:,0:1],w[0:,1:2])
    # getRMS(w )
    # print w
    # A={'A':0.0,'B':2.0}
    # for key,value in A.items():
    #     print key,value
