from multiprocessing import Process, Manager,Array,Value, Lock
from sklearn import preprocessing
from ctypes import c_bool
import math
import DataStructure
import signal
import binascii
from sklearn import preprocessing
from sklearn.decomposition import PCA


import time
import struct
import threading
import pickle
import collections
import json,httplib
import select 
import signal
import sys
import  os
from pynput import keyboard
# import pyqtgraph as pg

import numpy as np
from bluepy import btle
import fastdtw
import DTW
#---for save data
#import self-defined module--------------------------------------------------------------



sys.path.append("OpenGL")
import myOpenGL
from myOpenGL import myCube
import QTRealTimeScatter
import QTwebcam
import QTRealLine
import Mahony
import scipy

#signal.signal(signal.SIGALRM, handler)

# matplotlib.use('TkAgg')
#global variables -----------------------------------------------------------------------
numOfDongle = 3
#ifae 0 dongle is responsible for scanning peripheral
#list of node connecting 

#conn_list = []
#myopenGL's object
saveData = []
acc_divider = 4095.999718
#acc_divider = 4096
gyro_divider = 65.500002
#gyro_divider = 65.5
DEG2RAD = 0.01745329251
ccount=1
#mahony_list = []



limit_num_of_received_data=50000    
# fp = open('dataset1.dat', "wb") #append
# fp1 = open('KNNClassifier.dat', "rb")


index = 0
KNNmodel=0
lastYaw = 1000
Trainning_minimum=[]
Trainning_diff=[]
#class -----------------------------------------------------------------------------------
lastYaw=0
data1=np.NAN
data2=np.NAN
data3=np.NAN

'''
[xxx]: xxx is a function
Calibration:
            Acc: calculate calibration value beforhand
            Gyro: calculate at phase 1(startup phasess) in the function: [GetBacicData]
            Mag: calculate calibration value beforhand

Process: 
            # [ScanProcess](Scan) -> startup phasess ( [GetBacicData](connect, get Gyro calibaration value and stopping threshold) )
            # -> create plot process([QTRun]) ->
            # Get IMU data by notification  -> mackdwick(get angle) 
            # -> remove gravity -> smoothing data
            # every 0.05 to execute the function [realtime] 
            **** 0.05 is the window size

Data:   
        Inst.dataBlockSet:
                        ['Acc'] : rawXYZ(save raw data), filteredXYZ(save filterd data)
                        ['Gyo'] : rawXYZ(save raw data), filteredXYZ(save filterd data)
                        ['Mag'] : rawXYZ(save raw data), filteredXYZ(save filterd data)
                        ['Angle'] : rawXYZ(save raw data), filteredXYZ(save filterd data)

        We keep the sensor and angle data whatever the user is in the static or moving state
        So use the other varible(Inst.Windows.workingIDX) to keep the index of data when moving 

        Inst.Windows.workingIDX : the index of data when moving 
'''

class MyDelegate(btle.DefaultDelegate):
    def __init__(self,node ):
        btle.DefaultDelegate.__init__(self)
        self.node = node

    def handleNotification(self, cHandle, data):
        b2a =  binascii.b2a_hex(data)
        self.node.noti = Uint4Toshort([b2a[0:4],b2a[4:8],b2a[8:12],b2a[12:16],b2a[16:20],b2a[20:24],b2a[24:28],b2a[28:32],b2a[32:36],b2a[36:40]])
        # ... perhaps check cHandle
        # ... process 'data'





class myNode(object):
    '''' a class to maintain connection and the calibration value of sensors'''
    def __init__(self):
        self.Peripheral = None
        self.nodeCube = None
        self.drawWindowNumber = -1
        self.accBias = [0.0,0.0,0.0]
        self.gyroBias = [0.0,0.0,0.0]
        self.magBias = [0.0,0.0,0.0]
        self.magScale = [0.0,0.0,0.0]
        self.magCalibration = [0.0,0.0,0.0]
        self.noti = None
        self.fail_notify=0
        self.workingtime=0.0
        self.datagram=[]
        self.seq=0
        self.count_received_data=0
S = np.array([[  2.42754810e-04,   3.41614666e-07,  -2.07507663e-07],
[  3.41614666e-07,   2.43926399e-04,   1.68822071e-07],
[ -2.07507663e-07,   1.68822071e-07,   2.43800712e-04]])
B = [-28.43905915,  51.22161875, -72.33527491]

global S,B   

def BLEconnection(connNode,addr,connType,iface):
    ''' do ble connection '''

    connNode.Peripheral = btle.Peripheral(addr , connType , iface = iface)
    connNode.Peripheral.setDelegate(MyDelegate(connNode))

    magCalibration = binascii.b2a_hex(connNode.Peripheral.readCharacteristic(0x4C))
    calibrationData = [magCalibration[0:8], magCalibration[8:16], magCalibration[16:24]]
    connNode.magCalibration = Uint8Tofloat(calibrationData)
    connNode.accBias = [-0.039746094, -0.012792969, -0.056347656]
    connNode.gyroBias = [1.477862573, 0.088549618, -1.477862597]
    # connNode.magBias = [57.712502, 27.521484, -37.898438 ]
    # connNode.magScale = [0.990893, 1.042146, 0.969697]
    connNode.magBias =  [52.190625, 26.627929687499996, -24.46171875]
    connNode.magScale =   [1.0418410041841004, 0.9688715953307393, 0.9920318725099602]

    #home
    # connNode.magBias = [48.312499, 41.460943, -21.877735 ]
    # connNode.magScale = [1.005747, 1.009227, 0.985360]
    print("accScales: ",S)
    print("accBias: ",B)
    # print("gyroBias: ",connNode.gyroBias)
    print("magBias: ",connNode.magBias)
    print("magScale: ",connNode.magScale)
    print("magCalibration: ",connNode.magCalibration)
    print("connect successfully")


#Try to get Service , Characteristic and set notification
    try:
        #need to add 0000fed0-0000-1000-8000-00805f9b34fb
        service = connNode.Peripheral.getServiceByUUID("0000FED0-0000-1000-8000-00805f9b34fb")
        char = service.getCharacteristics("0000FED7-0000-1000-8000-00805f9b34fb")[0] 
        connNode.Peripheral.writeCharacteristic(char.handle + 2,struct.pack('<bb', 0x01, 0x00),True)
    except:
        print("get service, characteristic or set notification failed")


def ScanProcess(iface=0):
    '''Scan '''
    scanner = btle.Scanner(iface)
    while True :
        print("Still scanning... count: %s"  % 1)
        try:
            devcies = scanner.scan(timeout = 3)   
            # print    devcies     
            for dev in devcies:
                # print "xx"
                if dev.addr == "3c:cd:40:18:c1:98":  #3c:cd:40:18:c3:46   3c:cd:40:0b:c0:48 #3c:cd:40:0b:c1:11 #3c:cd:40:18:c1:98
                    print("devcies %s (%s) , RSSI = %d dB" %(dev.addr , dev.addrType , dev.rssi))
                    return
                        #Try to create connection          
        except:
            print "failed scan" 
            exit()      



#[['1', 5, 15, matrix([[-10.35904978]]), 'GoStraight', 12.979903258208235]]

def MergeOverLap(VaildList,newDdata,Inst): 
    ''' test if the newDdata overlap the element of VaildList
        
        if overlap too many windows -> the gesture of the element of VaildList and the gesture of newDdata have strong relation     
        
        Args:
            VaildList : a list to saved all possible gestures information : [['0', 0, 12, matrix([[-10.35904978]]), 'Left', 12.979903258208235],['0', 13, 25, matrix([[16.2256]]), 'Right', 12.979903258208235]]
            newDdata : new gesture information, like that ['1', 5, 15, matrix([[-10.35904978]]), 'GoStraight', 12.979903258208235]
            Inst : an instance to save the information of IMU
    '''

    startIdx = Inst.Windows.workingIDX[newDdata[1]][0]
    endIdx = Inst.Windows.workingIDX [newDdata[2]][1]
    
    if newDdata[4] == 'GoStraight':
        LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[endIdx,0]) - 45)
        GoAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[endIdx,0]) - 0)
        if LeftRightAngle < GoAngle:
            if Inst.dataBlockSet['Angle'].filteredXYZ[endIdx,0] < 0:
                newDdata[4] = 'RightGoStraight'
            else:
                newDdata[4] = 'LeftGoStraight'
        else:
            newDdata[4] = 'GoStraight'   
        # print  startIdx,endIdx ,newDdata[1] ,newDdata[2]
    elif newDdata[4] == 'BackStraight':
        LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx,0]) - 45)
        BackAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx,0]) - 0)
        if LeftRightAngle < BackAngle:
            if Inst.dataBlockSet['Angle'].filteredXYZ[startIdx,0] < 0:
                newDdata[4] = 'RightBackStraight'
            else:
                newDdata[4] = 'LeftBackStraight'
        else:
            newDdata[4] = 'BackStraight'
    # VaildList.append(newDdata)
    # return 
    elif newDdata[4] == 'LeftGoStraight' or newDdata[4] =='RightGoStraight':
        LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[endIdx,0]) - 45)
        GoAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[endIdx,0]) - 0) 
        if LeftRightAngle > GoAngle: 
            newDdata[4] = 'GoStraight'
    elif newDdata[4] == 'LeftBackStraight' or newDdata[4] == 'RightBackStraight':
        LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx,0]) - 45)
        BackAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx,0]) - 0) 
        if LeftRightAngle > BackAngle: 
            newDdata[4] = 'BackStraight'

    #first gesture pass the threshold, but may be replace by next candidate
    # print "\033[1;31m",newDdata,"\033[0;0m"
    if len(VaildList) == 0:    
        VaildList.append(newDdata)

        return 
    for i in range(len(VaildList)-1,-1,-1):
        item = VaildList[i]

        if newDdata[1] >= item[2] and i == len(VaildList) -1:
            VaildList.append(newDdata)

            break
        
        if item[2] >=  newDdata[2]:
            Next = item
            Prev = newDdata

        else:
            Next = newDdata
            Prev = item


        Datalength = Prev[2] - Prev[1]
        thres = float(Datalength/2)

        #replace it
        
        if Prev[ 2 ] - Next[ 1 ] >= thres  :
            
            if newDdata[5] < item[5]:
                # print "(MergeOverLap)\033[1;31m",VaildList[i][4],"is replaced by",newDdata[4],"\033[0;0m"
                VaildList[i] = newDdata

            break 
        elif i == 0:
            # print "(MergeOverLap)\033[1;31m","What happended???","\033[0;0m"
            VaildList.append(newDdata)
            break    

threshold = [15,5,10]
def absDist(A,B):
    return np.sum(np.abs(A-B))
def AxisChange(VarInst,VarAngle,VarGyo,VarAcc,idx,Inst):
    ''' A function to find the time point of the difference from minus to positive or positive to minus
        The difference is the V_{current} - V_{last step}
        Args:
            VarInst : an instance to keep the lasting time of  value with same direction 
            VarAngle : The difference of angle 
            VarGyo & VarAcc : The difference of VarGyo and VarAcc
    '''

    global threshold,DTWModel,AnglePose
    for i in range(0,VarAngle.shape[1]):
        axisPos = str(i)
        AngleState = None
        if VarAngle[0,i] < 0:
            AngleState = 'N'
            VarInst.DataVar['Angle'].VarList[i].append(AngleState)
        else:
            AngleState = 'P'
            VarInst.DataVar['Angle'].VarList[i].append(AngleState)


        if VarAcc[0,i] < 0:
            VarInst.DataVar['Acc'].VarList[i].append('N')
        else:
            VarInst.DataVar['Acc'].VarList[i].append('P')


        if VarInst.DataVar['Angle'].currentState[axisPos] == None:
            # VarInst.DataVar['Angle'].currentState[axisPos] = AngleState
            VarInst.DataVar['Angle'].transitionDir(axisPos,AngleState,Inst.Windows.workingIDX,Inst.dataBlockSet['Angle'])
            continue

        #transition - direction changing
        if AngleState != VarInst.DataVar['Angle'].currentState[axisPos]:
            # print VarInst.DataVar['Angle'].getLastState(i)

            #  create a recode which contain the index of windows in the same state          
            VarInst.DataVar['Angle'].transitionDir(axisPos,AngleState,Inst.Windows.workingIDX,Inst.dataBlockSet['Angle'])


            if AngleState == 'N':
                PreviousState = 'P'
            else:
                PreviousState = 'N'
            
            PreviousStateList = VarInst.DataVar['Angle'].dataIndex[axisPos][PreviousState]
            startIdx = Inst.Windows.workingIDX[ PreviousStateList[-1][0] ]
            endIdx = Inst.Windows.workingIDX[ PreviousStateList[-1][1] ]
            firstData = np.mean( Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0]:startIdx[1],i],axis=0)
            endData = np.mean( Inst.dataBlockSet['Angle'].filteredXYZ[endIdx[0]:endIdx[1],i] ,axis=0)
            PreviousStateList[-1].append(endData - firstData)
            # print endData - firstData
            AngleDiff = Inst.dataBlockSet['Angle'].filteredXYZ[endIdx[1],:] - Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],:]
            if np.abs(endData - firstData) > threshold[i]:
                #VarInst.DataVar['Angle'].ValidWindows.append([axisPos,PreviousStateList[-1][0],PreviousStateList[-1][1],endData - firstData])
                candidate = [axisPos,PreviousStateList[-1][0],PreviousStateList[-1][1],endData - firstData]
                
                loweDist = 2000
                gesture = None
                if PreviousStateList[-1][1] - PreviousStateList[-1][0] < 5:
                    continue

                A = Inst.dataBlockSet['Acc'].filteredXYZ[startIdx[0]:endIdx[1],:]
                A = preprocessing.normalize(A , norm='l2')
                acc = np.zeros((0,3))
                for idx in  Inst.Windows.workingIDX[PreviousStateList[-1][0]:PreviousStateList[-1][1]+1]:
                    Sidx = idx[0] - startIdx[0]
                    Eidx = idx[1] - startIdx[0]
                    mean = np.mean(A[Sidx:Eidx,:],axis=0)
                    acc = np.concatenate ( (acc,[mean]),axis=0 )


                for key in DTWModel[axisPos][PreviousState].keys() :
                    # distance, path = fastdtw.fastdtw(A, DTWModel[axisPos][PreviousState][key], dist= absDist)
                    distance = DTW.distance(acc, DTWModel[axisPos][PreviousState][key])
                    # print "(AxisChange)distance:","\033[1;34m",distance,key,"\033[0;0m",np.abs(AnglePose[key][0][1]),np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[endIdx[1],i]),i
                    if loweDist > distance:
                        #,DTWModel[axisPos][PreviousState].keys()
                        loweDist = distance
                        gesture = key
                candidate.append(gesture)
                candidate.append(loweDist)
                candidate.append(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],:])
                candidate.append(Inst.dataBlockSet['Angle'].filteredXYZ[endIdx[1],:])
                VarInst.DataVar['Angle'].AllGestureWindows.append(candidate)
                #test if candidate gesture is the gesture or it is noise
                MergeOverLap(VarInst.DataVar['Angle'].ValidWindows,candidate,Inst)

            VarInst.DataVar['Angle'].getLastInfo(i).append( [VarInst.DataVar['Angle'].getLastState(i)] )
            VarInst.DataVar['Angle'].getLastInfo(i)[-1].append(VarAngle)  
     
        else:
            
            if VarInst.DataVar['Angle'].currentState[axisPos] != None:
                VarInst.DataVar['Angle'].dataIndex[axisPos][AngleState][-1][1] = len(VarInst.DataVar['Angle'].VarList[i]) 
                VarInst.DataVar['Angle'].dataIndex[axisPos]['lastIndex'] = len(VarInst.DataVar['Angle'].VarList[i]) + 1 

inverse = np.mat([[0,0,0]])
tempAngle =  np.mat([[0,0,0]])
tempAcc =  np.mat([[0,0,0]])
tempGyo =  np.mat([[0,0,0]])

def CrossingData(Inst,dataPack,Axis,windowsIdx,lastOne=False):
    ''' Get crossing zero point 
        
    '''    
    positive = dataPack > 0     
    ret = np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
    # print type(ret)
    if Axis == 'X':
        axis = 0
    elif Axis == 'Y':
        axis = 1
    else:
        axis = 2
    if lastOne == True:
        ret = np.array ([Inst.Windows.windowsIdx[-1][1]-Inst.Windows.windowsIdx[-1][0]])
    if len(ret) != 0:

        if positive[0] == True: 
            keys = ['P2N','N2P']
            if len(Inst.Windows.windowsCrossDataIndex[Axis]['N2P']) != 0:
                lastIndex = Inst.Windows.windowsCrossDataIndex[Axis]['N2P'][-1][1]
                lastWindowsIdx = Inst.Windows.windowsCrossDataIndex[Axis]['N2P'][-1][3]
            else:
                # first crossing data
                lastIndex = Inst.Windows.workingIDX[0][0]  
                lastWindowsIdx = 0                
        else:
            keys = ['N2P','P2N']
            if len(Inst.Windows.windowsCrossDataIndex[Axis]['P2N']) != 0:
                lastIndex = Inst.Windows.windowsCrossDataIndex[Axis]['P2N'][-1][1]
                lastWindowsIdx = Inst.Windows.windowsCrossDataIndex[Axis]['P2N'][-1][3]
            else:
                # first crossing data
                lastIndex =  Inst.Windows.workingIDX[0][0] 
                lastWindowsIdx = 0

        # if Axis == 'X': 
        #     print lastIndex,Inst.Windows.workingIDX[0][0] , Axis

        #the newest data index
        offset = Inst.Windows.workingIDX[-1][0]
        for index,i in zip(ret,range(0,ret.shape[0])):
            if index == 0:
                continue
            if i >0:
               windowsIdx = lastWindowsIdx 
            Inst.Windows.windowsCrossDataIndex[Axis][ keys[i%2] ].append([lastIndex,offset + index,lastWindowsIdx,windowsIdx]) 
            # Inst.Windows.windowsCrossDataIndex[Axis][ keys[i%2] ][-1].append(np.var(Inst.dataBlockSet['Acc'].filteredXYZ[lastIndex:offset + index,axis]))
            Inst.Windows.windowsCrossDataIndex[Axis][ keys[i%2] ][-1].append(DataStructure.getRMS(Inst.dataBlockSet['Acc'].filteredXYZ[lastIndex:offset + index,axis]))
            if DataStructure.getRMS(Inst.dataBlockSet['Acc'].filteredXYZ[lastIndex:offset + index,axis]) > 0.2:
                VarInst.DataVar['Acc'].VarGestureWindows.append(Inst.Windows.windowsCrossDataIndex[Axis][ keys[i%2] ][-1]+[Axis,keys[i%2]])
            lastIndex =  offset + index 
            # print lastIndex,keys[i%2]

def  setPlotData (Inst,plotList,isStatic,endIdx,dataLengthList,Timestamp):
    '''  set data to the shared memory, then the other process(QTRun) can plot the data  

        Args:
            Inst : an instance to save IMU information
            isStatic : if isStatic.value == true then plot graph (shared memory) 
            endIdx : the number of data to plot (shared memory) 
            Timestamp : not used
    '''

    # the number of windows we ploted, and now we plot the new data 
    for i in range(Inst.Windows.plotNum,len(Inst.Windows.workingIDX)):
        dataLengthList[i*2] = Inst.Windows.workingIDX[i][0]
        dataLengthList[i*2+1] = Inst.Windows.workingIDX[i][1]

    if Inst.Windows.plotNum  == 0:
        start = Inst.Windows.workingIDX[Inst.Windows.plotNum][0]
    else:
        start = Inst.Windows.workingIDX[Inst.Windows.plotNum][1]
    end = Inst.Windows.workingIDX[len(Inst.Windows.workingIDX) - 1][1]
    # print start,end,Inst.Windows.plotNum,Inst.Windows.workingIDX
    Inst.Windows.plotNum = len(Inst.Windows.workingIDX) - 1
    length = end - start
    

    filterdData = Inst.dataBlockSet['Acc'].filteredXYZ[ start:end, ]
    rawData = Inst.dataBlockSet['Angle'].rawXYZ[ start:end, ]
    # filterdGyoData = Inst.dataBlockSet['Gyo'].filteredXYZ[ start:end, ]

    filterdData = filterdData.transpose()
    rawData = rawData.transpose()
    # filterdGyoData = filterdGyoData.transpose()
    
    try:
        plotList[0][0:length] = filterdData[0,:].tolist()[0]
        plotList[1][0:length] = filterdData[1,:].tolist()[0]
        plotList[2][0:length] = filterdData[2,:].tolist()[0]
        plotList[3][0:length] = rawData[0,:].tolist()[0]
        plotList[4][0:length] = rawData[1,:].tolist()[0]
        plotList[5][0:length] = rawData[2,:].tolist()[0]

        endIdx.value = length
        isStatic.value = False    
    except:
        pass
        # print length,filterdData.shape

def  setPlotDataWithCam (Inst,plotList,isStatic,endIdx,dataLengthList,Timestamp,start):
    # we plot it once a window
    
    end = Inst.Windows.workingIDX[len(Inst.Windows.workingIDX) - 1][1]
    filterdData = Inst.dataBlockSet['Acc'].filteredXYZ[ start:end, ]
    rawData = Inst.dataBlockSet['Angle'].rawXYZ[ start:end, ]
    # filterdGyoData = Inst.dataBlockSet['Gyo'].filteredXYZ[ start:end, ]

    filterdData = filterdData.transpose()
    rawData = rawData.transpose()
    length = end - start
    

    try:
        plotList[0][0:length] = filterdData[0,:].tolist()[0]
        plotList[1][0:length] = filterdData[1,:].tolist()[0]
        plotList[2][0:length] = filterdData[2,:].tolist()[0]
        plotList[3][0:length] = rawData[0,:].tolist()[0]
        plotList[4][0:length] = rawData[1,:].tolist()[0]
        plotList[5][0:length] = rawData[2,:].tolist()[0]
        Timestamp[0:length] = Inst.timestamp[ start:end]


        endIdx.value = length
        isStatic.value = False  
        Inst.Windows.startPlotPoint =  end 
    except:
        pass


def MovingProcess(VarInst,Inst,IdxBound,AccMean,plotList,isStatic,endIdx,resetFlag,dataLengthList,Timestamp,withCamera=False,staticChannel=False):
    ''' 
        test if the user is static or moving using the mean of the acc in the windows
        window size we set is 0.05s, and overlap time is 0.015
        If user is in moving state,we call MovingProcess

        Args:
            VarInst : an Inst to save the different of the mean data between two windows
            Inst : an instance to save IMU information
            IdxBound : windows start point and end point
            AccMean : the mean of acc in the windows
            plotList : the data list to plot 
            isStatic : For plot, if we want to plot, we should set it to false
            endIdx : how many point we update to the plot list
            resetFlag : the flag to replot the graph
            saverState : if saverState.value == 3 , we save data
            dataLengthList : the number of point to plot of each windows
            Timestamp : for timestamp
            withCamera : does we use camera
    '''
    global inverse,stateData
    global XAction,YAction,ZAction,XModel,YModel,ZModel,tempAngle,tempAcc,tempGyo,varList,varAccList

    #when sensor state:STATIC - > MOVING, we should do reset at first time
    if staticChannel == False:        
        if len(Inst.Windows.RawAccStata) == 0:
            inverse = np.mat([[0,0,0]])
            stateData = [[-1,-1],[-1,-1],[-1,-1]]
            tempAcc =  np.mat([[0,0,0]])
            tempGyo =  np.mat([[0,0,0]])
            tempAngle =  np.mat([[0,0,0]])
            resetFlag.value = True
            Inst.Windows.ResetZeroCrossData(IdxBound[0])
            Inst.Windows.plotNum = 0
            Inst.Windows.startPlotPoint = IdxBound[0]
    Inst.Windows.workingIDX.append(IdxBound)
    
  
    # print Inst.Windows.workingIDX
    filterdData = Inst.dataBlockSet['Acc'].filteredXYZ[ IdxBound[0]:IdxBound[1], ]
    filterdGyoData = Inst.dataBlockSet['Gyo'].filteredXYZ[ IdxBound[0]:IdxBound[1], ]
    filterdAngleData = Inst.dataBlockSet['Angle'].rawXYZ[ IdxBound[0]:IdxBound[1], ]

    # print DataStructure.getRMS( filterdData ),np.mean(filterdAngleData,axis=0)

    AccData =  preprocessing.normalize( filterdData, norm='l2')
    AccData = np.mean(AccData,axis=0) 
    GyoData =  preprocessing.normalize( filterdGyoData, norm='l2')
    GyoData = np.mean(GyoData,axis=0)

    Angle  = np.mean( filterdAngleData,axis=0 )
    Acc = np.mean( filterdData,axis=0 )
    Gyo = np.mean( filterdGyoData,axis=0 )

    # 
    varAngle = Angle - tempAngle
    varAcc = Acc - tempAcc
    if len(Inst.Windows.workingIDX) != 1:         
        AxisChange(VarInst,varAngle, np.mat([[0,0,0]]) , varAcc,len(Inst.Windows.workingIDX),Inst)
        VarInst.DataVar['Angle'].value[0].append(varAngle[0,0])
        VarInst.DataVar['Angle'].value[1].append(varAngle[0,1])
        VarInst.DataVar['Angle'].value[2].append(varAngle[0,2])    
    tempAngle = Angle 
    tempAcc = Acc
    tempGyo = Gyo

    Inst.static = False

    # filterdAngleData = filterdAngleData.transpose()
    # filterdData = filterdData.transpose()
    
    if withCamera == False :
        if len(Inst.Windows.workingIDX) % 3  == 0:
            # we plot data every 3 windows
            setPlotData(Inst,plotList,isStatic,endIdx,dataLengthList,Timestamp)
    else:
        setPlotDataWithCam(Inst,plotList,isStatic,endIdx,dataLengthList,Timestamp,Inst.Windows.startPlotPoint)

    if staticChannel == False:
        Inst.staticCount = 0

    direction = np.argsort( np.abs( AccData ))
    GyoDirection = np.argsort( np.abs( GyoData ))
    


    CrossingData(Inst,filterdData[:,0:1],'X',len(Inst.Windows.workingIDX)-1)
    CrossingData(Inst,filterdData[:,1:2],'Y',len(Inst.Windows.workingIDX)-1)
    CrossingData(Inst,filterdData[:,2:3],'Z',len(Inst.Windows.workingIDX)-1)

    if len(Inst.Windows.RawAccStata) != 0:
        pass

    Inst.Windows.SetcurrSingleState(direction[2],GyoDirection[2])
    Inst.Windows.RawAccStata.append(direction[2])
    Inst.Windows.RawGyoStata.append(GyoDirection[2])

def DTWJudge(startWDS,endWDS,Inst,angleDiff,i):
    global velocity,displacement,fileN,threshold
    if np.abs(angleDiff) > threshold[i]:
        loweDist = 2000
        gesture = None
        startIdx = Inst.Windows.workingIDX[startWDS]
        end = Inst.Windows.workingIDX[endWDS]
        acc = np.zeros((0,3)) 
        try:
            A = Inst.dataBlockSet['Acc'].filteredXYZ[startIdx[0]:end[1],:]
            A = preprocessing.normalize(A , norm='l2')
        except:
            return None
        for idx in  Inst.Windows.workingIDX[ startWDS:endWDS+1]:
            Sidx = idx[0] - startIdx[0]
            Eidx = idx[1] - startIdx[0]
            mean = np.mean(A[Sidx:Eidx,:],axis=0)            
            acc = np.concatenate ( (acc,[mean]),axis=0 )                        
        acc = acc.tolist()
        if  angleDiff[0,0] < 0:
            Dir = 'N'
        else:
            Dir = 'P'

        for key in DTWModel[str(i)][Dir].keys():                             
            distance = DTW.distance(acc, DTWModel[str(i)][Dir][key])
            #print "(realtime) distance:","\033[1;36m",distance,key,"\033[0;0m","Angle:",np.abs(AnglePose[key][0][1]),np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[end[1],i]),i
            if loweDist > distance:                                
                loweDist = distance
                gesture = key
        if gesture == 'GoStraight':
            LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[end[1],0]) - 45)
            GoAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[end[1],0]) - 0)
            if LeftRightAngle < GoAngle:
                if Inst.dataBlockSet['Angle'].filteredXYZ[end[1],0] < 0:
                    gesture = 'RightGoStraight'
                else:
                    gesture = 'LeftGoStraight'
            else:
                gesture = 'GoStraight'   
            # print  startIdx,endIdx ,newDdata[1] ,newDdata[2]
        elif gesture == 'BackStraight':
            LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],0]) - 45)
            BackAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],0]) - 0)
            if LeftRightAngle < BackAngle:
                if Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],0] < 0:
                    gesture = 'RightBackStraight'
                else:
                    gesture = 'LeftBackStraight'
            else:
                gesture = 'BackStraight'


        elif gesture == 'LeftGoStraight' or gesture =='RightGoStraight':
            LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[end[1],0]) - 45)
            GoAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[end[1],0]) - 0) 
            if LeftRightAngle > GoAngle: 
                gesture = 'GoStraight'
        elif gesture == 'LeftBackStraight' or gesture == 'RightBackStraight':
            LeftRightAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],0]) - 45)
            BackAngle = np.abs(np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],0]) - 0) 
            if LeftRightAngle > BackAngle: 
                gesture = 'BackStraight'
        return [i,startWDS,endWDS,angleDiff,gesture,loweDist]
    else:
        return None

def ExceptionProcess(VaildGesture,MaybeGesture,Inst):
    Axis = {}
    Axis['0'] = 'Yaw'
    Axis['1'] = 'Pitch'
    Axis['2'] = 'Roll'
    GestureMapping = {}
    GestureMapping['GoStraight'] = 'Forward'
    GestureMapping['BackStraight'] = 'Backward'
    GestureMapping['Left'] = 'Left'
    GestureMapping['Right'] = 'Right'
    GestureMapping['UpStraight'] = 'Up'
    GestureMapping['DownStraight'] = 'Down'
    GestureMapping['RightGoStraight'] = 'Rightforward'
    GestureMapping['LeftGoStraight'] = 'Leftforward'
    GestureMapping['RightBackStraight'] = 'RightBackward'
    GestureMapping['LeftBackStraight'] = 'LeftBackward'

    correlated = []
    # print MaybeGesture
    # print VaildGesture
    for mayGesture in MaybeGesture:
      correlated.append([mayGesture])
      for preGesture in VaildGesture:
          if np.abs( mayGesture[1] - preGesture[1]) <= 5 or np.abs( mayGesture[2] - preGesture[2]) <= 5:
              correlated[-1].append(preGesture)
    # for data in  correlated:
    #     print "\033[1;36m",data,"\033[0;0m"
    for Alist in correlated:
        if len(Alist) == 1:
            continue
        mayGesture = Alist[0]
        minimumWin = [2000,-2]
        maximumWin = [0,-2]
        replaced = False
        for i in range(1,len(Alist)):
            preGesture =  Alist[i]
            startWDS = preGesture[1]
            endWDS = preGesture[2]
            # we test it if the gesture 
            I = np.mean(Inst.dataBlockSet['Angle'].filteredXYZ[Inst.Windows.workingIDX[startWDS][0]:Inst.Windows.workingIDX[startWDS][1],int(mayGesture[0])],axis=0)
            J = np.mean(Inst.dataBlockSet['Angle'].filteredXYZ[Inst.Windows.workingIDX[endWDS][0]:Inst.Windows.workingIDX[endWDS][1],int(mayGesture[0])],axis=0)                                 
            result = DTWJudge(startWDS,endWDS,Inst,(J-I),int(mayGesture[0]))
            if preGesture[1] < minimumWin[0] :
              minimumWin[0] = preGesture[1] 
              minimumWin[1] = i
            if preGesture[2] > maximumWin[0]:
              maximumWin[0] = preGesture[2]
              maximumWin[1] = i

            if result != None and result[5] < preGesture[5] :
                replaced = True
                for i in range(0,len(VaildGesture)):
                    dis = VaildGesture[i][5]
                    Swin = VaildGesture[i][1]
                    Ewin = VaildGesture[i][2]

                    if dis == preGesture[5] and Swin == preGesture[1] and Ewin ==  preGesture[2]:
                        # print "Gesture:",VaildGesture[i],dis
                        VaildGesture[i] = result  
                        break
        
        # if replaced == False:
        mayGesture =  Alist[0]
        preGesture = Alist[ maximumWin[1] ]
        # the situation is  ['Left',30.282790160196825, 0, 28,'Yaw'] ,['Up', 5.273493876939512, 0, 18,'Pitch']  28 - 18 >3
        if np.abs( mayGesture[2] - preGesture[2]) > 3 :
            startWDS = preGesture[2]
            endWDS = mayGesture[2]
            I = np.mean(Inst.dataBlockSet['Angle'].filteredXYZ[Inst.Windows.workingIDX[startWDS][0]:Inst.Windows.workingIDX[startWDS][1],int(mayGesture[0])],axis=0)
            J = np.mean(Inst.dataBlockSet['Angle'].filteredXYZ[Inst.Windows.workingIDX[endWDS][0]:Inst.Windows.workingIDX[endWDS][1],int(mayGesture[0])],axis=0)                                 
            result = DTWJudge(startWDS,endWDS,Inst,(J-I),int(mayGesture[0]))                                            
            VaildGesture.append(result)
        # the situation is  ['Left',30.282790160196825, 0, 28,'Yaw'] ,['Up', 5.273493876939512, 18, 26,'Pitch']  18 -0 >3
        mayGesture =  Alist[0]
        preGesture = Alist[ minimumWin[1] ]
        if np.abs( mayGesture[1] - preGesture[1]) > 3 :
            startWDS = mayGesture[1]
            endWDS = preGesture[1]
            I = np.mean(Inst.dataBlockSet['Angle'].filteredXYZ[Inst.Windows.workingIDX[startWDS][0]:Inst.Windows.workingIDX[startWDS][1],int(mayGesture[0])],axis=0)
            J = np.mean(Inst.dataBlockSet['Angle'].filteredXYZ[Inst.Windows.workingIDX[endWDS][0]:Inst.Windows.workingIDX[endWDS][1],int(mayGesture[0])],axis=0)                                 
            result = DTWJudge(startWDS,endWDS,Inst,(J-I),int(mayGesture[0]))                                               
            VaildGesture.append(result)

    WDX = Inst.Windows.workingIDX[-1]
    FinalRollAngle = Inst.dataBlockSet['Angle'].filteredXYZ[WDX[1]-1,2]

    if np.abs(FinalRollAngle) > 40:
        Inst.RecognizeFlag = 0
        return 1 

    if Inst.RecognizeFlag == 1:
        orderList = []
        for valid in VaildGesture:
            if valid != None:
                orderList.append(valid[2])
        orderList = np.argsort(orderList) 
        gestureSet = []
        WDS = [] 
        for order in  orderList:
            gesture = VaildGesture[order]
            gestureSet.append(GestureMapping[gesture[4]])
            WDS.append([gesture[1],gesture[2]])
        gesStr =""
        for gest in gestureSet:
            gesStr += "\033[1;34m"
            gesStr += gest
            gesStr += "\033[0;0m"
            gesStr += "   ->    "
        print gesStr[:-9]
        wdsStr =""
        for Idx in WDS:
            wdsStr += "\033[1;31m"  
            wdsStr += str(Idx[0])
            wdsStr += "   "
            wdsStr += str(Idx[1])
            wdsStr += "\033[0;0m"
            wdsStr += "   ->    "
        print wdsStr[:-9]
        print"-----------------------------------"
        return 0   
    if Inst.RecognizeFlag == 0:
        if np.abs(FinalRollAngle) < 15:
            Inst.RecognizeFlag=1
            # Inst.Windows.RestartWorking()
            # Inst.Windows.ResetState() 
            # VarInst.reset()
            return 1            

fileN = 36
def realtime(VarInst,Inst,boundary,plotList,isStatic,endIdx,resetFlag,saverState,dataLengthList,Timestamp,withCamera=False):
    ''' 
        test if the user is static or moving using the mean of the acc in the windows
        window size we set is 0.05s, and overlap time is 0.015
        If user is in moving state,we call MovingProcess

        Args:
            VarInst : an Inst to save the different of the mean data between two windows
            Inst : an instance to save IMU information
            boundary : stopping threshold
            isStatic : For plot, if we want to plot, we should set it to false
            endIdx : how many point we update to the plot list
            resetFlag : the flag to replot the graph
            saverState : if saverState.value == 3 , we save data
            dataLengthList : the number of point to plot of each windows
            Timestamp : for timestamp
            withCamera : does we use camera
    '''
    global velocity,displacement,fileN,threshold
    IdxBound = Inst.Windows.windowsIdx[-1]

    AccMean = np.mean ( Inst.dataBlockSet['Acc'].rawXYZ[ IdxBound[0]:IdxBound[1], ],axis=0 )
    AccMagnitude = math.sqrt(AccMean[0,0]*AccMean[0,0]+AccMean[0,1]*AccMean[0,1]+AccMean[0,2]*AccMean[0,2])
    Gyomean = np.mean ( Inst.dataBlockSet['Gyo'].rawXYZ[ IdxBound[0]:IdxBound[1], ],axis=0 )
    GyoMagnitude = math.sqrt(Gyomean[0,0]*Gyomean[0,0]+Gyomean[0,1]*Gyomean[0,1]+Gyomean[0,2]*Gyomean[0,2])

    # test if the AccMagnitude pass the stopping threshold
    if (AccMagnitude < boundary[0] ) : #or GyoMagnitude < boundary[1]
        
        # if the AccMagnitude do not pass the stopping threshold twice continuously
        #, we think it enter to the static mode
        if Inst.staticCount > 0:

            if len(Inst.Windows.RawAccStata) !=0:
                # print  boundary[0],AccMagnitude
                if len(VarInst.DataVar['Angle'].VarList[0]) < 4:
                    return
                
                for i in ['0','1','2']:

                    StateList = VarInst.DataVar['Angle'].dataIndex[i][ VarInst.DataVar['Angle'].currentState[i] ]

                    startIdx = Inst.Windows.workingIDX[ StateList[-1][0] ]
                    end = Inst.Windows.workingIDX[ StateList[-1][1] ]
                    firstData = np.mean( Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0]:startIdx[1],int(i):int(i)+1],axis=0)
                    endData = np.mean( Inst.dataBlockSet['Angle'].filteredXYZ[end[0]:end[1],int(i):int(i)+1] ,axis=0)
                    StateList[-1].append(endData - firstData)
                    if np.abs(endData - firstData) > threshold[int(i)]:

                        candidate = [i,StateList[-1][0],StateList[-1][1],endData - firstData]
                        # VarInst.DataVar['Angle'].AllGestureWindows.append(candidate+[Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],:],Inst.dataBlockSet['Angle'].filteredXYZ[end[1],:]])
                        loweDist = 2000
                        gesture = None
                        
                        acc = np.zeros((0,3)) 
                        A = Inst.dataBlockSet['Acc'].filteredXYZ[startIdx[0]:end[1],:]
                        A = preprocessing.normalize(A , norm='l2')

                        #Append the mean of acc of each windows to acc
                        for idx in  Inst.Windows.workingIDX[ StateList[-1][0]:StateList[-1][1]+1]:
                            Sidx = idx[0] - startIdx[0]
                            Eidx = idx[1] - startIdx[0]
                            mean = np.mean(A[Sidx:Eidx,:],axis=0)
                            
                            acc = np.concatenate ( (acc,[mean]),axis=0 )                        
                        acc = acc.tolist()

                        # AngleDiff = Inst.dataBlockSet['Angle'].filteredXYZ[end[1],:] - Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],:]
                        if StateList[-1][-1][0,0] < 0:
                            Dir = 'N'
                        else:
                            Dir = 'P'
                        
                        for key in DTWModel[i][Dir].keys():                             
                            distance = DTW.distance(acc, DTWModel[i][Dir][key])
                            #print "(realtime) distance:","\033[1;36m",distance,key,"\033[0;0m","Angle:",np.abs(AnglePose[key][0][1]),np.abs(Inst.dataBlockSet['Angle'].filteredXYZ[end[1],i]),i
                            if loweDist > distance:                                
                                loweDist = distance
                                gesture = key
                        candidate.append(gesture)
                        candidate.append(loweDist)
                        candidate.append(Inst.dataBlockSet['Angle'].filteredXYZ[startIdx[0],:])
                        candidate.append(Inst.dataBlockSet['Angle'].filteredXYZ[end[1],:])
                        VarInst.DataVar['Angle'].AllGestureWindows.append(candidate)


                        MergeOverLap(VarInst.DataVar['Angle'].ValidWindows,candidate,Inst)


                # finally we do ExceptionProcess
                ret = ExceptionProcess(VarInst.DataVar['Angle'].ValidWindows,VarInst.DataVar['Angle'].AllGestureWindows,Inst)
 
                # ret=0 normal situation
                if ret == 0:
                    startIdx = Inst.Windows.workingIDX[-1][0] 
                    end = Inst.Windows.workingIDX[-1][1] 
                    Acc = Inst.dataBlockSet['Acc'].filteredXYZ[startIdx:end,:]
                    CrossingData(Inst,Acc[:,0:1],'X',len(Inst.Windows.workingIDX)-1,True)
                    CrossingData(Inst,Acc[:,1:2],'Y',len(Inst.Windows.workingIDX)-1,True)
                    CrossingData(Inst,Acc[:,2:3],'Z',len(Inst.Windows.workingIDX)-1,True)
                    if withCamera == False :
                        setPlotData(Inst,plotList,isStatic,endIdx,dataLengthList,Timestamp)

                    else:
                        setPlotDataWithCam(Inst,plotList,isStatic,endIdx,dataLengthList,Timestamp,Inst.Windows.startPlotPoint)

                    lock = Lock()
                    if saverState.value == 3:
                        SaveProcess = threading.Thread(target = write_to_txt,args=( [Inst,VarInst,fileN,lock,saverState]))  #[data_chunk] make  data_chunk as a arguement
                        SaveProcess.start()
                        fileN = fileN + 1 
                        SaveProcess.join()   
                # print Inst.Windows.IdxforFilterd
                # print Inst.Windows.workingIDX
            Inst.Windows.RestartWorking()
            Inst.Windows.ResetState() 
            VarInst.reset()
            
            Inst.static = True 
            # print Inst 
        else:
            # print IdxBound 
            MovingProcess(VarInst,Inst,IdxBound,AccMean,plotList,isStatic,endIdx,resetFlag,dataLengthList,Timestamp,withCamera,staticChannel=True)

        Inst.staticCount += 1
    else:

        MovingProcess(VarInst,Inst,IdxBound,AccMean,plotList,isStatic,endIdx,resetFlag,dataLengthList,Timestamp,withCamera,staticChannel=False)



def write_to_txt(Inst,VarInst, fileN,lock, saverState): 

    if len(Inst.Windows.workingIDX) <= 4:
        return
    # filteredIdx = Inst.Windows.IdxforFilterd[0:-1]
    workingIdx = Inst.Windows.workingIDX
    # print Inst.Windows.windowsCrossDataIndex
    # print filteredIdx
    
    # startIdx = filteredIdx[0][0]
    # endIdx = filteredIdx[-1][1]
    workingStartIdx = workingIdx[0][0]
    workingEndIdx = workingIdx[-1][1]
    # print Inst.dataBlockSet['Angle'].shape,


    fp=open('./Fortest/Jay/Complex/JayDownLeftRightforward'+str(fileN)+'.dat', "wb")  

    lock.acquire()  #critical section -avoid other thread excute it while i'm excuting it. 
    item_dict = collections.OrderedDict()
    item_dict['Acc'] = Inst.dataBlockSet['Acc'].filteredXYZ[workingStartIdx:workingEndIdx,]
    item_dict['Gyo'] = Inst.dataBlockSet['Gyo'].filteredXYZ[workingStartIdx:workingEndIdx,]
    item_dict['Mag'] = Inst.dataBlockSet['Mag'].rawXYZ[workingStartIdx:workingEndIdx,]
    item_dict['Angle'] = Inst.dataBlockSet['Angle'].rawXYZ[workingStartIdx:workingEndIdx,]
    item_dict['timestamp'] = Inst.timestamp[workingStartIdx:workingEndIdx]
    item_dict['seq'] = Inst.sequence[workingStartIdx:workingEndIdx]
    item_dict['filteredIdx'] = workingIdx
    item_dict['workingIdx'] = workingIdx
    item_dict['VarDataIdx'] = VarInst.DataVar['Angle'].dataIndex
    item_dict['VarDataValue'] = VarInst.DataVar['Angle'].value
    item_dict['windowsCrossDataIndex'] = Inst.Windows.windowsCrossDataIndex
    item_dict['AccRawState'] = Inst.Windows.RawAccStata
    item_dict['GyoRawStata'] = Inst.Windows.RawGyoStata

    pickle.dump(item_dict,fp)
    fp.close()
    lock.release()  #down -release lock
    print("finish saving file: %d" % fileN)
    saverState.value = 1

def struct_isqrt(number):
        
    threehalfs = 1.5
    x2 = number * 0.5
    y = number
    packed_y = struct.pack('f', y) 

    i = struct.unpack('i', packed_y)[0]  # treat float's bytes as int 

    i = 0x5f3759df - (i >> 1)            # arithmetic with magic number
    packed_i = struct.pack('i', i)
    y = struct.unpack('f', packed_i)[0]  # treat int's bytes as float
    y = y * (threehalfs - (x2 * y * y))  # Newton's method
    return y

def QTRun(plotMyData,plot1,plot2,plot3,plot4,plot5,plot6,dataLengthList,Timestamp,Idx,resetFlag,isStatic): 

    data=[[],[],[],[],[],[]]
    windowsLen = []
    # print "xxxxxxxxxxxxxxxx"
    while True:
        # continue
        tEnd = time.time()
        while isStatic.value == True: 
            pass
        if  resetFlag.value == True:
            plotMyData.ResetGraph()
            resetFlag.value = False
        endIdx = Idx.value
        data[0]= plot1[0:endIdx]
        data[1]= plot2[0:endIdx]
        data[2]= plot3[0:endIdx]
        data[3]= plot4[0:endIdx]
        data[4]= plot5[0:endIdx]
        data[5]= plot6[0:endIdx]
        windowsLen.append([dataLengthList[0],dataLengthList[1]])
        windowsLen.append([dataLengthList[2],dataLengthList[3]])
        windowsLen.append([dataLengthList[4],dataLengthList[5]])
        isStatic.value = True
        # data[3].append(plot4.value)
        # data[4].append(plot5.value)
        # data[5].append(plot6.value)
        # data[6].append(timestamp.value)
            #,isCapturing.value
        
        plotMyData.setMyData(data,windowsLen)   #isCapturing.value              
        data=[[],[],[],[],[],[],[]]
        windowsLen = []
        # tStart = time.time() 
        changeMode = 1


def QTWebCam(plotMyData,plot1,plot2,plot3,plot4,plot5,plot6,Timestamp,isCapturing,isStatic,resetFlag): 

    data=[[],[],[],[],[],[],[]]
    windowsLen = []
    while True:
        tStart = time.time() 
        while isStatic.value == True: 
            pass
        if  resetFlag.value == True:
            plotMyData.ResetGraph()
            resetFlag.value = False
        endIdx = Idx.value
        data[0]= plot1[0:endIdx]
        data[1]= plot2[0:endIdx]
        data[2]= plot3[0:endIdx]
        data[3]= plot4[0:endIdx]
        data[4]= plot5[0:endIdx]
        data[5]= plot6[0:endIdx]
        data[6]= Timestamp[0:endIdx]
        


        plotMyData.setMyData(data,isCapturing.value)              
        data=[[],[],[],[],[],[],[],[]]
        windowsLen = []
        tStart = time.time() 
        isStatic.value = True




def GetBacicData(node,addr,connType,mahony,iface):
    '''Get the stopping threshold and the calibration of gyro
        Args:
            node : 
            addr :  sensor ble address
            connType : pubilc/ramdon
            iface : which dongle you use to construct the connection
    '''

    yawCalibration=0.0
   
    BLEconnection(node,addr,connType,iface=iface)
    count = 0
    gravity = 0
    staticLinearAcc = []
    staticLinearGyo = []
    print "Do not moving!!!"
    while count!= 300:
        if node.Peripheral.waitForNotifications(0.01):
            count = count + 1
           
            rawdata = node.noti
            removeGravity(rawdata) 
            node.gyroBias[0] += rawdata[3] 
            node.gyroBias[1] += rawdata[4]
            node.gyroBias[2] += rawdata[5]
    node.gyroBias[0] = node.gyroBias[0]/gyro_divider/300
    node.gyroBias[1] = node.gyroBias[1]/gyro_divider/300
    node.gyroBias[2] = node.gyroBias[2]/gyro_divider/300
    print node.gyroBias[0],node.gyroBias[1],node.gyroBias[2]
    count = 0        
    while count !=150:
        if node.Peripheral.waitForNotifications(0.01):
            count = count + 1
            
            rawdata = node.noti
            
            removeGravity(rawdata)
             
            # print rawdata[0],rawdata[1],rawdata[2]
            if count <20:         
                if rawdata[6] != 0 and rawdata[7] != 0 and rawdata[8] != 0:          
                    my = (rawdata[6]*0.15*node.magCalibration[0] - node.magBias[0])*node.magScale[0]*10
                    mx = (rawdata[7]*0.15*node.magCalibration[1] - node.magBias[1])*node.magScale[1]*10
                    mz = (rawdata[8]*0.15*node.magCalibration[2] - node.magBias[2])*node.magScale[2]*10
                    my = my * -1
                    mx = mx 

                recipNorm = struct_isqrt(mx * mx + my * my + mz * mz)
                my *= recipNorm
                mx *= recipNorm
                yaw = math.atan2(my,mx)* 57.2957795
                # yawCalibration += yaw
                acc = S.dot( np.array([rawdata[0],rawdata[1],rawdata[2]]) - B)
                gravity += math.sqrt(acc[0]*acc[0]+acc[1]*acc[1]+acc[2]*acc[2])

                if yaw < 0:
                    yaw = -180 - yaw 
                else:
                    yaw = 180 - yaw

                yawCalibration = yawCalibration + yaw 

                if count !=1:
                    gravity =   gravity/2
                    yawCalibration =  yawCalibration/2
                # count = 0  
                # print "end!"           
            else: 

                shortdata = doMahony(rawdata,node,mahony)
                angle = mahony.quatern2euler()
                angle[0] = 0
                angle[1] = 0
                angle[2] = 0
                # print shortdata[0],shortdata[1],shortdata[2]
                gravityLRF = [-1*gravity*math.sin(angle[1]*0.017453) ,gravity*math.cos(angle[1]*0.017453)*math.sin(angle[2]*0.017453),gravity*math.cos(angle[1]*0.017453)*math.cos(angle[2]*0.017453)]
                shortdata[0] = shortdata[0] - gravityLRF[0]
                shortdata[1] = shortdata[1] - gravityLRF[1]
                shortdata[2] = shortdata[2] - gravityLRF[2]                
                staticLinearAcc.append( math.sqrt(shortdata[0]*shortdata[0]+shortdata[1]*shortdata[1]+shortdata[2]*shortdata[2]) )
                staticLinearGyo.append( math.sqrt(shortdata[3]*shortdata[3]+shortdata[4]*shortdata[4]+shortdata[5]*shortdata[5]) )
                # print math.sqrt(shortdata[0]*shortdata[0]+shortdata[1]*shortdata[1]+shortdata[2]*shortdata[2])
    # print staticLinearAcc
    staticLinearAcc = np.max(staticLinearAcc)
    staticLinearGyo = np.max(staticLinearGyo)


    return gravity,yawCalibration,staticLinearAcc,staticLinearGyo

    # print yawCalibration


        

def Uint4Toshort(tenData):
    #print(threeData)
    retVal =[]
    
    for data in tenData:
    #(data)
        i = 0
        byteArray = []
        while(i != 4):
            byteArray.append(int(data[i:i+2], 16))
        #print(int(data, 16))
            i=i+2

        b = ''.join(chr(i) for i in byteArray)
        if data == tenData[9]:
            retVal.append(struct.unpack('<H',b)[0])        
        else:
            retVal.append(struct.unpack('<h',b)[0])
    # print retVal
    return retVal

def Uint8Tofloat(threeData):
    #print(threeData)
    retVal =[]
    
    for data in threeData:
    #(data)
        i = 0
        byteArray = []
        while(i != 8):
            byteArray.append(int(data[i:i+2], 16))
        #print(int(data, 16))
            i=i+2

        b = ''.join(chr(i) for i in byteArray)
        retVal.append(struct.unpack('<f',b)[0])
    return retVal


gravityList = [0.0,0.0,1.0]
def removeGravity(rawdata):
    ''' remove gravity by lowpass filter for magdwick. This is based on how android remove gravity'''


    global gravityList
    alpha = 0.8
    acc = S.dot( np.array([rawdata[0],rawdata[1],rawdata[2]]) - B)
    gravityList[0] = alpha * gravityList[0] + (1 - alpha) * acc[0]
    gravityList[1] = alpha * gravityList[1] + (1 - alpha) * acc[1]
    gravityList[2] = alpha * gravityList[2] + (1 - alpha) * acc[2]



   

global sumGyoZ ,count
sumGyoZ = 0
sumGyoX = 0
sumGyoY = 0
count = 0

def doMahony(rawdata, node,mahony):
    '''do Madgwick to get the angle
        Args:
            rawdata : acc, gyro, mag
            mahony : an instance to save the current quaternion
            node : an instance to save the calibration value and maintain the ble connect  

    '''
    global gravityList
    mag = [0.0]*3
    if rawdata[6] != 0 and rawdata[7] != 0 and rawdata[8] != 0:
        
        mag[0] = rawdata[6]*0.15*node.magCalibration[0] - node.magBias[0]
        mag[1] = rawdata[7]*0.15*node.magCalibration[1] - node.magBias[1]
        mag[2] = rawdata[8]*0.15*node.magCalibration[2] - node.magBias[2]
        #print(mag)

        mag[0] *= node.magScale[0]*10
        mag[1] *= node.magScale[1]*10
        mag[2] *= node.magScale[2]*10
    
    acc = S.dot( np.array([rawdata[0],rawdata[1],rawdata[2]]) - B)
    
    # acc = [None]*3
    # acc[0] = rawdata[0]/acc_divider - node.accBias[0]
    # acc[1] = (rawdata[1]/acc_divider - node.accBias[1])
    # acc[2] = (rawdata[2]/acc_divider -node.accBias[2])
    #print ("acc_Bia:",conn_list[count].accBias)
    gyro = [None]*3
    gyro[0] = (rawdata[3]/gyro_divider - node.gyroBias[0])*DEG2RAD
    gyro[1] = (rawdata[4]/gyro_divider - node.gyroBias[1])*DEG2RAD
    gyro[2] = (rawdata[5]/gyro_divider - node.gyroBias[2])*DEG2RAD


    # MahonyAHRSupdate
    mahony.MadgwickAHRSupdate(-gyro[0], gyro[1], -gyro[2], -gravityList[0],gravityList[1],-gravityList[2],-mag[1], mag[0], mag[2],5000/1000000.0)#  rawdata[9]/1000000.0
    # mahony_list[count].MahonyAHRSupdate(gyro[0], gyro[1], gyro[2], acc[0], acc[1], acc[2], 0.0, 0.0, 0.0,5000/1000000.0)#rawdata[9]/1000000.0

    #print("\n")
    return [-acc[0],acc[1],-acc[2],-gyro[0],gyro[1],-gyro[2],-mag[0],mag[1],mag[2],rawdata[9]]

def getYawByMag(normMag,Roll,Pitch):
    # return math.atan2( (normMag[1]*math.cos(Roll) + normMag[2]*math.sin(Roll) ) , (normMag[0]*math.cos(Pitch) + normMag[1]*math.sin(Pitch)*math.sin(Roll) - normMag[2]*math.sin(Pitch)*math.cos(Roll)) ) * 57.2957795
    # return math.atan2( (normMag[0]*math.cos(Pitch) + normMag[2]*math.sin(Pitch) ) , (normMag[0]*math.sin(Pitch)*math.sin(Roll) + normMag[1]*math.cos(Roll) - normMag[2]*math.sin(Roll)*math.cos(Pitch)) ) * 57.2957795
    return math.atan2(normMag[1],normMag[0])* 57.2957795
# saverState = None
def on_press(key):
    global saverState

    try: k = key.char # single-char keys
    except: k = key.name # other keys
    if key == keyboard.Key.esc: return False # stop listener
    if k == '0': # keys interested      
        saverState.value=0
        print('Gloabl')
        print('Key pressed: ' + k)
    if k == '1': # keys interested
        saverState.value=1
        print('Local')
        print('Key pressed: ' + k)
    elif k == '4': # keys interested
        saverState.value=2        
        print('Key pressed: ' + k)
    elif k == '2': # keys interested
        saverState.value=3
        print('Key pressed: ' + k)
        print('Saving Mode')
    elif k=='c' or k == 'C':
        saverState.value=8
        print('Key pressed: ' + k)
        return False
    elif k=='D' or k == 'd':
        saverState.value=9
        print('Key pressed: ' + k)
        return False
 
        # return False # remove this if want more keys
global SMAaverage
global SMAList
global bufferFlag
def SMA(Data,maxLen):
    ''' simple moving average : smoothing data
    
        Args:
            Data : raw data removed gravity

    '''

    global SMAaverage,SMAList,bufferFlag
    if SMAList.shape[0] == maxLen:
        bufferFlag = 1
        SMAaverage = np.mean (SMAList,axis=0)
        SMAaverage = SMAaverage - SMAList[0,:]/float(maxLen) + Data/float(maxLen)
        SMAList = np.delete(SMAList, 0,axis=0)
        SMAList = np.concatenate ((SMAList, Data),axis=0)
        # print SMAaverage
        return SMAaverage
    else:
        SMAList = np.concatenate ((SMAList, Data),axis=0)
        
        return np.zeros((0,6))


    
def emaWeight(numSamples):
    return 2 / float(numSamples + 1)

global EMAaverage
global EMAList 
def EMA(Data,numSamples):
    global EMAList,EMAaverage 
    # print EMAaverage,'1'
    EMAaverage = (Data - EMAaverage)*emaWeight(numSamples) + EMAaverage
    # print EMAaverage,Data
    return EMAaverage

def CalibrationMag(node):
    global saverState

    sample_count = 0;
    mag_bias = [0.0, 0.0, 0.0]
    mag_scale = [0.0, 0.0, 0.0]
    mag_max = [-32760, -32760, -32760]
    mag_min = [32760, 32760, 32760]
    mag_temp = [0.0, 0.0, 0.0]
    print "calibration mag!"
    while saverState.value != 9:
        if node.Peripheral.waitForNotifications(0.01):            
            rawdata = [node.Peripheral.gyrodata[0:4],node.Peripheral.gyrodata[4:8],node.Peripheral.gyrodata[8:12],node.Peripheral.gyrodata[12:16],node.Peripheral.gyrodata[16:20],node.Peripheral.gyrodata[20:24],node.Peripheral.gyrodata[24:28],node.Peripheral.gyrodata[28:32],node.Peripheral.gyrodata[32:36],node.Peripheral.gyrodata[36:40]]
            shortdata = Uint4Toshort(rawdata)
            mag_temp = [ shortdata[6],shortdata[7],shortdata[8] ]

        for i in range(0,3):
            if(mag_temp[i] > mag_max[i]):
                mag_max[i] = mag_temp[i]
            if(mag_temp[i] < mag_min[i]):
                mag_min[i] = mag_temp[i]
    # Get hard iron correction
    mag_bias[0]  = (mag_max[0] + mag_min[0])/2
    mag_bias[1]  = (mag_max[1] + mag_min[1])/2
    mag_bias[2]  = (mag_max[2] + mag_min[2])/2

    mag_bias[0] =  mag_bias[0]*0.15*node.magCalibration[0]  
    mag_bias[1] =  mag_bias[1]*0.15*node.magCalibration[1]   
    mag_bias[2] =  mag_bias[2]*0.15*node.magCalibration[2]           

    # Get soft iron correction estimate
    mag_scale[0]  = (mag_max[0] - mag_min[0])/2
    mag_scale[1]  = (mag_max[1] - mag_min[1])/2
    mag_scale[2]  = (mag_max[2] - mag_min[2])/2

    avg_rad = mag_scale[0] + mag_scale[1] + mag_scale[2];
    avg_rad /= 3.0

    mag_scale[0] = avg_rad/(mag_scale[0]);
    mag_scale[1] = avg_rad/(mag_scale[1]);
    mag_scale[2] = avg_rad/(mag_scale[2]);

    print "Bias",mag_bias,"Scale:",mag_scale
global DTWModel 
global AnglePose
if __name__ == '__main__':

    global saverState,AxisDict,XAction,YAction,ZAction,XModel,YModel,ZModel,SMAList,bufferFlag,EMAList,EMAaverage,DTWModel,JayAnglePose #Xn2p,Xn2pScaler,Xp2n,Xp2nScaler,Yn2p,Yn2pScaler,Yp2n,Yp2nScaler,Zn2p,Zn2pScaler,Zp2n,Zp2nScaler
    SMAList = np.zeros( (0,6) )
    bufferFlag = 0
    EMAList = np.zeros( (0,3) )
    ret = np.zeros( (0,6) )
    Acceration = np.zeros( (0,3) )
    SMAaverage = None
    EMAaverage = None
    
    NotRecognize = 0



    DTWModel = pickle.load(open("JayDTW.dat", "rb"))
    AnglePose = pickle.load(open("JayAnglePose.dat", "rb"))

    lis = keyboard.Listener(on_press=on_press)
    lis.start() # start to listen on a separate thread
    iface = 1
    # try to scan it
    t =threading.Thread(target = ScanProcess,args=( iface, ) ) 
    t.start()
    t.join()
    
    connList = []
    mahony = Mahony.MahonyClass()
    node = myNode() 

        
    connList.append("3c:cd:40:18:c1:98")  

    #connect and Draw
    gravity,yawCalibration,staticLinearAcc,staticLinearGyo = GetBacicData(node,connList[-1],"public",mahony,iface )
    # yawCalibration = 0
    staticLinearAcc = 0.0857
    saverState = Value('i',-1)
    node.nodeCube = myCube()
    mydraw = myOpenGL.myDraw(node.nodeCube)



    plot1 = Array('f',[0.0 for i in range(0,200)])
    plot2 = Array('f',[0.0 for i in range(0,200)])
    plot3 = Array('f',[0.0 for i in range(0,200)])
    plot4 = Array('f',[0.0 for i in range(0,200)])
    plot5 = Array('f',[0.0 for i in range(0,200)])
    plot6 = Array('f',[0.0 for i in range(0,200)])
    plot7 = Array('i',[0 for i in range(0,650)])   
    Timestamp = Array('f',[0.0 for i in range(0,200)])

    resetFlag = Value(c_bool,False)
    Idx =  Value('i',-1)
    staticFlag = Value(c_bool,True)
    isCapturing = Value(c_bool,False)

    print "Drawing"
    # I am not sure it is still work well......
    # QTwebcam ........ plot
    plotMyData = QTwebcam.QtCapture(0)  
    plotRealTimeData = Process(target=QTWebCam,args=(plotMyData,plot1,plot2,plot3,plot4,plot5,plot6,Timestamp,isCapturing,staticFlag,resetFlag))
    plotRealTimeData.daemon=True
    plotRealTimeData.start()

    # QTGraph ........ plot
    # plotMyData = QTRealLine.MyRealTimePlot()  
    # plotRealTimeData = Process(target=QTRun,args=(plotMyData,plot1,plot2,plot3,plot4,plot5,plot6,plot7,Timestamp,Idx,resetFlag,staticFlag))
    # plotRealTimeData.daemon=True
    # plotRealTimeData.start()

    #QTGraph ........ ScatterPlot
    # A scatter graph to visualize the mag and see if the value of mag is normal
    # plotMyData = QTRealTimeScatter.MyRealTimeScatterPlot()  
    # plotRealTimeData = Process(target=QTRun,args=(plotMyData,plot1,plot2,plot3,flag))
    # plotRealTimeData.daemon=True
    # plotRealTimeData.start()
    if saverState.value == 8:
        CalibrationMag(node)


    getNum = 0
    lostNum = 0
    countX  = 0
    seq = 0

    Inst = DataStructure.DataController(0.05,0.015)
    VarInst = DataStructure.AxisVar()
    # how many data to plot
    numberOfPlot = 20
    temp = 0
    # how many samples used for filter
    numSamples = 20
    while True:
                     
        tStart = time.time()
        if node.Peripheral.waitForNotifications(0.01):
            
            getNum = getNum + 1
            # rawdata = [node.Peripheral.gyrodata[0:4],node.Peripheral.gyrodata[4:8],node.Peripheral.gyrodata[8:12],node.Peripheral.gyrodata[12:16],node.Peripheral.gyrodata[16:20],node.Peripheral.gyrodata[20:24],node.Peripheral.gyrodata[24:28],node.Peripheral.gyrodata[28:32],node.Peripheral.gyrodata[32:36],node.Peripheral.gyrodata[36:40]]
            shortdata = node.noti
            removeGravity(shortdata)
            shortdata = doMahony(shortdata,node,mahony)

            # quatern to euler 
            node.nodeCube.angle = mahony.quatern2euler()
            node.nodeCube.angle[0] = node.nodeCube.angle[0] - yawCalibration
            '''After magdwick, we can get the angle and rotation matrix.By rotation matrix, we can get the more precise gravity value'''
            gravityLRF = [-1*gravity*math.sin(node.nodeCube.angle[1]*0.017453) ,gravity*math.cos(node.nodeCube.angle[1]*0.017453)*math.sin(node.nodeCube.angle[2]*0.017453),gravity*math.cos(node.nodeCube.angle[1]*0.017453)*math.cos(node.nodeCube.angle[2]*0.017453)]

            #to make the angle range from -180 - 180. I think it exist some problem with subtracting Magnetic declination from yaw 
            for i in range(0,3):
                if node.nodeCube.angle[i] < -180:
                    node.nodeCube.angle[i]=node.nodeCube.angle[i] +360
                elif node.nodeCube.angle[i] > 180:
                    node.nodeCube.angle[i]=node.nodeCube.angle[i]-360 
            Acceration = np.concatenate( (np.mat([[shortdata[0],shortdata[1] ,shortdata[2] ]])),axis=0 )

            # get linear acceleration  linear acceleration = total acceleration - gravity
            shortdata[0] = shortdata[0] - gravityLRF[0]
            shortdata[1] = shortdata[1] - gravityLRF[1]
            shortdata[2] = shortdata[2] - gravityLRF[2]

            matAcc = np.mat([[shortdata[0],shortdata[1] ,shortdata[2] ]])
            
            # transform data from local frame to global frame, it may have some problem 
            if saverState.value == 0:
                matAcc = matAcc * mahony.EularRotateMatrixInverse(node.nodeCube.angle)
                # matAcc = matAcc * mahony.InverseQuatern()


            # 0  : for global frame , 1 : for local frame  , 3 : save the moving data as dat  
            if saverState.value == 1  or saverState.value ==3 or saverState.value == 0 :
                
                #smoothing data (acc and gyro)  - I found sma with buffer size 30 it is not enough for gyro
                filtterdData = SMA(np.concatenate( (matAcc,np.mat([shortdata[3],shortdata[4],shortdata[5]]) ) ,axis=1),30)
                
                # a flag - when sma buffer is not full we continue to fill it.
                if bufferFlag == 0:
                    continue
                if filtterdData.shape[0] != 0 :

                    # save raw data and filter data to the instance Inst
                    Inst.SetRawXYZData(matAcc,np.mat([shortdata[3],shortdata[4],shortdata[5]]),np.mat([shortdata[6],shortdata[7],shortdata[8]]),np.mat([ node.nodeCube.angle[0], node.nodeCube.angle[1], node.nodeCube.angle[2]]))
                    Inst.SetFilteredXYZData(filtterdData[0,0:3],filtterdData[0,3:6],np.mat([shortdata[6],shortdata[7],shortdata[8]]),np.mat([ node.nodeCube.angle[0], node.nodeCube.angle[1], node.nodeCube.angle[2]]))
                    ret = np.concatenate( (ret, np.concatenate( (Acceration,filtterdData[0,0:3]),axis=1 )) ,axis=0 ) 
                tEnd = time.time()
                node.workingtime = node.workingtime+(tEnd-tStart)
                Inst.timestamp.append ( node.workingtime )
                Inst.sequence.append(seq)
                windowStartTime = Inst.getTimeStamp(Inst.Windows.realTimeStartIdx)
                # print windowStartTime,node.workingtime 
                # print windowStartTime
                #0.05 - 0.015 = 0.035                                                                  # X + 0.05 - 
                if (node.workingtime - windowStartTime >= 0.035  and  Inst.Windows.saveOverLapIdx == True ):
                    Inst.Windows.overLapIdx = seq
                    # print node.workingtime - windowStartTime,seq,node.workingtime
                    Inst.Windows.saveOverLapIdx = False
                if (node.workingtime - windowStartTime >= 0.05  ): 
                    Inst.Windows.saveOverLapIdx   = True             
                    Inst.Windows.RealTimeIdx(seq)
                                        
                    if saverState.value == 1 or saverState.value ==3 or saverState.value == 0:

                        idx = Inst.Windows.windowsIdx[-1]
                        realtime(VarInst,Inst , [staticLinearAcc,staticLinearGyo],[plot1,plot2,plot3,plot4,plot5,plot6],staticFlag,Idx,resetFlag,saverState,plot7,Timestamp,withCamera=False)

                    Inst.Windows.realTimeStartIdx = Inst.Windows.overLapIdx

                    if node.workingtime > 150 and Inst.static == True and saverState.value != 3:
                        # print "reset"
                        Inst = DataStructure.DataController(0.05,0.015)
                        VarInst = DataStructure.AxisVar()
                        node.workingtime = 0 
                        seq = 0
                        continue
                
                # print Inst.dataBlockSet['Acc'].rawXYZ.shape,Inst.dataBlockSet['Acc'].filteredXYZ.shape,seq
                seq = seq + 1

                # if ret.shape[0] == numberOfPlot:
                #     ret = ret.transpose()
                #     # print ret
                #     plot1[0:numberOfPlot] = ret[0,:].tolist()[0]
                #     plot2[0:numberOfPlot] = ret[1,:].tolist()[0]
                #     plot3[0:numberOfPlot] = ret[2,:].tolist()[0]
                #     plot4[0:numberOfPlot] = ret[3,:].tolist()[0]
                #     plot5[0:numberOfPlot] = ret[4,:].tolist()[0]
                #     plot6[0:numberOfPlot] = ret[5,:].tolist()[0]
                #     Idx.value = numberOfPlot
                #     staticFlag.value = False
                #     ret = np.zeros( (0,6) ) 
                countX = countX + 1 

            xx =time.time()

        else:
            lostNum = lostNum + 1
            continue


