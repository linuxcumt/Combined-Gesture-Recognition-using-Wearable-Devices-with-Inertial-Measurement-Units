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

    #connNode.setCalValue(connNode.accBias, connNode.gyroBias,connNode.magBias,connNode.magScale,connNode.magCalibration)
    
    #iface = (iface + 1) % numOfDongle + 1


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
        windowsLen.append([0,1])
        windowsLen.append([0,1])
        windowsLen.append([0,1])
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
            node.gyroBias[0] += rawdata[3] 
            node.gyroBias[1] += rawdata[4]
            node.gyroBias[2] += rawdata[5]
    node.gyroBias[0] = node.gyroBias[0]/gyro_divider/300
    node.gyroBias[1] = node.gyroBias[1]/gyro_divider/300
    node.gyroBias[2] = node.gyroBias[2]/gyro_divider/300

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
S = np.array([[  2.42754810e-04,   3.41614666e-07,  -2.07507663e-07],
[  3.41614666e-07,   2.43926399e-04,   1.68822071e-07],
[ -2.07507663e-07,   1.68822071e-07,   2.43800712e-04]])
B = [-28.43905915,  51.22161875, -72.33527491]


global S,B   


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
    GetBacicData(node,connList[-1],"public",mahony,iface )
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

    # how many data to plot
    numberOfPlot = 20
    temp = 0
    # how many samples used for filter
    numSamples = 20
    while True:
                     
        tStart = time.time()
        if node.Peripheral.waitForNotifications(0.01):
            gyro = [None]*3
            mag = [None]*3
            shortdata = node.noti
            if shortdata[6] != 0 and shortdata[7] != 0 and shortdata[8] != 0:
        
                mag[0] = shortdata[6]*0.15*node.magCalibration[0] - node.magBias[0]
                mag[1] = shortdata[7]*0.15*node.magCalibration[1] - node.magBias[1]
                mag[2] = shortdata[8]*0.15*node.magCalibration[2] - node.magBias[2]
                #print(mag)

                mag[0] *= node.magScale[0]*10
                mag[1] *= node.magScale[1]*10
                mag[2] *= node.magScale[2]*10
    
            acc = S.dot( np.array([shortdata[0],shortdata[1],shortdata[2]]) - B)
            acc = np.asmatrix(acc)

            gyro = [None]*3
            gyro[0] = (shortdata[3]/gyro_divider - node.gyroBias[0])*DEG2RAD
            gyro[1] = (shortdata[4]/gyro_divider - node.gyroBias[1])*DEG2RAD
            gyro[2] = (shortdata[5]/gyro_divider - node.gyroBias[2])*DEG2RAD
            gyro = np.asmatrix(gyro)
            #smoothing data (acc and gyro)  - I found sma with buffer size 30 it is not enough for gyro
            filtterdData = SMA(np.concatenate(  (acc,gyro),axis=1  ) ,30)
            if bufferFlag == 0:
                    continue   

            rawFilterdData = np.concatenate((acc,filtterdData[0,0:3]),axis=1 )
            ret = np.concatenate( (ret,rawFilterdData),axis=0 )
             
            if filtterdData.shape[0] != 0 :            
                if ret.shape[0] == numberOfPlot:
                    ret = ret.transpose()
                    # print ret
                    plot1[0:numberOfPlot] = ret[0,:].tolist()[0]
                    plot2[0:numberOfPlot] = ret[1,:].tolist()[0]
                    plot3[0:numberOfPlot] = ret[2,:].tolist()[0]
                    plot4[0:numberOfPlot] = ret[3,:].tolist()[0]
                    plot5[0:numberOfPlot] = ret[4,:].tolist()[0]
                    plot6[0:numberOfPlot] = ret[5,:].tolist()[0]
                    Idx.value = numberOfPlot
                    staticFlag.value = False
                    ret = np.zeros( (0,6) ) 
                 
        else:
            continue


