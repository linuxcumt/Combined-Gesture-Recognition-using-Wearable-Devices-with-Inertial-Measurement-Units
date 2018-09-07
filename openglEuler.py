from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt4 import QtGui
from PyQt4.QtOpenGL import *
import numpy as np
import math
import pickle
import collections
from sklearn import preprocessing
from PyQt4.QtCore import *
from PyQt4.QtGui import *


class LabelDataSet():
    def __init__(self,label,filename):
        # for clazz in self.__class__.__bases__: 
        #     clazz.__init__(self, filename)
        # super(LabelDataSet,self).__init__(filename)
        self.fileName = filename
        self.label=[]
        self.label.append(label)
        self.choiceDataParam=['Angle','Acc','Gyo','Mag']#you should know how many dataSet
        self.choiceAxisParam=[0x7,0x7,0x7,0x7]#you should know  whick axis you want

        self.normalizeData=[]
        self.filteredData=[]
        self.AccMagnitude=[]
        self.GyoMagnitude=[]
        self.MagMagnitude=[]

        self.gravity = -0.964141136218

        self.fileName=filename
        self.OrderedDict_list=[]
        self.keyList=[]
        self.dataSet=[]
        self.readPickle()

    def readPickle(self): #read file
        print self.fileName
        fp = open(self.fileName, "rb")
        while 1:
            try:
                self.OrderedDict_list.append(pickle.load(fp))
            except:
                break    
        self.collectInfo()

    def collectInfo(self):
        self.keyList=self.OrderedDict_list[0].keys()
        self.dataSet=collections.OrderedDict()
        for key in self.keyList:
            self.dataSet[key]=[]
        self.RawDataProcessing()

    def RawDataProcessing(self): #put OrderedDict_list item to be a list by key 
        for item in self.OrderedDict_list:
            for key,value in item.items():
                if len(self.dataSet[key])==0:
                    if "matrix" in  str(type(self.OrderedDict_list[0][key])) :
                        self.dataSet[key]=value
                    else :
                        self.dataSet[key].append(value)
                else:
                    if "matrix" in  str(type(self.dataSet[key])) :
                        self.dataSet[key]=np.concatenate((self.dataSet[key],value), axis=0)
                    else:
                        self.dataSet[key].append(value)
    def printAngle(self):
        for angle,acc in zip(self.dataSet['Angle'],self.dataSet['Acc']):
            print angle,acc

    def lowPassFilter(self, xData, N, Wn):
        b, a = signal.butter(N, Wn)
        yData = signal.filtfilt(b, a, xData)
        return yData  

class MainWindow(QtGui.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.widget = glWidget(self)

        self.button = QtGui.QPushButton('Start', self)
        self.button.clicked.connect(self.getItem)

        self.yawLabel= QtGui.QLabel("Yaw:")
        self.yaw = QtGui.QLineEdit()
        self.rollLabel= QtGui.QLabel("roll:")
        self.roll = QtGui.QLineEdit()
        self.pitchLabel= QtGui.QLabel("pitch:")
        self.pitch = QtGui.QLineEdit()       
        self.yaw.setText("0")
        self.pitch.setText("0")
        self.roll.setText("0")

        self.mainLayout = QtGui.QHBoxLayout()
        secondLayout = QtGui.QVBoxLayout()

        secondLayout.addWidget(self.yawLabel)
        secondLayout.addWidget(self.yaw)
        secondLayout.addWidget(self.pitchLabel)
        secondLayout.addWidget(self.pitch)
        secondLayout.addWidget(self.rollLabel)
        secondLayout.addWidget(self.roll)
        secondLayout.addWidget(self.button)

        self.mainLayout.addWidget(self.widget)
        self.mainLayout.addLayout(secondLayout)

        self.setLayout(self.mainLayout)


    def getItem(self):
        self.widget.doRotate(float(self.yaw.text()),float(self.roll.text()),float(self.pitch.text()))
        self.widget.grabFrameBuffer().save(self.yaw.text()+self.roll.text()+self.pitch.text()+'.png')

class glWidget(QGLWidget):


    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(640, 480)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background-color:white;")

        self.yaw=0
        self.pitch=0
        self.roll=0
    def paintGL(self): #,yaw,roll,pitch
        

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
         
        glLoadIdentity()
        glTranslatef(0.0,0.0,-6.0)

        glRotatef(self.yaw ,0.0,1.0,0.0)#yaw - opengl y
        glRotatef(self.pitch ,1.0,0.0,0.0)#pitch - opengl x
        glRotatef(self.roll ,0.0,0.0,1.0)#roll - opengl z
         
        # Draw Cube (multiple quads)
        glBegin(GL_QUADS)
         
        glColor3f(0.0,1.0,0.0)
        glVertex3f( 2.0, 0.5,-1.0)
        glVertex3f(-2.0, 0.5,-1.0)
        glVertex3f(-2.0, 0.5, 1.0)
        glVertex3f( 2.0, 0.5, 1.0) 
         
        glColor3f(1.0,0.0,0.0)
        glVertex3f( 2.0,-0.5, 1.0)
        glVertex3f(-2.0,-0.5, 1.0)
        glVertex3f(-2.0,-0.5,-1.0)
        glVertex3f( 2.0,-0.5,-1.0) 
         
        glColor3f(0.3,0.6,0.3)
        #glColor3f(1.0,1.0,1.0)
        glVertex3f( 2.0, 0.5, 1.0)
        glVertex3f(-2.0, 0.5, 1.0)
        glVertex3f(-2.0,-0.5, 1.0)
        glVertex3f( 2.0,-0.5, 1.0)
     
        glColor3f(1.0,1.0,0.0)
        glVertex3f( 2.0,-0.5,-1.0)
        glVertex3f(-2.0,-0.5,-1.0)
        glVertex3f(-2.0, 0.5,-1.0)
        glVertex3f( 2.0, 0.5,-1.0)
     
        glColor3f(0.0,0.0,1.0)
        glVertex3f(-2.0, 0.5, 1.0) 
        glVertex3f(-2.0, 0.5,-1.0)
        glVertex3f(-2.0,-0.5,-1.0) 
        glVertex3f(-2.0,-0.5, 1.0) 
     
        glColor3f(1.0,0.0,1.0)
        glVertex3f( 2.0, 0.5,-1.0) 
        glVertex3f( 2.0, 0.5, 1.0)
        glVertex3f( 2.0,-0.5, 1.0)
        glVertex3f( 2.0,-0.5,-1.0)
        glEnd() 
        # glFlush()

    def doRotate(self,yaw=0,roll=0,pitch=0):
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch
        inverseMat = self.EularRotateMatrixInverse([yaw,pitch,roll])
        data = np.mat([[ 0.46854237 ,-0.02791549, -0.25500786]])
        print data*inverseMat  
        self.updateGL()  


    def initializeGL(self):


        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0) 
        # glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)   
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(640)/float(480), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def EularRotateMatrixInverse(self,angles):
        rotmat = [[0,0,0],[0,0,0],[0,0,0]]
        Yaw =  angles[0] * 0.01745329251
        Pitch =  angles[1] * 0.01745329251
        Roll =  angles[2] * 0.01745329251

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


if __name__ == '__main__':
    app = QtGui.QApplication(['Yo'])
    window = MainWindow()
    window.show()
    app.exec_()