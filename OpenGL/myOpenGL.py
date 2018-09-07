from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import os
import threading
 
ESCAPE = '\033'

cubeList = [] # for demo in this code
count = 1



lock = threading.Lock()

class myDraw:
 
    def __init__(self,cube):

        # self.v = nodeList
        self.window = []   
        self.deleWindowNumber = -1
        self.cube=cube

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(300,240)
        glutInitWindowPosition(0,0)

        windowNum = glutCreateWindow('OpenGL Python Cube 1')
        glutDisplayFunc(self.cube.DrawGLScene)
        self.cube.drawWindowNumber = windowNum

        glutKeyboardFunc(self.keyPressed)
        self.InitGL(300, 240)
        self.window.append(windowNum)
        
        glutIdleFunc(self.refresh)   #key point

        threading.Thread(target = glutMainLoop).start()

    def InitGL(self,Width, Height): 
 
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0) 
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)   
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def keyPressed(self,*args):
        if args[0] == ESCAPE:
                sys.exit()

    def delWindowNum(self,windowNum):
        self.deleWindowNumber = windowNum

    def refresh(self):

        ## delete window
        if self.deleWindowNumber != -1:
            
            self.window.remove(self.deleWindowNumber)

            try:
                print "try to destory window"
                glutDestroyWindow(self.deleWindowNumber)
            except:
                print "error"
            self.deleWindowNumber = -1

        ## update the current condition
        for wi in self.window:   
            glutSetWindow(wi)
            glutPostRedisplay()

        ## find new node ,make a new window
        # if(len(self.window) < len(self.nodeList)):

        #     for node in self.nodeList:
        #         if(node.drawWindowNumber == -1):
        #             windowName = 'OpenGL Python Cube %d' % (len(self.window)+1)
        #             windowNum = glutCreateWindow(windowName) # return number of windows
        #             node.drawWindowNumber = windowNum

        #             glutSetWindow(windowNum)
        #             glutDisplayFunc(node.nodeCube.DrawGLScene)
        #             glutKeyboardFunc(self.keyPressed)
        #             self.InitGL(600, 480)
        #             self.window.append(windowNum)

class myCube(object):

    def __init__(self,num = 0,x_axis = 0.0,y_axis = 0.0,z_axis = 0.0):

        self.num = num     
        self.angle =[0.0,0.0,0.0]

        self.angle[0] = y_axis
        self.angle[1]  = x_axis
        self.angle[2]  = z_axis

    def DrawGLScene(self):
           
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
         
        glLoadIdentity()
        glTranslatef(0.0,0.0,-6.0)

        glRotatef(self.angle[0] ,0.0,1.0,0.0)#yaw - opengl y
        glRotatef(self.angle[1] ,1.0,0.0,0.0)#pitch - opengl x
        glRotatef(self.angle[2] ,0.0,0.0,1.0)#roll - opengl z
         
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
         
        glColor3f(1.0,1.0,1.0)
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
        glutSwapBuffers()


def Demomain():
        cube0 = myCube(0)
        cube1 = myCube(1)

        cubeList.append(cube0)
        cubeList.append(cube1)

        if len(cubeList) > 0:
            mydraw = myDraw()

        while(len(cubeList) < 10):
            cubeList.append(myCube(len(cubeList)+1))
        print("exit while loop ,len = %d" % len(cubeList))


if __name__ == "__main__":
        Demomain() 
