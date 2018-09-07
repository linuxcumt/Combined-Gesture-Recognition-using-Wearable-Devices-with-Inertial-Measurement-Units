

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import os
import threading
 
ESCAPE = '\033'
flag = False 

windows = []
cubeList = []



class myDraw:

    def __init__(self, num = 0, mycube = None):

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640,480)
        glutInitWindowPosition(200,200)

        glutCreateWindow('Hello GLUT')
        glutIdleFunc(self.refreshMyDraw)
        glutKeyboardFunc(self.keyPressed)
        self.InitGL(640, 480)
        self.count = 0

    def goLoop(self):
        glutMainLoop()

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
    def refreshMyDraw(self):
        global windows
        print(len(windows))
        for wi in windows:
            print(wi)
            glutSetWindow(wi)
            glutPostRedisplay()

    def add(self,newCube):
        print ("test")
       
        tmp = glutSetWindow(glutCreateWindow('OpenGL Python Cube %d') , newCube.num)
        glutDisplayFunc(newCube.DrawGLScene)
        glutKeyboardFunc(keyPressed)
        InitGL(600, 480)
        windows.append(count)
        count = count +1

class myCube:

    def __init__(self,num = 0,x_axis = 0.0,y_axis = 0.0,z_axis = 0.0):

        self.num = num     
        self.X_AXIS = x_axis
        self.Y_AXIS = y_axis
        self.Z_AXIS = z_axis


    def DrawGLScene(self):
           
     
        self.X_AXIS = self.X_AXIS - 0.10
        self.Z_AXIS = self.Z_AXIS - 0.10


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
         
        glLoadIdentity()
        glTranslatef(0.0,0.0,-6.0)

        glRotatef(self.X_AXIS ,1.0,0.0,0.0)
        glRotatef(self.Y_AXIS ,0.0,1.0,0.0)
        glRotatef(self.Z_AXIS ,0.0,0.0,1.0)
         
            # Draw Cube (multiple quads)
        glBegin(GL_QUADS)
         
        glColor3f(0.0,1.0,0.0)
        glVertex3f( 1.0, 1.0,-1.0)
        glVertex3f(-1.0, 1.0,-1.0)
        glVertex3f(-1.0, 1.0, 1.0)
        glVertex3f( 1.0, 1.0, 1.0) 
         
        glColor3f(1.0,0.0,0.0)
        glVertex3f( 1.0,-1.0, 1.0)
        glVertex3f(-1.0,-1.0, 1.0)
        glVertex3f(-1.0,-1.0,-1.0)
        glVertex3f( 1.0,-1.0,-1.0) 
         
        glColor3f(0.0,1.0,0.0)
        glVertex3f( 1.0, 1.0, 1.0)
        glVertex3f(-1.0, 1.0, 1.0)
        glVertex3f(-1.0,-1.0, 1.0)
        glVertex3f( 1.0,-1.0, 1.0)
     
        glColor3f(1.0,1.0,0.0)
        glVertex3f( 1.0,-1.0,-1.0)
        glVertex3f(-1.0,-1.0,-1.0)
        glVertex3f(-1.0, 1.0,-1.0)
        glVertex3f( 1.0, 1.0,-1.0)
     
        glColor3f(0.0,0.0,1.0)
        glVertex3f(-1.0, 1.0, 1.0) 
        glVertex3f(-1.0, 1.0,-1.0)
        glVertex3f(-1.0,-1.0,-1.0) 
        glVertex3f(-1.0,-1.0, 1.0) 
     
        glColor3f(1.0,0.0,1.0)
        glVertex3f( 1.0, 1.0,-1.0) 
        glVertex3f( 1.0, 1.0, 1.0)
        glVertex3f( 1.0,-1.0, 1.0)
        glVertex3f( 1.0,-1.0,-1.0)
        glEnd() 
        glutSwapBuffers()

def main():

    cubeList.append(myCube(0))
    cubeList.append(myCube(1))

    mydraw = myDraw()
    
    threading.Thread(target = mydraw.goLoop).start()

    mydraw.add(cubeList[0])
    mydraw.add(cubeList[1])
if __name__ == "__main__":
        main() 
