# Combined-Gesture-Recognition-using-Wearable-Devices-with-Inertial-Measurement-Units

<h3>Introduction:</h3> <br />
事先定義好幾個基本動作，並且藉由這些基本動作組成不同的 複合型手勢。理論上，我們只需要做每個基本手勢一次就可以了。<br />
複合型手勢 則會將其切割成好幾個基本手勢再做辨識。<br />
切割手勢的方式 Threshold 以及 angle<br />
辨識的方法使用 DTW(Dynamic Time Wrapping)<br />
*切割的方法 : 如果用deep learning 可能會需要 複合型手勢 切割的時間點的ground truth(不易取得，可以利用攝影機，webcam)，且讓計算成本更高了。我們定義出來的基本手勢是簡單的，因此採用threshold base 以及angle 來做切割。<br /><br />
*辨識應該可以換成其他的 ML model,是因為DTW適合比對不同長度的sequence data.如果換成其他ML，有可能得收集更多的資料，要做基本手勢好幾次。<br /><br />




<h3>Requirement:</h3> <br />

*** Run on python 2.7<br />
  sudo apt-get install python-xlib<br />
  pip install pynput==1.2<br />
  sudo apt-get install python-pip libglib2.0-dev<br />
  sudo pip install bluepy<br />
  sudo apt-get install libv4l-dev<br />
  pip install v4l2capture<br />
  pip install Pillow<br />
  sudo apt-get install python-imaging<br />
  sudo apt-get install python-opencv<br />
  sudo apt-get install python-qt4<br />
  pip install pyqtgraph<br />

<h3>File Description: </h3><br />
  
  CGRrealtime.py : Recognize gesture in real-time.Load the DTW pattern: JayDTW.dat<br /><br />
  QTRealLine.py : Draw the Real-Time curve using pyqt & pyqtgraph for accelerometer, gyroscope data and Euler angle(Yaw,Roll Pitch) <br /> <br />
  QTRealTimeScatter.py : Draw the scatter data to see if the magnetometer is calibrated successfully<br /><br />
  QTwebcam.py : Draw the Real-Time curve and captrue the video by webcam<br /><br />
  openglEuler.py : Combined OpenGL and PYQT. Can demo the Gimbal Lock problem<br />
  It can show data transformed from sensor frame to be a global frame by multiplied rotation matrix<br /> <br />
  Mahony.py : a python library for Madgwick and Mahony. It repharses from C <a href=http://x-io.co.uk/open-source-imu-and-ahrs-algorithms/> x-io Technologies </a> <br /> <br />
*** Maybe I will write a document to elaborate on Madgwick and the math of rotation.<br />
  The template of drawing real-time can be find in the other <a href=https://github.com/nthuepl/Realtime-Plot-Template> github repository </a>  <br /><br />
  
  <h4>The Offline:</h4><br />  
  Offline/Jay/ : save all motion data<br />  <br /> <br />
  QTofflineRealLine.py : To create the DTW pattern, and test the accuracy of the motion data<br /> <br /><br />
  

  
 
  
<h3>Issue: </h3><br />


  For some unknow reason the webcam can not execute successfully. <br />
  Maybe we can try github: <a href=https://github.com/gebart/python-v4l2capture> python-v4l2capture </a><br />
  , instead of using pip to install the v4l2capture package.: Not tried yet <br />
  the example code: http://www.morethantechnical.com/2016/03/04/python-opencv-capturing-from-a-v4l2-device/
<h3>Some idea what I want to do:</h3><br />


  利用攝影機做連續的動作切割，並且標出切割的時間點。利用時間點，做為ground truth. <br />
  拿sensor 的資料做為Input，利用LSTM 來training，讓LSTM能只靠sensor的資料就能切割出連續動作 <br />  
  難點: sensor 資料如何與影像同步 <br />



  
  
