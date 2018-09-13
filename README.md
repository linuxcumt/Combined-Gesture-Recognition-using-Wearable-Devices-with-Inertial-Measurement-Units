# Combined-Gesture-Recognition-using-Wearable-Devices-with-Inertial-Measurement-Units
<h3>Requirement:</h3> <br />


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
  
  
<h3>Issue: </h3><br />


  For some unknow reason the webcam can not execute successfully. <br />
  Maybe we can try github: <a href=https://github.com/gebart/python-v4l2capture> python-v4l2capture </a><br />
  , instead of using pip to install the v4l2capture package.: Not tried yet <br />
  the example code: http://www.morethantechnical.com/2016/03/04/python-opencv-capturing-from-a-v4l2-device/
<h3>Some idea what I want to do:</h3><br />


  利用攝影機做連續的動作切割，並且標出切割的時間點。利用時間點，做為ground truth. <br />
  拿sensor 的資料做為Input，利用LSTM 來training，讓LSTM能只靠sensor的資料就能切割出連續動作 <br />  
  難點: sensor 資料如何與影像同步 <br />



  
  
