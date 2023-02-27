# Machine Learning Methods for Assistive Solutiond

## Description:

This project aims to assist visually impaired people to navigate indoor environments. It uses pre-trained datasets associated with Facebook's Detectron2 and a depth camera (currently using Intel Realsense D455) to detect and describe objects as well as their distance from the camera.

## Getting Started:

After making sure the prerequisites are met, the project can be cloned and ran on any local machine

## Prerequisites:

Intel Realsense depth camera, preferably the D455 model as the confguration matches it perfectly.
Linux OS (currently using Ubuntu 22.04.1 LTS Jammy Jellyfish)
Python 3 
Numpy, OpenCV, Pyrealsense, Detectron2 libraries
Any Python supporting IDE (currently using Visual Studio Code)


## Installing on Linux:
### Installing Realsense SDK:
  
Official instructions to install Realsense SDK on linux can be found here: https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide

In the case of Ubuntu 22.04 please refer to this link: https://github.com/mengyui/librealsense2-dkms/releases/tag/initial-support-for-kernel-5.15

### Installing Python3:
Most Linux distributions come with Python3 pre-installed, but you can check whether Python3 is already installed on your machine by opening a terminal and running the following command:

    python3 --version
 
 If Python 3 is not installed, you can install it by running the following command:
 
    sudo apt-get install python3
 
 ### Installing OpenCV:
You can install OpenCV on Linux using the following command:
 
    pip install opencv-python
 
### Installing Pyrealsense:
Pyrealsense is a Python wrapper for the Intel RealSense SDK, which provides support for Intel RealSense cameras. To install Pyrealsense, follow these steps:

#### 1. Add the Intel RealSense package repository to your system's package manager:
    sudo apt-key adv --keyserver keys.gnupg.net --recv-key D6FB2970
    sudo sh -c 'echo "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo $(lsb_release -sc) main" > /etc/apt/sources.list.d/realsense-public.list'

#### 2.Update the package manager:
    sudo apt-get update

#### 3. Install the Pyrealsense package:
    pip install pyrealsense2

### Installing Detectron2:
Detectron2 is a computer vision library developed by Facebook AI Research. To install Detectron2, follow these steps:

#### 1. Install the required dependencies if you don't have them yet:
    sudo apt-get install python3-dev python3-pip build-essential cmake

#### 2. Install Pytorch:
    pip install torch torchvision

#### 3. Install Detectron2:
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

for additional assistance visit the official guides in: https://detectron2.readthedocs.ioen/latest/

After installing these libraries, you should be able to use them in your Python projects.


## Usage: 
### 1. main.py
Running main.py will present you with a list of testing units you can choose from:
 
1: Detect objects on image
2: Detect objects on video
3: Detect objects on webcam
4: List detected objects in image
5: List detected objects in video
6: List detected objects from webcam
7: Show average distance in webcam frame
8: Show minimum distance in webcam frame
9: Print position of objects in image
10: Print position of objects in video
11: Print position and distance of object in webcam
12: Quit
                      
                      
Please not that any unit mentioning/using the webcam will not run without having the realsense depth camera plugged in.
Please also note that the tests are done with images and videos already present in the project, if you with to use different images/videos, make sure to place them in the right folders and refer to them correctly in the main menu file.
Selecting an object in order to get information exclusive to it has been recently added and is not part of the menu yet.

### 2. selector/obj_select.py
This file can be compiled and ran on its own. It contains a function exclusive to webcam use so the realsense camera is needed in order for this to work. 
This file's function unables the user to see a list of the objects detected, choose whether they want to select one of them and then if yes, they can type the name of the object. The output would be the distance and position of the selected object alone. 

if you would like the distance and position of all objects detected, please refer to option 11 in the main menu

Please note that the loop that allows the distance calculation is ran every few seconds (depending on the file), this wait period can be changed in the time.sleep(x) line in each file where it applies (mainly obj_select.py, position_in_frame.py and position2.py)

## Authors:
Khadidja Djebairia

## Contributing:
Moeen Valipoor

## License:    
    
    
    
    
    
    
    
    
    
    
    
