import cv2
from pyrealsense2 import *
import pyrealsense2.pyrealsense2 as rs
from realsense_depth import *
from object_detections import *

#For testing purposes: realsense camera starts, two frames are displayed, one is exclusively for depth while the other shows current scene.
#In color frame, a circle is drawned at (300,300) and the depth of its center is displayed. If mouse is moved on frame, depth will changed based on its position

center = (300, 300)

def show_distance(event, x, y, args, params):
    global center
    center = (x, y)

# starting device
cam = DepthCamera()

# Creating mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

while True:
    ret, depth_frame, color_frame = cam.get_frame()
    #detector = Detector(model_type="PS")

    # Showing distance for a specific point
    cv2.circle(color_frame, center, 4, (0, 0, 255))
    distance = depth_frame[center[1], center[0]]

    cv2.putText(color_frame, "{}mm".format(distance), (center[0], center[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break    
    #detector.onWebcam()