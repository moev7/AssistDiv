import cv2
import numpy as np
from pyrealsense2 import *
import pyrealsense2.pyrealsense2 as rs
import realsense_depth as RD
from realsense_depth import *
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import time


cfg = get_cfg()

#load model config and pretrained model

cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda" #cpu or cuda

predictor = DefaultPredictor(cfg)


def Select_onWebcam():
# Initialize Camera Intel Realsense
    cam = DepthCamera()
    depth_frame, color_frame = cam.get_frame()
    predictions = predictor(color_frame)

    while True:

        #Show description of objects in boxes

        viz = Visualizer(color_frame, metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        #find framing boxes coordinates and their midpoints + midpoints' distance
        ##midpoints' distance can be printed in cpu but won't display on frame

        boxes = predictions["instances"].pred_boxes
        detected_class_indexes = predictions["instances"].pred_classes
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        class_catalog = metadata.thing_classes
        info = []

        for i, coordinates in enumerate(boxes):
            results = coordinates.cpu().numpy()
            midpoint = ((int((results[0]+results[2])*0.5)), (int((results[1]+results[3])*0.5)))
            dist = depth_frame[midpoint[1] , midpoint[0]]
            class_index = detected_class_indexes[i]
            class_name = class_catalog[class_index]
            if dist>0:
                info.append((class_name, dist, [midpoint[0], midpoint[1]]))

            #determining position on frame
            if dist == 0:
                continue
        if len(info):
            for c in info:
                print(c[0])

        #selecting an object
        inp = input("Would you like to select an object? no=0/yes=1/exit=2")
        if inp == '0':
            pass
        if inp == 2:
            break
        if inp == '1':
            selected = input("Please provide the element you'd like to select: ")
            for e in info:
                if e[0] == selected:
                    if e[1] <= 300:
                        print("The ", e[0], " is quite close to the camera")
                    if (e[2][0]<=213):
                        print("The ", e[0], " is ", e[1]/1000, " m away on your left side")
                    if (e[2][0]>213 and e[2][0]<=427):
                        print("The ", e[0], " is ", e[1]/1000, " m away in front of you")
                    if (e[2][0]>427):
                        print("The ", e[0], " is ", e[1]/1000, " m away on your right side")



        print("Rescanning scene...")
        print("\n")

        time.sleep(2)

        #display the output
        cv2.imshow("Color frame", output.get_image()[:,:,::-1])





        key = cv2.waitKey(2)
        if key == 27:
            break   
Select_onWebcam()