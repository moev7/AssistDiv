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


def posOnWebcam():

# Initialize Camera Intel Realsense
    cam = DepthCamera()

    while True:
        
        depth_frame, color_frame = cam.get_frame()
        predictions = predictor(color_frame)
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
                info.append((class_name, dist))
            
            if dist == 0:
                continue
            if dist <= 300:
                print("The ", class_name, " is quite close to the camera")
            if (midpoint[0]<=213):
                print("The ", class_name, " is ", dist/1000, " m away on your left side")
            if (midpoint[0]>213 and midpoint[0]<=427):
                print("The ", class_name, " is ", dist/1000, " m away in front of you")
            if (midpoint[0]>427):
                print("The ", class_name, " is ", dist/1000, " m away on your right side")

            #engine.say(buffer.getvalue())
        print("Rescanning scene...")
        print("\n")
        
        time.sleep(5)
        #display the output
        cv2.imshow("Color frame", output.get_image()[:,:,::-1])
        


        key = cv2.waitKey(1)
        if key == 27:
            break   
        
posOnWebcam()

#more user-friendly print on cpu, currently using 2s sleep
#not the same things are found in all or consecutive frames so avg is hard to find
#current structure doesn't allow distance separation based on name for avg calculation due to loop