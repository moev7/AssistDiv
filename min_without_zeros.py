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

"""
This file explores the possibility of avoiding the zero value for the minimum distance.
Even without zeros, the minimum would appear to be 1, 2 or 3 mm, which is still inacurate
"""


cfg = get_cfg()

#load model config and pretrained model

cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda" #cpu or cuda

predictor = DefaultPredictor(cfg)

point = (300, 300)



def onWebcam():

# Initialize Camera Intel Realsense
    cam = DepthCamera()

    while True:
        depth_frame, color_frame = cam.get_frame()

        #calculate min depth of frame in general and exlude 0 
        
        masked_a = np.ma.masked_equal(color_frame, 0, copy=False)
        min = masked_a.min()

        # Show distance for a specific point (minimum in frame)
        cv2.circle(color_frame, point, 5, (0, 0, 230))
        cv2.putText(color_frame, "{} mm".format(int(min)), (point[0], point[1] - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        #Show description of objects in boxes
        predictions = predictor(color_frame)
        viz = Visualizer(color_frame, metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        #display the output
        cv2.imshow("Color frame", output.get_image()[:,:,::-1])
        


        key = cv2.waitKey(1)
        if key == 27:
            break   
        