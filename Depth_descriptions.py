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
        ret, depth_frame, color_frame = cam.get_frame()

        #calculate avg and min depth of frame in general
        avg = np.mean(color_frame)
        min = np.amin(color_frame)

        # Show distance for a specific point (minimum and average in frame)
        cv2.circle(color_frame, point, 5, (0, 0, 230))

        avg_distance = int(avg)
        min_distance = int(min)

        cv2.putText(color_frame, "{} mm".format(avg_distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(color_frame, "{} mm".format(min_distance), (point[0], point[1] - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        #Show description of objects in boxes
        predictions = predictor(color_frame)
        viz = Visualizer(color_frame, metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        #find framing boxes coordinates and their midpoints + midpoints' distance
        ##midpoints' distance can be printed in cpu but won't display on frame...
        boxes = predictions["instances"].pred_boxes
        mp = []
        dis = []
        for i in boxes.__iter__():
            results = i.cpu().numpy()
            midpoint = ((int((results[0]+results[2])*0.5)), (int((results[1]+results[3])*0.5)))
            dist = depth_frame[midpoint[1] , midpoint[0]]
            mp.append((midpoint[0],midpoint[1]))
            dis.append(int(dist))
            #print(dist)
        cv2.putText(color_frame, str(dist), (point[0]-10, point[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(color_frame, "{}.".format(output.get_image()), (point[0], point[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        """"
        #This prints the names of all categories of model, will be used for listing all objects present in a frame

        names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes 
        f = predictions._fields
        for x in f["pred_classes"]:
            print(x)
        """

        #display the output
        cv2.imshow("Color frame", output.get_image()[:,:,::-1])
        


        key = cv2.waitKey(1)
        if key == 27:
            break   
        