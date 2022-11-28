from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import cv2
import numpy as np

#for testing purposes: trying different types of object detection with detectron2 on image, video or webcam

class Detector:
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        #load model config and pretrained model
        if model_type == "OD" : #object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        
        elif model_type == "IS": #instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type == "KP": #keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type == "LVIS": #LVIS segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        
        elif model_type == "PS": #Panoptic segmentation -> currently works best on webcam
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" #cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self,imagePath):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            predictions = self.predictor(image)

            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            instance_mode = ColorMode.IMAGE_BW)

            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
        boxes = predictions["instances"].pred_boxes
        mp = []
        for i in boxes.__iter__():
            results = i.cpu().numpy()
            midpoint = ((int((results[0]+results[2])/2)), (int((results[1]+results[3])/2)))
            mp.append((midpoint[0],midpoint[1]))
        print(mp)
            #print(i.cpu().numpy())
            #print(midpoints[0])
        cv2.namedWindow("Result")
        cv2.putText(image, "MIDPOINT", (30, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)

    def onVideo(self,videoPath):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened()==False):
            print ("Error opening video file...")
            return
        
        (sucess, image) = cap.read()

        while sucess:
            if self.model_type != "PS":
                predictions = self.predictor(image)

                viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode = ColorMode.IMAGE)

                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            else:
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            cv2.imshow("Result", output.get_image()[:,:,::-1])

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            (sucess, image) = cap.read()

    def onWebcam(self):
        cap = cv2.VideoCapture("/dev/video6")

        if (cap.isOpened()==False):
            print ("Error opening webcam...")
            return
        
        (sucess, image) = cap.read()

        while sucess:
            if self.model_type != "PS":
                predictions = self.predictor(image)

                viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode = ColorMode.IMAGE)

                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            else:
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            cv2.imshow("Result", output.get_image()[:,:,::-1])

            key = cv2.waitKey(1)

            if key == 27:
                break
            (sucess, image) = cap.read()