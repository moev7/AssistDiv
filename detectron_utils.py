import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog
import cv2
import numpy as np

def initialize_detectron():
    cfg = get_cfg()
    #load model config and pretrained model
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda" #cpu or cuda
    predictor = DefaultPredictor(cfg)
    return predictor



def run_object_detection(predictor, color_image):
    outputs = predictor(color_image)
    return outputs

def visualize_and_get_detected_objects(color_image, depth_image, outputs, cfg):
    v = Visualizer(color_image, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    detected_objects = []

    for i, (class_idx, mask, box) in enumerate(zip(outputs["instances"].pred_classes.to("cpu"), outputs["instances"].pred_masks.to("cpu"), outputs["instances"].pred_boxes.tensor)):
        instance_mask = mask.cpu().numpy().astype(np.uint8)
        masked_depth_image = depth_image.copy() * instance_mask
        distances = masked_depth_image[np.nonzero(masked_depth_image)]

        if len(distances) > 0:
            mean_distance = np.mean(distances) / 1000
        else:
            mean_distance = 0

        distance_text = f"{mean_distance:.2f} m" 
        class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_idx]

        y, x = np.nonzero(instance_mask)
        x_center, y_center = int(np.mean(x)), int(np.mean(y))

        centroid = (x_center, y_center)

        detected_objects.append({
            "id": i,
            "name": class_name,
            "distance": mean_distance,
            "centroid": centroid,
            "box": box,
            "mask": instance_mask
        })
        cv2.putText(color_image, str(distance_text), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    output_image = out.get_image()[:, :, ::-1]
    output_image_resized = cv2.resize(output_image, (color_image.shape[1], color_image.shape[0]))

    cv2.imshow("Instance Segmentation and Distance", np.hstack((color_image, output_image_resized)))

    # Sort detected_objects based on the x-coordinate of centroids
    detected_objects.sort(key=lambda obj: obj['centroid'][0])

    return detected_objects



def initialize_detectron():
    cfg = get_cfg()
    #load model config and pretrained model
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda" #cpu or cuda
    predictor = DefaultPredictor(cfg)
    return predictor, cfg