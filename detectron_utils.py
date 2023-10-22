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
from speech_utils import speak

def get_objects_by_position(detected_objects):
    left_objects = []
    front_objects = []
    right_objects = []

    image_width = 720  # Assuming the width of the image is 720 pixels
    
    left_threshold = image_width // 3
    right_threshold = (2 * image_width) // 3
    
    for obj in detected_objects:
        # Get all the x coordinates of the pixels occupied by the object
        x_pixels = np.where(obj['mask'])[1]
        left_pixels = np.sum(x_pixels < left_threshold)
        right_pixels = np.sum(x_pixels > right_threshold)
        front_pixels = np.sum((x_pixels >= left_threshold) & (x_pixels <= right_threshold))

        # Choose the section where the object has the most pixels
        if left_pixels > right_pixels and left_pixels > front_pixels:
            left_objects.append(obj)
            print("left objs: ")
            print(left_objects)
        elif right_pixels > left_pixels and right_pixels > front_pixels:
            right_objects.append(obj)
            print("right objs: ")
            print(right_objects)
        else:
            front_objects.append(obj)
            print("front objs: ")
            print(front_objects)

    # Round the distance to one decimal place for each object
    for obj in left_objects:
        obj["distance"] = round(obj["distance"], 1)
    for obj in front_objects:
        obj["distance"] = round(obj["distance"], 1)
    for obj in right_objects:
        obj["distance"] = round(obj["distance"], 1)
    
    speak("There are " + str(len(detected_objects)) + " objects detected in the scene.")

    if left_objects:
        speak("Objects on the left side are:")
        for obj in left_objects:
            speak(f"{obj['name']} at {obj['distance']} meters")

    if front_objects:
        speak("\nObjects in front of you are:")
        for obj in front_objects:
            speak(f"{obj['name']} at {obj['distance']} meters")

    if right_objects:
        speak("\nObjects on your right side are:")
        for obj in right_objects:
            speak(f"{obj['name']} at {obj['distance']} meters")



def run_object_detection(predictor, color_image):
    outputs = predictor(color_image)
    return outputs


'''
def visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg):
    outputs = run_object_detection(predictor, color_image)
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
    cv2.waitKey(1000)
    #cv2.destroyWindow("Instance Segmentation and Distance")

    # Sort detected_objects based on the x-coordinate of centroids
    detected_objects.sort(key=lambda obj: obj['centroid'][0])

    return detected_objects
'''

import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

def visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg):
    outputs = run_object_detection(predictor, color_image)
    v = Visualizer(color_image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    detected_objects = []

    # Create a black image to display distances
    distance_image = np.zeros((color_image.shape[0], 200, 3), dtype=np.uint8)  # width of 300px

    text_position_start = 30  # y position to start writing text

    for i, (class_idx, mask, box) in enumerate(zip(outputs["instances"].pred_classes.to("cpu"), outputs["instances"].pred_masks.to("cpu"), outputs["instances"].pred_boxes.tensor)):
        instance_mask = mask.cpu().numpy().astype(np.uint8)
        masked_depth_image = depth_image.copy() * instance_mask
        distances = masked_depth_image[np.nonzero(masked_depth_image)]

        if len(distances) > 0:
            mean_distance = np.mean(distances) / 1000  # assuming the depth is in millimeters
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

        # Write class name and distance to the distance image
        text = f"{class_name}: {distance_text}"
        cv2.putText(distance_image, text, (10, text_position_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_position_start += 30  # move down by 30px for next text

    output_image = out.get_image()[:, :, ::-1]

    # Concatenate images horizontally
    images_concat = np.hstack((distance_image, output_image))

    cv2.imshow("Distances and Instance Segmentation", images_concat)
    cv2.waitKey(1)  # Change this value as needed or use cv2.waitKey(0) to wait until any key is pressed
    #cv2.destroyAllWindows()

    # Sort detected_objects based on the x-coordinate of centroids
    detected_objects.sort(key=lambda obj: obj['centroid'][0])

    return detected_objects


def initialize_detectron():
    cfg = get_cfg()
    #load model config and pretrained model
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.MODEL.DEVICE = "cuda" #cpu or cuda
    predictor = DefaultPredictor(cfg)
    return predictor, cfg
