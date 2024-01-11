# detectron_utils.py

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
import cv2
import numpy as np
from speech_utils import speak
from translations import LANGUAGE


def get_objects_by_position_categorized(detected_objects, language):
    categorized_objects = {category: [] for category in LANGUAGE[language].keys()}

    for obj in detected_objects:
        for category, objects in LANGUAGE[language].items():
            if obj['name'] in objects:
                categorized_objects[category].append(obj)

    for category, objects in categorized_objects.items():
        if objects:
            speak(f"Category: {category}", language)
            for i, obj in enumerate(objects, start=1):
                speak(f"{i} - {LANGUAGE[language].get(obj['name'], obj['name'])} at {obj['distance']} meters", language)


def get_objects_by_position(detected_objects, language='en'):
    left_objects = []
    front_objects = []
    right_objects = []

    image_width = 1280

    left_threshold = image_width // 3
    right_threshold = (2 * image_width) // 3

    for obj in detected_objects:
        x_pixels = np.where(obj['mask'])[1]
        left_pixels = np.sum(x_pixels < left_threshold)
        right_pixels = np.sum(x_pixels > right_threshold)
        front_pixels = np.sum((x_pixels >= left_threshold) & (x_pixels <= right_threshold))

        if left_pixels > right_pixels and left_pixels > front_pixels:
            left_objects.append(obj)
        elif right_pixels > left_pixels and right_pixels > front_pixels:
            right_objects.append(obj)
        else:
            front_objects.append(obj)

    for obj in left_objects:
        obj["distance"] = round(obj["distance"], 1)
    for obj in front_objects:
        obj["distance"] = round(obj["distance"], 1)
    for obj in right_objects:
        obj["distance"] = round(obj["distance"], 1)

    if left_objects:
        speak("Objects on the left side are:", language)
        for obj in left_objects:
            speak(f"{LANGUAGE[language].get(obj['name'], obj['name'])} at {obj['distance']} meters", language)

    if front_objects:
        speak("\nObjects in front of you are:", language)
        for obj in front_objects:
            speak(f"{LANGUAGE[language].get(obj['name'], obj['name'])} at {obj['distance']} meters", language)

    if right_objects:
        speak("\nObjects on your right side are:", language)
        for obj in right_objects:
            speak(f"{LANGUAGE[language].get(obj['name'], obj['name'])} at {obj['distance']} meters", language)


def run_object_detection(predictor, color_image):
    outputs = predictor(color_image)
    return outputs




def visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language='en'):
    outputs = run_object_detection(predictor, color_image)
    v = Visualizer(color_image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    detected_objects = []

    distance_image = np.zeros((color_image.shape[0], 400, 3), dtype=np.uint8)
    text_position_start = 30

    categorized_objects = {category: [] for category in LANGUAGE[language].keys()}

    # Iterate over detected objects and categorize them
    for i, (class_idx, mask, box) in enumerate(zip(outputs["instances"].pred_classes.to("cpu"), outputs["instances"].pred_masks.to("cpu"), outputs["instances"].pred_boxes.tensor)):
        instance_mask = mask.cpu().numpy().astype(np.uint8)
        masked_depth_image = depth_image.copy() * instance_mask
        distances = masked_depth_image[np.nonzero(masked_depth_image)]

        # Calculate the mean distance of the object
        if len(distances) > 0:
            mean_distance = np.mean(distances) / 1000
        else:
            mean_distance = 0

        class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_idx]
        # class_name = LANGUAGE[language].get(class_name, class_name)

        y, x = np.nonzero(instance_mask)
        x_center, y_center = int(np.mean(x)), int(np.mean(y))

        centroid = (x_center, y_center)

        detected_object = {
            "id": i,
            "name": class_name,
            "distance": mean_distance,
            "centroid": centroid,
            "box": box,
            "mask": instance_mask
        }

        # Add the object to the categorized_objects
        for category, objects in LANGUAGE[language].items():
            if detected_object['name'] in objects:
                categorized_objects[category].append(detected_object)
                break

    # Display and speak Objects by Category and Position after processing all objects
    for category, objects in categorized_objects.items():
        if objects:
            speak(f"Category: {category}", language)

        # Print Objects by Category
        for i, obj in enumerate(objects, start=1):
            numeration = f"-{i}" if len(objects) > 1 else ""
            speak(f"{category}{numeration}: {LANGUAGE[language].get(obj['name'], obj['name'])} at {obj['distance']:.1f} meters", language)

        # Print Objects by Position
        for i, obj in enumerate(objects, start=1):
            numeration = f"-{i}" if len(objects) > 1 else ""
            text = f"{category}{numeration}: {LANGUAGE[language].get(obj['name'], obj['name'])} at {obj['distance']:.1f} m"
            cv2.putText(distance_image, text, (10, text_position_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_position_start += 30

        # Display the image while printing Objects by Position
        output_image = out.get_image()[:, :, ::-1]
        images_concat = np.hstack((distance_image, output_image))
        cv2.imshow("Distances and Instance Segmentation", images_concat)
        cv2.waitKey(1)

    # Sort detected objects based on the x-coordinate of the centroid
    detected_objects.sort(key=lambda obj: obj['centroid'][0])

    # Return the sorted detected objects
    return detected_objects



def initialize_detectron():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


# Main execution
predictor, cfg = initialize_detectron()

# Set the desired width for your frame
desired_frame_width = 1280  # Change this value to the width you want
desired_frame_height = 680  # Change this value to the height you want

color_image = np.zeros((desired_frame_height, desired_frame_width, 3), dtype=np.uint8)  # Replace with your actual color image
depth_image = np.zeros((desired_frame_height, desired_frame_width), dtype=np.uint16) 


# Example: English
language = 'en'  # Define the language variable
if language == 'en':  # Use '==' for comparison
    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
    get_objects_by_position_categorized(detected_objects, language)
    get_objects_by_position(detected_objects, language)

# Example: Spanish
elif language == 'es':  # Use '==' for comparison
    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
    get_objects_by_position_categorized(detected_objects, language)
    get_objects_by_position(detected_objects, language)
