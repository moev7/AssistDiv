import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import torch
import time
import pyttsx3
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
import pygame
import time
import math
from gtts import gTTS
import os
import tempfile

from num2words import num2words

def initialize_camera():
    # Initialize the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.accel)
    pipeline.start(config)
    return pipeline

def get_camera_frames(pipeline):
    # Get camera frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Get the IMU frames
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    accel_frame = frames.first_or_default(rs.stream.accel)

    return depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame

pygame.mixer.init()

def play_beep_sound(updated_distance):
    beep = pygame.mixer.Sound("/home/goforem/Documents/AssistDiv_New/beep-07a.wav")
    if updated_distance > 2:
        interval = 2.0  # 1 second
    elif updated_distance > 1:
        interval = 0.6  # 600 milliseconds
    elif updated_distance > 0.5:
        interval = 0.3  # 300 milliseconds
    else:
        interval = 0.1  # 100 milliseconds

    beep.play()
    time.sleep(interval)



def describe_relationship(selected_obj, detected_objects):
    relationships = []
    for obj in detected_objects:
        if obj is not selected_obj:
            distance_diff = abs(obj['distance'] - selected_obj['distance'])
            if distance_diff <= 1:
                rel = ""
                print("DISTANCE DIFF " + str(distance_diff))

                ''' Check if the y-coordinate of the current object's centroid (obj['centroid'][1]) is less than the y-coordinate of the selected_obj's centroid minus 
                half the height of the selected_obj's bounding box.
                if true, the current object is considered to be "above" the selected_obj, and the relationship description rel is updated accordingly.'''
                
                if obj['centroid'][1] < selected_obj['centroid'][1] - selected_obj['box'][3] * 0.5:
                    rel += "above "
                elif obj['centroid'][1] > selected_obj['centroid'][1] + selected_obj['box'][3] * 0.5:
                    rel += "below "
                if obj['centroid'][0] < selected_obj['centroid'][0] - selected_obj['box'][2] * 0.5:
                    rel += "on the left side of "
                elif obj['centroid'][0] > selected_obj['centroid'][0] + selected_obj['box'][2] * 0.5:
                    rel += "on the right side of "

                if not rel:
                    rel = "around "

                relationships.append((obj, rel))

    if not relationships:
        print(f"No other objects detected around the {selected_obj['name']}.")
    else:
        for obj, rel in relationships:
            print(f"{obj['name']} is {rel}the {selected_obj['name']}.")
            speak(f"{obj['name']} is {rel}the {selected_obj['name']}.")


def speak(text, language='en', slow=False):
    tts = gTTS(text=text, lang=language, slow=slow)
    
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        temp_file = fp.name + ".mp3"
        tts.save(temp_file)
        os.system(f"mpg123 {temp_file}")

def announce_objects(detected_objects):
    if not detected_objects:
        speak("No objects detected.")
    else:
        obj_count = len(detected_objects)
        speak(f"Detected {obj_count} objects.")
        for obj in detected_objects:
            speak(f"{obj['name']} at {obj['distance']:.2f} meters.")

def get_object_distance(detected_objects, frame_width):
    if not detected_objects:
        print("No objects detected.")
        return

    print("Available objects:")
    for i, obj in enumerate(detected_objects):
        print(f"{i + 1}: {obj['name']}")
    print(f"{len(detected_objects) + 1}: Cancel")

    try:
        index = int(input("Enter the number of the object you want the distance for, or choose 'Cancel': ")) - 1
        if 0 <= index < len(detected_objects):
            selected_obj = detected_objects[index]
            x_center, y_center = selected_obj["centroid"]

            direction = "in front of you"
            if x_center < frame_width / 3:
                direction = "on your left side"
            elif x_center > 2 * frame_width / 3:
                direction = "on your right side"

            print(f"The {selected_obj['name']} is {direction} {selected_obj['distance']:.2f} meters away.")

            distance = selected_obj['distance']
            integer_part, decimal_part = divmod(distance, 1)
            decimal_part = round(decimal_part * 100)
            integer_part_in_words = num2words(int(integer_part))
            decimal_part_in_words = num2words(int(decimal_part))

            speak(f"The {selected_obj['name']} is {direction} {integer_part_in_words} point {decimal_part_in_words} meters away.")
            print("\nRelationships with other objects:")
            describe_relationship(selected_obj, detected_objects)

        elif index == len(detected_objects):
            print("Operation cancelled.")
        else:
            print("Invalid input. Please enter a number within the available range.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    return selected_obj['distance'], selected_obj


def get_updated_distance(selected_obj, detected_objects):
    print("Detected objects:", detected_objects)
    print("Selected object:", selected_obj)
    for obj in detected_objects:
        if obj["id"] == selected_obj["id"]:
            return obj["distance"]
    return None


def initialize_detectron():
    cfg = get_cfg()
    #load model config and pretrained model
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda" #cpu or cuda
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

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
            "box": box
        })
        cv2.putText(color_image, str(distance_text), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    output_image = out.get_image()[:, :, ::-1]
    output_image_resized = cv2.resize(output_image, (color_image.shape[1], color_image.shape[0]))
    
    cv2.imshow("Instance Segmentation and Distance", np.hstack((color_image, output_image_resized)))

    return detected_objects


beeping_enabled = False
pipeline = initialize_camera()

predictor, cfg = initialize_detectron()
try:
    while True:
        detected_objects = []
        
        # Get camera frames
        depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame = get_camera_frames(pipeline)
        

        # Run object detection
        outputs = run_object_detection(predictor, color_image)

        detected_objects = visualize_and_get_detected_objects(color_image, depth_image, outputs, cfg)
        if not beeping_enabled or not selected_obj_flag:        
            #speak("Hello, Welcome to AssistDiv. Select U for scene understanding and O for object detection.")

            user_input = input("Select 'u' for scene understanding and 'o' for object detection: ")
            if user_input.lower() == 'u':
                speak("There are " + str(len(detected_objects)) + " objects detected in the scene.")
                for i, obj in enumerate(detected_objects):
                    speak(f"{i + 1}: {obj['name']}")

            if user_input.lower() == 'o':
                speak("Select one of the following objects to find where it's placed:")
                for i, obj in enumerate(detected_objects):
                    speak(f"{i + 1}: {obj['name']}")
                user_input = input("Enter 's' to select an object or 'q' to quit: ")

                if user_input.lower() == 's':
                    selected_obj_distance, selected_obj = get_object_distance(detected_objects, 640)
                    selected_obj_flag = True
                elif user_input.lower() == 'b':
                    beeping_enabled = True
                elif user_input.lower() == 'q':
                    break

        if selected_obj_flag and not beeping_enabled :
            selected_obj_name = selected_obj["name"]
            if any(obj["name"] == selected_obj_name for obj in detected_objects):
                updated_distance = get_updated_distance(selected_obj, detected_objects)
                speak("Enter 'b' to enable beeping, or 'q' to quit: ")
                user_input = input("Enter 'b' to enable beeping, or 'q' to quit: ")
                if user_input.lower() == 'b':
                    beeping_enabled = True
                elif user_input.lower() == 'q':
                    break
            


        if beeping_enabled :
            selected_obj_name = selected_obj["name"]
            if any(obj["name"] == selected_obj_name for obj in detected_objects):
                updated_distance = get_updated_distance(selected_obj, detected_objects)
                play_beep_sound(updated_distance)


        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

