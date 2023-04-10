from camera_utils import initialize_camera, get_camera_frames
from detectron_utils import initialize_detectron, run_object_detection, visualize_and_get_detected_objects
from sound_utils import play_beep_sound
from speech_utils import speak, announce_objects
from relationship_utils import describe_relationship
from distance_utils import get_object_distance, get_updated_distance

import cv2
import numpy as np




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
