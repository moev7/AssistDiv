from camera_utils import initialize_camera, get_camera_frames
from detectron_utils import initialize_detectron, run_object_detection, visualize_and_get_detected_objects, get_objects_by_position
from sound_utils import play_beep_sound, play_obstacle_beep_sound
from speech_utils import speak, announce_objects, get_voice_input
from relationship_utils import describe_relationship
from distance_utils import get_object_distance, get_updated_distance
import time
import cv2
import numpy as np

beeping_enabled = False
selected_obj_flag = False
pipeline = initialize_camera()
depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame = get_camera_frames(pipeline)

predictor, cfg = initialize_detectron()
print("FRAME WIDTH: ")

#detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
frame_width = color_image.shape[1]
print(frame_width)
section_width = frame_width // 3

left_boundary = section_width
right_boundary = section_width * 2

try:
    while True:
        detected_objects = []

        # Get camera frames
        depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame = get_camera_frames(pipeline)
        detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
        #print("second pass")        
        
        
        if not beeping_enabled or not selected_obj_flag: 
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
            
            #speak("Hello, Welcome to Assist Div. Say Scan the scene for scene understanding and Find objects for object detection. Say quit to exit assist div")
            
            #user_input = input("Select 'u' for scene understanding and 'o' for object detection: ")
            user_input = 'scan the scene' #get_voice_input()
            print(user_input)

            if user_input == 'scan the scene':
                repeat = True
                detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
                get_objects_by_position(detected_objects)

                while repeat == True:
                    speak("Say repeat to repeat the object names. Say return to do another task. say quit to exit assist div.")
                    user_input = get_voice_input()
                    print(user_input)
                    if user_input == 'repeat':
                        get_objects_by_position(detected_objects)
                    elif user_input == "return":
                        repeat = False
                    elif user_input == 'quit':
                        break
                        
            if user_input == 'find objects':
                detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
                
                speak("Select one of the following objects to find where it's placed:")
                '''
                for i, obj in enumerate(detected_objects):
                    speak(f"{i + 1}: {obj['name']}")
                '''
                
                speak("Say Select to select an object and say Quit to exit AssistDiv.")
                user_input = "select" #get_voice_input()
                print(user_input)

                if user_input == 'select':
                    selected_obj_distance, selected_obj = get_object_distance(detected_objects, depth_image, 640)
                    selected_obj_flag = True
                elif user_input == 'start beeping':
                    beeping_enabled = True
                elif user_input == 'quit':
                    break

        if selected_obj_flag and not beeping_enabled :
            selected_obj_name = selected_obj["name"]
            if any(obj["name"] == selected_obj_name for obj in detected_objects):
                updated_distance = get_updated_distance(selected_obj, detected_objects)
                speak("Say start beeping to enable beeping, or say quit to exit AssistDiv: ")
                user_input = get_voice_input()
                if user_input == 'start beeping':
                    beeping_enabled = True
                elif user_input == 'quit':
                    break

        if beeping_enabled and selected_obj:
            detected_objects = []
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
            updated_distance, centroid = get_updated_distance(selected_obj, detected_objects)
            print(centroid)
            print(selected_obj)
            if centroid != 0:
                # Check if the object's centroid is in the center third of the frame
                if centroid[0] < left_boundary:
                    speak("Object is on your left side")
                elif centroid[0] > right_boundary:
                    speak("Object is on your right side")
                else:
                    play_beep_sound(updated_distance)
            else:
                speak("Objects is not in the frame")
        #play_obstacle_beep_sound(depth_image)
        
        if user_input == 'quit':
            break
        
        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
