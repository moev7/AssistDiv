from camera_utils import initialize_camera, get_camera_frames
from detectron_utils import initialize_detectron, run_object_detection, visualize_and_get_detected_objects
from sound_utils import play_beep_sound
from speech_utils import speak, announce_objects, get_voice_input
from relationship_utils import describe_relationship
from distance_utils import get_object_distance, get_updated_distance

import cv2
import numpy as np

beeping_enabled = False
selected_obj_flag = False
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
            speak("Hello, Welcome to Assist Div. Say Scan the scene for scene understanding and Find objects for object detection. Say quit to exit assist div")

            #user_input = input("Select 'u' for scene understanding and 'o' for object detection: ")
            user_input = get_voice_input()
            if user_input == 'scan the scene':
                speak("There are " + str(len(detected_objects)) + " objects detected in the scene. The objects located from left to right side of the frame are:")
                for i, obj in enumerate(detected_objects):
                    print("passed")
                    speak(f"{i + 1}: {obj['name']}")
                repeat = True
                while repeat == True:
                    speak("Say repeat to repeat the object names. Say return to do another task. say quit to exit assist div.")
                    user_input = get_voice_input()
                    if user_input == 'repeat':
                        speak("There are " + str(len(detected_objects)) + " objects detected in the scene. The objects located from left to right side of the frame are:")
                        for i, obj in enumerate(detected_objects):
                            print("passed")
                            speak(f"{i + 1}: {obj['name']}")
                    elif user_input == "return":
                        repeat = False
                    elif user_input == 'quit':
                        break



            if user_input == 'find objects':
                speak("Select one of the following objects to find where it's placed:")
                for i, obj in enumerate(detected_objects):
                    speak(f"{i + 1}: {obj['name']}")
                
                speak("Say Select to select an object and Quit to exit AssistDiv.")
                user_input = get_voice_input()

                if user_input == 'select':
                    selected_obj_distance, selected_obj = get_object_distance(detected_objects, depth_image,720)
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
            


        if beeping_enabled :
            selected_obj_name = selected_obj["name"]
            if any(obj["name"] == selected_obj_name for obj in detected_objects):
                updated_distance = get_updated_distance(selected_obj, detected_objects)
                play_beep_sound(updated_distance)

        if user_input == 'quit':
            break

        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
