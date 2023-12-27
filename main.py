from camera_utils import initialize_camera, get_camera_frames
from detectron_utils import initialize_detectron, run_object_detection, visualize_and_get_detected_objects, get_objects_by_position, get_objects_by_position_categorized
from sound_utils import play_beep_sound, play_obstacle_beep_sound
from speech_utils import speak, announce_objects, get_voice_input
from relationship_utils import describe_relationship
from distance_utils import get_object_distance, get_updated_distance
from language_actions import language_actions
import time
import cv2
import numpy as np

beeping_enabled = False
selected_obj_flag = False
pipeline = initialize_camera()
depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame = get_camera_frames(pipeline)

predictor, cfg = initialize_detectron()
print("FRAME WIDTH: ")

frame_width = color_image.shape[1]
print(frame_width)
section_width = frame_width // 3

left_boundary = section_width
right_boundary = section_width * 2


def select_language():
    speak("Speak English to select English or Spanish para español")
    user_input = 'spanish'#get_voice_input()

    if user_input:
        user_input_lower = user_input.lower()

        if user_input_lower == 'english' or user_input_lower == 'inglés':
            return 'en'
        elif user_input_lower == 'spanish' or user_input_lower == 'español':
            return 'es'
        elif user_input_lower == 'quit' or user_input_lower == 'salir':
            exit()

    print("Invalid selection. Please try again.")
    return select_language()  # Recursive call to get a valid language selection


def describe_scene(detected_objects, language, mode='general'):
    if mode == 'detailed':
        get_objects_by_position_categorized(detected_objects, language)
    elif mode == 'general':
        if detected_objects and 'category' in detected_objects[0]:
            categories = set(obj['category'] for obj in detected_objects)
            speak(f"The detected categories in the scene are: {', '.join(categories)}", language)
        else:
            speak("No categories detected in the scene.", language)

try:
    while True:
        detected_objects = []

        # Language selection
        language = select_language()

        # Get camera frames
        depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame = get_camera_frames(pipeline)
        #detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)

        if not beeping_enabled or not selected_obj_flag:
            #detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)

            speak(language_actions[language]['welcome_message'], language)

            # Other code for language selection
            user_input = 'escanear'#get_voice_input()
            print(user_input)

            if user_input == 'scan the scene' or user_input == 'escanear':
                speak("Say 'detailed' for detailed description or 'general' for general description of the scene.", language)

                user_input = 'general'#get_voice_input().lower()
                print(user_input)

                if user_input == 'detailed':
                    speak(language_actions[language]['scan_scene_message'], language)
                    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
                    describe_scene(detected_objects, language, mode='detailed')
                elif user_input == 'general':
                    speak(language_actions[language]['scan_scene_message'], language)
                    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
                    describe_scene(detected_objects, language, mode='general')
                else:
                    speak("Invalid selection. Please try again.", language)



            repeat = True

            while repeat:
                speak(language_actions[language]['repeat_message'], language)
                user_input = get_voice_input().lower()
                print(user_input)
                if user_input == 'repeat':
                    describe_scene(detected_objects, language, mode='detailed')
                elif user_input == "return":
                    repeat = False
                elif user_input == 'quit':
                    break

            if user_input == 'find objects' or user_input == 'buscar objetos':
                detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)

                speak("Select one of the following objects to find where it's placed:", language)

                speak("Say Select to select an object and say Quit to exit Assist Div.", language)
                user_input = get_voice_input()
                print(user_input)

                if user_input == 'select':
                    selected_obj_distance, selected_obj = get_object_distance(detected_objects, depth_image, 1280)
                    selected_obj_flag = True
                elif user_input == 'start beeping':
                    beeping_enabled = True
                elif user_input == 'quit':
                    break

        if selected_obj_flag and not beeping_enabled:
            selected_obj_name = selected_obj["name"]
            if any(obj["name"] == selected_obj_name for obj in detected_objects):
                updated_distance = get_updated_distance(selected_obj, detected_objects)
                speak("Say start beeping to enable beeping, or say quit to exit Assist Div: ", language)
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
                    speak("Object is on your left side", language)
                elif centroid[0] > right_boundary:
                    speak("Object is on your right side", language)
                else:
                    play_beep_sound(updated_distance)
            else:
                speak("Objects is not in the frame", language)

        if user_input == 'quit':
            break

        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

finally:
    pipeline.stop()