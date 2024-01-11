from camera_utils import initialize_camera, get_camera_frames
from detectron_utils import initialize_detectron, run_object_detection, visualize_and_get_detected_objects, get_objects_by_position, get_objects_by_position_categorized
from sound_utils import play_beep_sound, play_obstacle_beep_sound
from speech_utils import speak, announce_objects, get_voice_input
from relationship_utils import describe_relationship
from language_actions import language_actions
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

frame_width = color_image.shape[1]
print(frame_width)
section_width = frame_width // 3

left_boundary = section_width
right_boundary = section_width * 2

def select_language():
    while True:
        user_input = get_voice_input()
        print(user_input)
        if user_input == 'english':
            return 'en'
        elif user_input == 'spanish':
            return 'es'
        else:
            speak("Invalid language selection")

def get_user_input(language):
    return get_voice_input(language)

def process_main_menu(user_input,pipeline, predictor, color_image, depth_image, cfg, language):
    global beeping_enabled
    global selected_obj_flag

    detected_objects = []

    if not beeping_enabled or not selected_obj_flag:
        if user_input == 'scan' or user_input == 'escanear':
            repeat = True
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language)
            get_objects_by_position(detected_objects, language)

            while repeat:
                speak(language_actions[language]['repeat_message'], language)
                user_input = get_user_input(language)

                if user_input == 'repeat' or user_input == 'repetir':
                    get_objects_by_position(detected_objects)
                elif user_input == "return" or user_input == 'regresar':
                    repeat = False
                elif user_input == 'exit' or user_input == 'salir':
                    break

        elif user_input == 'find objects' or user_input == 'buscar objetos':
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language)
            speak("Select one of the following objects to find where it's placed:", language)
            speak("Say Select to select an object and say Exit to exit Assist Div.", language)
            user_input = get_user_input(language)

            if user_input == 'select':
                selected_obj_distance, selected_obj = get_object_distance(detected_objects, depth_image, 1280)
                selected_obj_flag = True
            elif user_input == 'start beeping':
                beeping_enabled = True
            elif user_input == 'exit':
                return 'exit'

try:
    speak("english or spanish?")
    language = select_language()
    speak(language_actions[language]['welcome_message'], language)


    while True:
        user_input = get_user_input(language)
        print(user_input)

        if user_input == 'exit' or user_input == 'salir':
            break

        result = process_main_menu(user_input, pipeline, predictor, color_image, depth_image, cfg, language)

        if result == 'exit' or result == 'salir':
            break

finally:
    pipeline.stop()