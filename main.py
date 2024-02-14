from camera_utils import initialize_camera, get_camera_frames
from detectron_utils import initialize_detectron, run_object_detection, visualize_and_get_detected_objects, get_objects_by_position
from sound_utils import play_beep_sound, play_obstacle_beep_sound
from speech_utils import speak, announce_objects, get_voice_input, play_beep
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


def process_main_menu(user_input, pipeline, predictor, color_image, depth_image, cfg, language, mode):
    global beeping_enabled
    global selected_obj_flag

    detected_objects = []

    if not beeping_enabled or not selected_obj_flag:
        if user_input == 'scan' or user_input == 'escanear':
            repeat = True
            scene_scan = True
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg,  language, mode='')
            while scene_scan == True:
                speak(language_actions[language]['select_category_message'],language)
                mode = get_user_input(language)
                print(mode)
                if mode == 'general':
                    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='general')
                    scene_scan = False
                elif mode == 'detail' or mode == 'detalle':
                    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='detail')
                    scene_scan = False
                else:
                    speak("Invalid", language)
                    continue
                                           
            while repeat:
                speak(language_actions[language]['repeat_message'], language)
                user_input = get_user_input(language)

                if user_input == 'repeat' or user_input == 'repetir':
                    get_objects_by_position(detected_objects, language)
                elif user_input == "return" or user_input == 'regresar':
                    return
                elif user_input == 'exit' or user_input == 'salir':
                    return 'exit'


        elif user_input == 'find objects' or user_input == 'buscar objetos':
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='')
            #get_objects_by_position(detected_objects, language)

            speak(language_actions[language]['select'], language)
            user_input = get_voice_input(language)
            print(user_input)
            if user_input == 'select' or user_input == 'seleccionar':
                selected_obj_distance, selected_obj = get_object_distance(detected_objects, depth_image, 1280)
                selected_obj_flag = True
            elif user_input == 'quit' or user_input == 'salir':
                return 'exit'

        if selected_obj_flag :
            selected_obj_name = selected_obj["name"]
            if any(obj["name"] == selected_obj_name for obj in detected_objects):
                updated_distance = get_updated_distance(selected_obj, detected_objects)
                speak(language_actions[language]['start_beeping'], language)
                user_input = get_voice_input(language)
                if user_input == 'start beeping':
                    beeping_enabled = True


        while beeping_enabled and selected_obj:
            detected_objects = []
            depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame = get_camera_frames(pipeline)
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='')
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

try:
    speak("english or spanish?")
    language = select_language()
    speak(language_actions[language]['welcome_message'], language)


    while True:
        user_input = get_user_input(language)
        print(user_input)

        if user_input == 'exit' or user_input == 'salir':
            break

        result = process_main_menu(user_input, pipeline, predictor, color_image, depth_image, cfg, language, mode='')

        if result == 'exit' or result == 'salir':
            break

finally:
    pipeline.stop()
