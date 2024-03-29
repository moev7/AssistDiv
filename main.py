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
from rapidfuzz import fuzz

beeping_enabled = False
selected_obj_flag = False
pipeline = initialize_camera()
depth_image, color_image = get_camera_frames(pipeline)

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


def process_main_menu(pipeline, predictor, color_image, depth_image, cfg, language, mode):
    speak(language_actions[language]['welcome_message'], language)
    global beeping_enabled
    global selected_obj_flag
    user_input = get_user_input(language)
    print(user_input)
    detected_objects = []
    
    if not beeping_enabled or not selected_obj_flag:
        if fuzz.ratio(user_input, 'scanner') > 50 or fuzz.ratio(user_input, 'escanear') > 50:
            print(fuzz.ratio(user_input, 'scanner'))
            repeat = True
            scene_scan = True
            depth_image, color_image = get_camera_frames(pipeline)
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg,  language, mode='')
            while scene_scan == True:
                speak(language_actions[language]['select_category_message'],language)
                mode = get_user_input(language)
                print(mode)
                if mode == 'general':
                    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='general')
                    scene_scan = False
                elif mode == 'detail' or mode == 'detallada':
                    detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='detail')
                    get_objects_by_position(detected_objects, language)
                    scene_scan = False
                else:
                    speak("Invalid", language)
                    continue
                                           
            while repeat:
                speak(language_actions[language]['repeat_message'], language)
                user_input = get_user_input(language)
                print(user_input)

                if fuzz.ratio(user_input, 'repeat') > 80 or fuzz.ratio(user_input, 'repetir') > 80: 
                    if mode == 'general':
                        detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='general') 
                    elif mode == 'detail' or mode == 'detallada':
                        detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='detail')
                        get_objects_by_position(detected_objects, language)
                elif fuzz.ratio(user_input, 'return') > 80 or fuzz.ratio(user_input, 'regresar') > 80:
                    repeat = False
                    print("Returning to main menu")
                    print(repeat)
                    #process_main_menu(pipeline, predictor, color_image, depth_image, cfg, language, mode)
                elif fuzz.ratio(user_input, 'exit') > 50 or fuzz.ratio(user_input, 'salir') > 50:
                    return 'exit'


        elif fuzz.ratio(user_input, 'find objects') > 50 or fuzz.ratio(user_input, 'buscar objetos') > 50:
            #selected_language = language
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='')
            get_objects_by_position(detected_objects, language)

            speak(language_actions[language]['select'], language)
            user_input = get_voice_input(language)
            print(user_input)
            if fuzz.ratio(user_input, 'select') > 50 or fuzz.ratio(user_input, 'seleccionar') > 50:    
                selected_obj = get_object_distance(detected_objects, depth_image, 640, language)
                selected_obj_flag = True
            elif fuzz.ratio(user_input, 'exit') > 50 or fuzz.ratio(user_input, 'salir') > 50:
                return 'exit'

            if selected_obj_flag :
                selected_obj_name = selected_obj["name"]
                if any(obj["name"] == selected_obj_name for obj in detected_objects):
                    updated_distance = get_updated_distance(selected_obj, detected_objects)
                    speak(language_actions[language]['start_beeping'], language)
                    user_input = get_voice_input(language)
                    if fuzz.ratio(user_input, 'start beeping') > 50 or fuzz.ratio(user_input, 'comenzar pitidos') > 50:
                        beeping_enabled = True
        
        elif fuzz.ratio(user_input, 'exit') > 50 or fuzz.ratio(user_input, 'salir') > 50:
            return 'exit'


        while beeping_enabled and selected_obj:
            # user_input = get_voice_input(language)
            # if fuzz.ratio(user_input, 'exit') > 50 or fuzz.ratio(user_input, 'salir') > 50:
            #     break
            detected_objects = []
            depth_image, color_image = get_camera_frames(pipeline)
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode='')
            updated_distance, centroid = get_updated_distance(selected_obj, detected_objects)
            if centroid != 0:
                # Check if the object's centroid is in the center third of the frame
                if centroid[0] < left_boundary:
                    speak(language_actions[language]['left_side_message'], language)
                elif centroid[0] > right_boundary:
                    speak(language_actions[language]['right_side_message'], language)
                else:
                    play_beep_sound(updated_distance, language)
            else:
                speak(language_actions[language]['not_in_frame_message'], language)  


try:
    speak("english or spanish?")
    language = select_language()


    while True:
        result = process_main_menu(pipeline, predictor, color_image, depth_image, cfg, language, mode='')

        if fuzz.ratio(result, 'exit') > 50 or fuzz.ratio(result, 'salir') > 50:
            break

finally:
    pipeline.stop()
