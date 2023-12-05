from camera_utils import initialize_camera, get_camera_frames
from detectron_utils import initialize_detectron, run_object_detection, visualize_and_get_detected_objects, get_objects_by_position, get_objects_by_position_categorized
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

frame_width = color_image.shape[1]
print(frame_width)
section_width = frame_width // 3

left_boundary = section_width
right_boundary = section_width * 2

# Define language actions
language_actions = {
    'en': {
        'welcome_message': 'Hello, Welcome to Assist Div. Say "Scan the scene" for scene understanding and "Find objects" for object detection. Say "quit" to exit Assist Div.',
        'scan_scene_message': 'Scanning the scene...',
        'find_objects_message': 'Finding objects. Please wait.',
        'beeping_message': 'The selected object is in range. Do you want to start beeping?',
        'left_side_message': 'The selected object is on the left side.',
        'right_side_message': 'The selected object is on the right side.',
        'not_in_frame_message': 'The selected object is not in the frame.',
        'repeat_message': 'Say "repeat" to repeat the object names. Say "return" to do another task. Say "quit" to exit Assist Div.'
    },
    'es': {
        'welcome_message': 'Hola, bienvenido a Assist Div. Diga "Escanear la escena" para comprenderla y "Buscar objetos" para detectar objetos. Diga "salir" para salir.',
        'scan_scene_message': 'Escaneando la escena...',
        'find_objects_message': 'Buscando objetos. Por favor, espera.',
        'beeping_message': 'El objeto seleccionado está dentro del rango. ¿Quieres comenzar a emitir pitidos?',
        'left_side_message': 'El objeto seleccionado está en el lado izquierdo.',
        'right_side_message': 'El objeto seleccionado está en el lado derecho.',
        'not_in_frame_message': 'El objeto seleccionado no está en el marco.',
        'repeat_message': 'Diga "repetir" para repetir los nombres de los objetos. Diga "regresar" para hacer otra tarea. Diga "salir" para salir de la división de asistencia.'
    }
}

try:
    while True:
        detected_objects = []

        # Get camera frames
        depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame = get_camera_frames(pipeline)
        #detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)

        if not beeping_enabled or not selected_obj_flag:
            #detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)

            # Language selection
            speak("Speak English to select English or Spanish para español")
            user_input = 'spanish'#get_voice_input()
            print(user_input)

            if user_input == 'english' or user_input == 'inglés':
                language = 'en'
                speak(language_actions[language]['welcome_message'], language)
            elif user_input == 'spanish' or user_input == 'español':
                language = 'es'
                speak(language_actions[language]['welcome_message'], language)
            elif user_input == 'quit' or user_input == 'salir':
                exit()

            user_input = 'escanear la escena'#get_voice_input()
            print(user_input)

            if user_input == 'scan the scene' or user_input == 'escanear la escena':
                if user_input == 'english' or user_input == 'inglés':
                    language = 'en'
                    speak(language_actions[language]['scan_scene_message'], language)
                elif user_input == 'spanish' or user_input == 'español':
                    language = 'es'
                    speak(language_actions[language]['scan_scene_message'], language)

            repeat = True
            detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)
            get_objects_by_position(detected_objects, language)

            while repeat:
                speak("Say repeat to repeat the object names. Say return to do another task. Say quit to exit Assist Div.", language)
                user_input = get_voice_input()
                print(user_input)
                if user_input == 'repeat':
                    get_objects_by_position_categorized(detected_objects, language)
                elif user_input == "return":
                    repeat = False
                elif user_input == 'quit':
                    break

            if user_input == 'find objects':
                detected_objects = visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg)

                speak("Select one of the following objects to find where it's placed:", language)

                speak("Say Select to select an object and say Quit to exit Assist Div.", language)
                user_input = get_voice_input()
                print(user_input)

                if user_input == 'select':
                    selected_obj_distance, selected_obj = get_object_distance(detected_objects, depth_image, 640)
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
    pipeline.stop
