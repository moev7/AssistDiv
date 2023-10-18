
from num2words import num2words
from speech_utils import speak, get_voice_input
from relationship_utils import describe_relationship, generate_scene_graph, describe_all_relationships, plot_scene_graph
import numpy as np


def get_updated_distance(selected_obj, detected_objects):
    #print("Selected object:", selected_obj)
    for obj in detected_objects:
        if obj["id"] == selected_obj["id"]:
            return obj["distance"]
    return None

def get_object_distance(detected_objects, depth_image ,frame_width):
    if not detected_objects:
        print("No objects detected.")
        speak("No objects detected.")
        return
    
    print("Available objects:")
    speak("Available objects are")
    for i, obj in enumerate(detected_objects):
        print(f"{i + 1}: {obj['name']}")
        speak(f"{i + 1}: {obj['name']}")
    #speak(detected_objects)
    

    try:
        speak("Say the name of the object you want the distance for, or say Cancel")
        object_name = get_voice_input()
        object_name = object_name.lower()
        if object_name == 'cancel':
            print("Operation cancelled.")
            return None, None

        # Find the object by name
        index = next((i for i, obj in enumerate(detected_objects) if obj["name"].lower() == object_name), -1)

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
            #relationships = describe_all_relationships(detected_objects)
            #elationship(selected_obj, detected_objects)
            path_message = find_clear_path(depth_image)
            print("PATH: " + path_message)
            
        elif index == -1:
            print("Object not found. Please say a valid object name.")
            speak("Object not found. Please say a valid object name.")
            
    except ValueError:
        print("Invalid input. Please enter a valid object name.")
        speak("Invalid input. Please enter a valid object name.")
        
    return selected_obj['distance'], selected_obj



def find_clear_path(depth_image):
    height, width = depth_image.shape
    min_distance = 2000  # In millimeters
    threshold = 0.8

    # Define the regions of interest (ROI) representing potential paths
    paths_roi = {
        'left': (0, int(width * 0.25)),
        'front': (int(width * 0.25), int(width * 0.75)),
        'right': (int(width * 0.75), width),
    }

    def is_path_clear(roi):
        roi_start, roi_end = roi
        roi_pixels = depth_image[:, roi_start:roi_end]
        clear_pixels = np.sum(roi_pixels > min_distance)

        return clear_pixels / roi_pixels.size > threshold

    clear_paths = [direction for direction, roi in paths_roi.items() if is_path_clear(roi)]

    if not clear_paths:
        return "No clear path found."
    else:
        return f"It seems like there is a way on your {', '.join(clear_paths)} side."
