from speech_utils import speak

import numpy as np

def describe_relationship2(selected_obj, detected_objects):
    relationships = []
    
    for obj in detected_objects:
        if obj is not selected_obj:
            # Calculate the overlap between the two object masks
            overlap = np.logical_and(obj['mask'], selected_obj['mask'])
            print("OVERLAP: " + str(overlap))
            # Calculate the vertical and horizontal differences
            vertical_diff = obj['mask'].shape[0] - np.sum(overlap, axis=0)
            horizontal_diff = obj['mask'].shape[1] - np.sum(overlap, axis=1)
            
            rel = ""
            # Check the vertical relationship
            if np.all(vertical_diff >= obj['mask'].shape[0] // 2):
                rel += "above and "
            elif np.all(vertical_diff <= -obj['mask'].shape[0] // 2):
                rel += "below and "
            
            # Check the horizontal relationship
            if np.all(horizontal_diff >= obj['mask'].shape[1] // 2):
                rel += "on the left side of "
            elif np.all(horizontal_diff <= -obj['mask'].shape[1] // 2):
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
                    rel += "above and "
                elif obj['centroid'][1] > selected_obj['centroid'][1] + selected_obj['box'][3] * 0.5:
                    rel += "below and "
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
