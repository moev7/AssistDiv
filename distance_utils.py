
from num2words import num2words
from speech_utils import speak
from relationship_utils import describe_relationship

def get_updated_distance(selected_obj, detected_objects):
    print("Detected objects:", detected_objects)
    print("Selected object:", selected_obj)
    for obj in detected_objects:
        if obj["id"] == selected_obj["id"]:
            return obj["distance"]
    return None

def get_object_distance(detected_objects, frame_width):
    if not detected_objects:
        print("No objects detected.")
        return

    print("Available objects:")
    for i, obj in enumerate(detected_objects):
        print(f"{i + 1}: {obj['name']}")
    print(f"{len(detected_objects) + 1}: Cancel")

    try:
        index = int(input("Enter the number of the object you want the distance for, or choose 'Cancel': ")) - 1
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
            describe_relationship(selected_obj, detected_objects)

        elif index == len(detected_objects):
            print("Operation cancelled.")
        else:
            print("Invalid input. Please enter a number within the available range.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    return selected_obj['distance'], selected_obj

