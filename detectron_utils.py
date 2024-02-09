from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog
from translations_en import LANGUAGE_EN
from translations_es import LANGUAGE_ES
import cv2
import numpy as np
from speech_utils import speak

def translate_object_name(object_name, language):
    translated_category = None
    translated_name = None

    if language == 'es':
        for category, objects in LANGUAGE_ES.items():
            if object_name in objects:
                translated_category = category
                translated_name = object_name
                break
    else:
        for category, objects in LANGUAGE_EN.items():
            if object_name in objects:
                translated_category = category
                translated_name = object_name
                break

    return translated_category, translated_name



# def translate_object_name(detected_object, language):
#     if language == "en":
#         translation_dict = LANGUAGE_EN 
#     elif language == "es":
#         translation_dict = LANGUAGE_ES

#     for category, objects in translation_dict.items():
#         if detected_object['name'] in objects:
#             return category, detected_object['name']
    

def get_objects_by_position(detected_objects, language):
    left_objects = []
    front_objects = []
    right_objects = []

    image_width = 1280 
    
    left_threshold = image_width // 3
    right_threshold = (2 * image_width) // 3
    
    for obj in detected_objects:
        # Get all the x coordinates of the pixels occupied by the object
        x_pixels = np.where(obj['mask'])[1]
        left_pixels = np.sum(x_pixels < left_threshold)
        right_pixels = np.sum(x_pixels > right_threshold)
        front_pixels = np.sum((x_pixels >= left_threshold) & (x_pixels <= right_threshold))

        # Choose the section where the object has the most pixels
        if left_pixels > right_pixels and left_pixels > front_pixels:
            left_objects.append(obj)
        elif right_pixels > left_pixels and right_pixels > front_pixels:
            right_objects.append(obj)
        else:
            front_objects.append(obj)

    # Round the distance to one decimal place for each object
    for obj in left_objects:
        obj["distance"] = round(obj["distance"], 1)
    for obj in front_objects:
        obj["distance"] = round(obj["distance"], 1)
    for obj in right_objects:
        obj["distance"] = round(obj["distance"], 1)
    
    # Speak the results with language-specific configurations
    if language == "en":
        speak(f"There are {len(detected_objects)} objects detected in the scene.", language)

        if left_objects:
            speak("Objects on the left side are:", language)
            for obj in left_objects:
                speak(f"{obj['name']} at {obj['distance']} meters", language)

        if front_objects:
            speak("\nObjects in front of you are:", language)
            for obj in front_objects:
                speak(f"{obj['name']} at {obj['distance']} meters", language)

        if right_objects:
            speak("\nObjects on your right side are:", language)
            for obj in right_objects:
                speak(f"{obj['name']} at {obj['distance']} meters", language)

    elif language == "es":
        speak(f"Hay {len(detected_objects)} objetos detectados en la escena.", language)

        if left_objects:
            speak("Objetos a tu izquierda son:", language)
            for obj in left_objects:
                speak(f"{obj['name']} a {obj['distance']} metros", language)

        if front_objects:
            speak("\nObjetos frente a ti son:", language)
            for obj in front_objects:
                speak(f"{obj['name']} a {obj['distance']} metros", language)

        if right_objects:
            speak("\nObjetos a tu derecha son:", language)
            for obj in right_objects:
                speak(f"{obj['name']} a {obj['distance']} metros", language)


def run_object_detection(predictor, color_image):
    outputs = predictor(color_image)
    return outputs



def visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg, language, mode=''):
    outputs = run_object_detection(predictor, color_image)
    v = Visualizer(color_image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    detected_objects = []

    # Create a black image to display distances
    distance_image = np.zeros((color_image.shape[0], 400, 3), dtype=np.uint8)
    text_position_start = 30  # y position to start writing text

    for i, (class_idx, mask, box) in enumerate(zip(outputs["instances"].pred_classes.to("cpu"), outputs["instances"].pred_masks.to("cpu"), outputs["instances"].pred_boxes.tensor)):
        instance_mask = mask.cpu().numpy().astype(np.uint8)
        masked_depth_image = depth_image.copy() * instance_mask
        distances = masked_depth_image[np.nonzero(masked_depth_image)]

        if len(distances) > 0:
            mean_distance = np.mean(distances) / 1000  # assuming the depth is in millimeters
        else:
            mean_distance = 0

        # Get the category and translated name
        # category, translated_name = translate_object_name (MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_idx], language)
        # class_name = LANGUAGE[language].get(class_name, class_name)
            


        object_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_idx]
        category, translated_name = translate_object_name(object_name, language)
        if translated_name is None:
            translated_name = "Unknown"


        # # Use translate_object_name to get the category and translated name
        # category, translated_name = translate_object_name({"name": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_idx]}, language)
        # if translated_name is None:
        #     translated_name = "Unknown"

        #enumerate the detected objects
        display_text = f"{category}: {translated_name}"
        distance_text = f"{mean_distance:.2f} m"


        # display_text = f"{category}: {translated_name}"
        # distance_text = f"{mean_distance:.2f} m"

        y, x = np.nonzero(instance_mask)
        x_center, y_center = int(np.mean(x)), int(np.mean(y))

        centroid = (x_center, y_center)

        detected_objects.append({
            "id": i,
            "name": translated_name,
            "distance": mean_distance,
            "centroid": centroid,
            "box": box,
            "mask": instance_mask
        })

        if mode == 'detail':
            text = f"{translated_name}: {distance_text}"
            speak(text, language)
            cv2.putText(distance_image, text, (10, text_position_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_position_start += 30  # move down by 30px for next text
        elif mode == 'general':
            distinct_objects = set(category)  # Get distinct objects from the category list
            text = f"{(category)}"
            speak(text, language)
            cv2.putText(distance_image, text, (10, text_position_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_position_start += 30
        else:
            print("Invalid mode")
            return None

        # text = f"{display_text}: {distance_text}"
        # cv2.putText(distance_image, text, (10, text_position_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # text_position_start += 30  # move down by 30px for next text

    output_image = out.get_image()[:, :, ::-1]

    # Concatenate images horizontally
    images_concat = np.hstack((distance_image, output_image))
    cv2.imshow("Distances and Instance Segmentation", images_concat)
    cv2.waitKey(0)

    # Sort detected_objects based on the x-coordinate of centroids
    detected_objects.sort(key=lambda obj: obj['centroid'][0])

    return detected_objects
    # cv2.destroyAllWindows()




def initialize_detectron():
    cfg = get_cfg()
    #load model config and pretrained model
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.MODEL.DEVICE = "cuda" #cpu or cuda
    predictor = DefaultPredictor(cfg)
    return predictor, cfg



