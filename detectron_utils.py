from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog
import cv2
import numpy as np
from speech_utils import speak


def get_objects_by_position(detected_objects, language='en'):
    left_objects = []
    front_objects = []
    right_objects = []

    image_width = 640 
    
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
    
    def speak_with_language(text, language='en'):
        if language == 'en':
            speak(text, language)
        elif language == 'es':
            translated_text = spanish_translations.get(text, text)
            speak(translated_text, language)

    speak_with_language("There are " + str(len(detected_objects)) + " objects detected in the scene.", language)
        
    if left_objects:
        speak_with_language("Objects on the left side are:", language)
        for obj in left_objects:
            speak_with_language(f"{obj['name']} at {obj['distance']} meters", language)

    if front_objects:
        speak_with_language("\nObjects in front of you are:", language)
        for obj in front_objects:
            speak_with_language(f"{obj['name']} at {obj['distance']} meters", language)

    if right_objects:
        speak_with_language("\nObjects on your right side are:", language)
        for obj in right_objects:
            speak_with_language(f"{obj['name']} at {obj['distance']} meters", language)



def run_object_detection(predictor, color_image):
    outputs = predictor(color_image)
    return outputs


    # Spanish translations for class names
spanish_translations = {
                "person": "persona",
                "bicycle": "bicicleta",
                "car": "coche",
                "motorcycle": "motocicleta",
                "airplane": "avión",
                "bus": "autobús",
                "train": "tren",
                "truck": "camión",
                "boat": "barco",
                "traffic light": "semáforo",
                "fire hydrant": "hidrante",
                "stop sign": "señal de stop",
                "parking meter": "parquímetro",
                "bench": "banco",
                "bird": "pájaro",
                "cat": "gato",
                "dog": "perro",
                "horse": "caballo",
                "sheep": "oveja",
                "cow": "vaca",
                "elephant": "elefante",
                "bear": "oso",
                "zebra": "cebra",
                "giraffe": "jirafa",
                "backpack": "mochila",
                "umbrella": "paraguas",
                "handbag": "bolso",
                "tie": "corbata",
                "suitcase": "maleta",
                "frisbee": "frisbee",
                "skis": "esquís",
                "snowboard": "tabla de snowboard",
                "sports ball": "pelota de deporte",
                "kite": "cometa",
                "baseball bat": "bate de béisbol",
                "baseball glove": "guante de béisbol",
                "skateboard": "monopatín",
                "surfboard": "tabla de surf",
                "tennis racket": "raqueta de tenis",
                "bottle": "botella",
                "wine glass": "copa de vino",
                "cup": "taza",
                "fork": "tenedor",
                "knife": "cuchillo",
                "spoon": "cuchara",
                "bowl": "bol",
                "banana": "plátano",
                "apple": "manzana",
                "sandwich": "sándwich",
                "orange": "naranja",
                "broccoli": "brócoli",
                "carrot": "zanahoria",
                "hot dog": "perro caliente",
                "pizza": "pizza",
                "donut": "donut",
                "cake": "pastel",
                "chair": "silla",
                "couch": "sofá",
                "potted plant": "planta en maceta",
                "bed": "cama",
                "dining table": "mesa de comedor",
                "toilet": "inodoro",
                "tv": "televisión",
                "laptop": "portátil",
                "mouse": "ratón",
                "remote": "mando a distancia",
                "keyboard": "teclado",
                "cell phone": "teléfono móvil",
                "microwave": "microondas",
                "oven": "horno",
                "toaster": "tostadora",
                "sink": "fregadero",
                "refrigerator": "refrigerador",
                "book": "libro",
                "clock": "reloj",
                "vase": "jarrón",
                "scissors": "tijeras",
                "teddy bear": "oso de peluche",
                "hair drier": "secador de pelo",
                "toothbrush": "cepillo de dientes",
            }


def visualize_and_get_detected_objects(predictor, color_image, depth_image, cfg):
    outputs = run_object_detection(predictor, color_image)
    v = Visualizer(color_image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    detected_objects = []

# ...

# Choose the appropriate language for the text based on the selected language
    language = 'es'  # Change the language to 'es' for Spanish

# ...

    # Create a black image to display distances
    distance_image = np.zeros((color_image.shape[0], 200, 3), dtype=np.uint8)  # width of 200px

    text_position_start = 30  # y position to start writing text

    for i, (class_idx, mask, box) in enumerate(zip(outputs["instances"].pred_classes.to("cpu"), outputs["instances"].pred_masks.to("cpu"), outputs["instances"].pred_boxes.tensor)):
        instance_mask = mask.cpu().numpy().astype(np.uint8)
        masked_depth_image = depth_image.copy() * instance_mask
        distances = masked_depth_image[np.nonzero(masked_depth_image)]

        if len(distances) > 0:
            mean_distance = np.mean(distances) / 1000  # assuming the depth is in millimeters
        else:
            mean_distance = 0

        distance_text = f"{mean_distance:.2f} m"
        class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_idx]

        y, x = np.nonzero(instance_mask)
        x_center, y_center = int(np.mean(x)), int(np.mean(y))

        centroid = (x_center, y_center)

        detected_objects.append({
            "id": i,
            "name": class_name,
            "distance": mean_distance,
            "centroid": centroid,
            "box": box,
            "mask": instance_mask
        })

        # Inside the loop, set the text for each object based on the selected language
        if language == 'en':
            text = f"{class_name}: {distance_text}"
        elif language == 'es':
            translated_class_name = spanish_translations.get(class_name, class_name)
            print(f"Original class name: {class_name}, Translated class name: {translated_class_name}")
            text = f"{translated_class_name}: {distance_text}"

        cv2.putText(distance_image, text, (10, text_position_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_position_start += 30  # move down by 30px for the next text

    # Concatenate the images horizontally after the loop
    output_image = out.get_image()[:, :, ::-1]
    images_concat = np.hstack((distance_image, output_image))

    # ...

    cv2.imshow("Distances and Instance Segmentation", images_concat)
    cv2.waitKey(1)  # Change this value as needed or use cv2.waitKey(0) to wait until any key is pressed
    #cv2.destroyAllWindows()

    # ...

    cv2.waitKey(1)  # Change this value as needed or use cv2.waitKey(0) to wait until any key is pressed
    #cv2.destroyAllWindows()

    # Sort detected_objects based on the x-coordinate of centroids
    detected_objects.sort(key=lambda obj: obj['centroid'][0])

    return detected_objects


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