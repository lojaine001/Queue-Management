import os
import cv2
import time
import math 
import yaml
import logging
import datetime
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from threading import Thread, Lock

from norfair import Detection, Tracker
from shapely.geometry import Polygon,Point

import numpy as np


class PersonDetectionLogger:
    """Creates visual blocks for vehicle detection events to improve log readability"""
    
    def __init__(self, logger, camera_id, confidence):
        self.logger = logger
        self.person_id = None  # Will be set when tracker assigns ID
        self.camera_id = camera_id
        self.initial_confidence = confidence
        self.start_time = datetime.datetime.now()
        self.block_started = False
        self.patterns_logged = []
        self.tracking_section_started = False
        
    def start_block(self):
        """Print detection block header"""
        if self.block_started:
            return
        self.block_started = True
        
        ts = self.start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.logger.info("=" * 80)
        self.logger.info(f">>> PERSON DETECTION #{self.person_id} | {ts}")
        self.logger.info(f">>> Camera: {self.camera_id} | Conf: {self.initial_confidence:.2f}")
        self.logger.info("=" * 80)
        self.logger.info("")
    
    def log_section(self, title):
        """Print section header"""
        self.logger.info(f"  --- {title} ---")
    
    def log_item(self, message, indent=4):
        """Print indented log item"""
        self.logger.info(" " * indent + message)
    
    def log_pattern_accepted(self, pattern_name, conf, conf_thresh, area, area_thresh, ratio, ratio_thresh, inclusion, inclusion_thresh):
        """Log accepted pattern"""
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_item(f"[{ts}] [OK] ACCEPTED | {pattern_name:<10} | Conf:{conf:.2f}>={conf_thresh} | Area:{area:.0f}>={area_thresh} | Ratio:{ratio:.2f}>={ratio_thresh} | Incl:{inclusion:.2f}>{inclusion_thresh}")
        self.patterns_logged.append(pattern_name)
    
    def log_pattern_rejected(self, pattern_name, reasons):
        """Log rejected pattern"""
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_item(f"[{ts}] [FAIL] REJECTED | {pattern_name:<10} | {' | '.join(reasons)}")
    
    def log_tracking_new(self, track_id, track_dur, age, conf):
        """Log new track creation"""
        if not self.tracking_section_started:
            self.log_section("Tracking Updates")
            self.tracking_section_started = True
        
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_item(f"[{ts}] NEW    | ID: {track_id:<4} | Track_Time: {track_dur:>5.2f}s | Track_Age: {age:>5.2f}s | Conf: {conf:.2f}")
    
    def log_tracking_update(self, track_id, track_dur, age, conf, iou=None, is_peak=False):
        """Log tracking update"""
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        iou_str = f" | IoU: {iou:.3f}" if iou is not None else ""
        peak_str = " ⭐" if is_peak else ""
        self.log_item(f"[{ts}] UPDATE | ID: {track_id:<4} | Track_Time: {track_dur:>5.2f}s | Track_Age: {age:>5.2f}s | Conf: {conf:.2f}{iou_str}{peak_str}")
    
    def end_block(self):
        """Print detection block footer"""
        if not self.block_started:
            return
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("")

def get_config(config_filepath: str) -> dict:
    try:
        with open(config_filepath) as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {}

def dump_config(config_filepath: str, confs: dict):
    try:
        with open(config_filepath, 'w') as outfile:
            yaml.dump(confs, outfile, default_flow_style=False)
    except Exception as e:
        print(e)

def send_rest_notification(url, username, password, cam_id, auth_type='basic'):
    """Send detection notification via REST API with Basic or Digest auth."""
    if not url:
        return
        
    payload = {
        "camID": cam_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": "TaxiFound"
    }
    
    auth = None
    if auth_type == 'basic':
        auth = HTTPBasicAuth(username, password)
    elif auth_type == 'digest':
        auth = HTTPDigestAuth(username, password)
        
    try:
        response = requests.post(url, json=payload, auth=auth, timeout=5)
        if response.status_code in [200, 201]:
            logging.info(f"REST Notification Sent: {response.status_code}")
        else:
            logging.error(f"REST Notification Failed: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"REST Notification Error: {str(e)}")

def get_norfair_detections(detections):
    face_norfair_detections = []

    for bbox in detections:
        bbox = np.array(bbox).reshape(2, 2)
        face_norfair_detections.append(Detection(points=bbox))

    return face_norfair_detections

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def cosine_distance(detection, tracked_object):
    dot_product = np.dot(detection.points - tracked_object.estimate)
    norm_A = np.linalg.norm(detection.points)
    norm_B = np.linalg.norm(tracked_object.estimate)
    return 1 - (dot_product / (norm_A * norm_B))

def center(points):
    return [np.mean(np.array(points), axis=0)]

def get_datetime_str():
    now = datetime.datetime.now()
    d1 = now.strftime("%Y-%m-%dT%H:%M:%S")
    return d1

# def IoU_Box(boxA, boxB):

#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     # compute the area of intersection rectangle
#     interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
#     if interArea == 0:
#         return 0

#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
#     boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

#     iou = float(interArea) / float(min(boxAArea,boxBArea))

#     # return the intersection over union value
#     return iou

def IoU_Box(boxA, boxB):
    """Calculate Standard IoU (Aligned with EV App)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    if interArea == 0:
        return 0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = float(interArea) / float(boxAArea + boxBArea - interArea)
    return iou

def iou(detection, tracked_object):
    # Detection points will be box A
    # Tracked objects point will be box B.

    box_a = np.concatenate([detection.points[0], detection.points[1]])
    box_b = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and tracker
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)

    return float(1.0 / iou) if iou else 10000

import logging

import logging

def match_boxes(Person_boxes, Face_boxes, Persons_conf, Faces_conf,
                           Genders, Ages,
                           min_iou=0.5, FaceThreshold=(0.2, 0, 0), debug=True):
    """
    Match detected faces to detected persons using containment ratio, including gender and age.
    
    Args:
        Person_boxes (list): [[x1,y1,x2,y2], ...] for persons
        Face_boxes (list): [[x1,y1,x2,y2], ...] for faces
        Persons_conf (list): confidence scores for persons
        Faces_conf (list): confidence scores for faces
        Genders (list): gender for each face ("M"/"F" etc.)
        Ages (list): age for each face
        min_iou (float): minimum containment ratio (face inside person)
        FaceThreshold (tuple): (confidence_min, area_min, ratio_min)
        debug (bool): enable debug logging
        
    Returns:
        List of tuples: (person_box, person_conf, faces_in_person)
        Each face in faces_in_person is a tuple:
        (face_box, face_conf, area, ratio, containment, gender, age)
    """
    
    persons = []
    trace_logger = logging.getLogger('Detections')

    for person_box, person_conf in zip(Person_boxes, Persons_conf):
        faces_inside = []

        for face_box, face_conf, gender, age in zip(Face_boxes, Faces_conf, Genders, Ages):
            # Compute intersection rectangle
            xA = max(person_box[0], face_box[0])
            yA = max(person_box[1], face_box[1])
            xB = min(person_box[2], face_box[2])
            yB = min(person_box[3], face_box[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            faceArea = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
            containment = interArea / faceArea if faceArea > 0 else 0

            if containment > min_iou:
                w = face_box[2] - face_box[0]
                h = face_box[3] - face_box[1]
                area = w * h
                ratio = round(w / h, 2) if h != 0 else 0

                conf_ok = float(face_conf) >= FaceThreshold[0]
                area_ok = area >= FaceThreshold[1]
                ratio_ok = ratio >= FaceThreshold[2]

                if conf_ok and area_ok and ratio_ok:
                    faces_inside.append((face_box, face_conf, area, ratio, containment, gender, age))
                    if debug:
                        trace_logger.debug(
                            f"[OK] ACCEPTED | Conf:{face_conf:.2f}>={FaceThreshold[0]} | "
                            f"Area:{area:.0f}>={FaceThreshold[1]} | Ratio:{ratio:.2f}>={FaceThreshold[2]} | "
                            f"Incl:{containment:.2f}>{min_iou} | Gender:{gender} Age:{age}"
                        )
                elif debug:
                    reasons = []
                    if not conf_ok: reasons.append(f"Conf:{face_conf:.2f}<{FaceThreshold[0]}")
                    if not area_ok: reasons.append(f"Area:{area:.0f}<{FaceThreshold[1]}")
                    if not ratio_ok: reasons.append(f"Ratio:{ratio:.2f}<{FaceThreshold[2]}")
                    trace_logger.debug(f"[FAIL] REJECTED | {' | '.join(reasons)} | Incl:{containment:.2f}>{min_iou}")
            elif debug and containment > 0:
                trace_logger.debug(f"Face Match | Containment: {containment:.3f} <= {min_iou} (FAIL)")
        
        persons.append((person_box, person_conf, faces_inside))
    
    return persons

def create_logger(name, level = 'DEBUG', log_dir=None, file = None):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    #create log_dir
    #log_dir = os.path.join(log_dir, file)
    os.makedirs(log_dir, exist_ok=True)

    #if no streamhandler present, add one
    if sum([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]) == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s>>>%(message)s', "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(ch)

    #if a file handler is requested, check for existence then add
    if file is not None:
        if sum([isinstance(handler, logging.FileHandler) for handler in logger.handlers]) == 0:
            log_dir = os.path.join(log_dir, file)
            ch = logging.FileHandler(log_dir, mode='a', encoding='utf-8')
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(ch)
        
    return logger

# def write_text(frame, text, fps, x=25, y=10):
#     text = text + str(math.ceil(fps))
#     text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#     # h, w = frame.shape[:2]
#     text_w, text_h = text_size[0]
#     cv2.rectangle(frame, (x, y), (x+text_w+5, y+text_h+30), (255, 255, 255), -1) #x+text_w+5
#     cv2.putText(frame, text, (x+5, y+text_h+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, lineType=cv2.LINE_AA)


def write_text(frame, text, position="top_left", padding_ratio=0.02, color=(0, 0, 0)):
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(w, h) / 1000
    thickness = max(1, int(font_scale * 2))

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    padding = int(padding_ratio * min(w, h))

    if position == "top_left":
        x = padding
        y = padding + text_h

    elif position == "top_right":
        x = w - text_w - padding
        y = padding + text_h

    elif position == "bottom_left":
        x = padding
        y = h - padding

    elif position == "bottom_right":
        x = w - text_w - padding
        y = h - padding

    else:
        raise ValueError("position must be: top_left, top_right, bottom_left, bottom_right")

    # Background rectangle
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + 5),
        (255, 255, 255),
        -1
    )

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def overlay_text_count(frame, text, count):
    text = text + str(count)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size[0]; x, y = (15, 10)
    cv2.rectangle(frame, (x, y), (x+text_w+30, y+text_h+30), (255, 255, 255), -1)
    cv2.putText(frame, text, (x+15, y+text_h+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)

def norfair_pts_and_id(tracked_objects):
    tracked_boxes = []; ids =[]
    for obj in tracked_objects:                   
        points = obj.estimate
        points = points.astype(int)
        pt1, pt2 = points[0, :], points[1, :]
        point = np.array([pt1, pt2]).ravel().tolist()
        tracked_boxes.append(point); ids.append(obj.id)
        return tracked_boxes, ids

def check_inclusion(points, bbox) -> bool:
    Incl = False
    plg = Polygon(points)
    center = Point(bbox[0] + (bbox[2] - bbox[0])/2, bbox[1] + (bbox[3] - bbox[1])/2)
    # Option 0: this check center of the bbox is inside the polygon
    if plg.contains(center):
        Incl = True
    return Incl

def resize_img(img0, img_size, stride=32):
    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img 

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def plot_one_box(box, img, color=None, label=None, line_thickness=2):
    """Draw one bounding box on image. Same signature as reference EV project."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def bg_image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def convert_to_yolov7(image, bbox):
    img_h, img_w, _ = image.shape

    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]

    # Transform the bbox co-ordinates as per the format required by YOLO v7
    center_x = (int(x_min) + int(x_max)) / 2
    center_y = (int(y_min) + int(y_max)) / 2
    width = int(x_max) - int(x_min)
    height = int(y_max) - int(y_min)

    # Normalise the co-ordinates by the dimensions of the image
    center_x /= img_w
    center_y /= img_h
    width /= img_w
    height /= img_h

    # Format the values with three decimal places
    center_x = "{:.3f}".format(center_x)
    center_y = "{:.3f}".format(center_y)
    width = "{:.3f}".format(width)
    height = "{:.3f}".format(height)

    return center_x, center_y, width, height
