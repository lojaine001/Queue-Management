import os
import logging
import sys
from tracemalloc import start
import cv2
import copy
import time
import numpy as np
import requests
import argparse
from datetime import datetime
from threading import Thread, Lock
import json
import norfair
from shapely.geometry import Polygon, Point
from norfair import Detection, Tracker
from uniface.analyzer import FaceAnalyzer
from uniface.detection import RetinaFace, YOLOv8Face
from uniface.attribute import AgeGender
from utils.db_logger import DBLogger

from utils.queue_utils import (
    get_config, create_logger, plot_one_box, iou, match_boxes,
    get_datetime_str, write_text, PersonDetectionLogger
)
from utils.yolov9 import YOLOv9

from uniface import set_cache_dir
set_cache_dir(os.path.join(os.getcwd(), 'models'))

systems_logger = create_logger(name='Systems', log_dir='LOGs', file='systems.log')
detection_logger = create_logger(name='Detections', level='DEBUG', log_dir='LOGs', file='detections.log')


class VideoStream:
    def __init__(self, cap_url: str, cap_loop: bool):
        self.stream = cv2.VideoCapture(cap_url)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.cap_loop = cap_loop
        self.cap_url = cap_url
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                print(get_datetime_str(), " WARNING: no frame grabbed!")
                systems_logger.debug('No frame grabbed!')
                if self.cap_loop:
                    self.stream.release()
                    self.stream.open(self.cap_url)
                    print(get_datetime_str(), " INFO: trying to re-open the video capture")
                    systems_logger.info('trying to re-open the video capture')
                    time.sleep(1)
                    continue
                else:
                    self.stream.release()
                    print(get_datetime_str(), " WARNING: no loop request. Will terminate the capture thread")
                    systems_logger.debug('no loop request. Will terminate the capture thread')
                    break
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
            time.sleep(0.01)

    def read(self):
        self.read_lock.acquire()
        if not isinstance(self.frame, type(None)):
            frame = self.frame.copy()
        else:
            frame = None
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()

    def get_fps(self) -> int:
        return int(self.stream.get(cv2.CAP_PROP_FPS))

    def get_width(self) -> int:
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> int:
        return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


def get_capture_thread(cap_url: str, cap_loop: bool):
    new_thread = VideoStream(cap_url=cap_url, cap_loop=cap_loop)
    return new_thread


class OnnxDetector:
    def __init__(self, device='cpu', model_path='yolov9s.onnx', score_threshold=0.3, conf_threshold=0.4, iou_threshold=0.4):
        self.model = YOLOv9(model_path=model_path,
                            score_threshold=score_threshold,
                            conf_thresold=conf_threshold,
                            iou_threshold=iou_threshold,
                            device=device)

    def __call__(self, image):
        h, w, _ = image.shape
        image = copy.deepcopy(image)
        boxes = self.model.detect(image, h, w)
        return boxes


def commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view_img', default=True, help='display results')
    parser.add_argument('--save_demo', action='store_true', help='save_demos')
    args = parser.parse_args()
    return args


def main():
    args = commandline_args()

    config_name = './config.yml'

    # Ensure LOGs directory exists
    os.makedirs('LOGs', exist_ok=True)

    # ── MODIFICATION 1: Initialize PostgreSQL connection ──
    db = DBLogger()

    stored_config = get_config(config_filepath=config_name)
    config2 = get_config(config_filepath='config2.yml')

    camID = stored_config.get('camID', 'camera1')
    ip_address = stored_config.get('ip_address', '192.168.1.136:1033/axis-media/media.amp')

    endpoint = stored_config.get('endpoint', 'http://127.0.0.1:5000/vsens')
    mode = stored_config.get('mode', 'restAPI')
    auth_type = stored_config.get('auth_type', 'none')

    username = stored_config.get('username', 'admin')
    password = stored_config.get('password', 'Wt@5651%')

    pretrained_model = stored_config.get('pretrained_model', 'models/yolov9s.onnx')

    score = stored_config.get('score', 0.3)
    face_score = stored_config.get('face_score', 0.5)
    min_score = stored_config.get('min_score', 0.3)
    iou_score = stored_config.get('iou_score', 0.3)
    min_delay = stored_config.get('min_delay', 0.0)
    device = args.device

    debug_mode = config2.get('debug_mode', True)
    max_distance_between_points = config2.get('max_distance_between_points', 2)
    max_age = config2.get('max_age', 10)
    expect_fps = config2.get('expect_fps', 3)
    min_elapsed_time = config2.get('min_elapsed_time', 1)

    detector = OnnxDetector(
        device=device,
        model_path=pretrained_model,
        score_threshold=score,
        conf_threshold=min_score,
        iou_threshold=iou_score,
    )

    points = stored_config.get('points', [])
    polygons = []
    polygon = []
    for dot in points:
        if len(dot) == 0:
            if len(polygon) > 0:
                polygons.append(polygon.copy())
            polygon.clear()
        else:
            polygon.append(dot)
    if len(polygon) > 0:
        polygons.append(polygon.copy())
        polygon.clear()

    face_analyzer = FaceAnalyzer(
        detector=RetinaFace(confidence_threshold=face_score),
        age_gender=AgeGender(),
    )

    tracker = Tracker(
        distance_function=iou,
        distance_threshold=max_distance_between_points,
        hit_counter_max=max_age,
        past_detections_length=max_age,
        initialization_delay=min_delay,
    )

    # Set video stream link
    if len(username) > 0 and len(password) > 0 and len(ip_address) > 0:
        if ip_address.startswith('http'):
            link = ip_address
        else:
            link = 'rtsp://' + username + ':' + password + '@' + ip_address
    else:
        if len(ip_address) > 0:
            if any(ext in ip_address.lower() for ext in [".avi", ".m4v", ".mp4"]):
                link = ip_address
            elif ip_address.startswith('http'):
                link = ip_address
            else:
                link = 'rtsp://' + ip_address
        else:
            link = 'live_test_video.mp4'
            camID = 'live_test_video'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('live_stream_demo.mp4', fourcc, 12, (640, 480))

    save_path = 'Detections_JSON'
    os.makedirs(save_path, exist_ok=True)

    snapshot_path = 'debug_snapshots'
    os.makedirs(snapshot_path, exist_ok=True)

    captured_dataset_folder = 'captured_dataset'
    os.makedirs(captured_dataset_folder, exist_ok=True)

    thread = get_capture_thread(cap_url=link, cap_loop=True)
    thread.start()

    thread_fps = 3

    no_dets = len(os.listdir(save_path)) + 1
    no_snapshots = len(os.listdir(snapshot_path)) + 1

    last_det_time = datetime.now()
    triggered_ids = set()

    person_loggers = {}
    track_start_times = {}

    try:
        while True:
            start_time = time.time()
            im0 = thread.read()

            if im0 is None:
                systems_logger.warning('Received empty frame from video stream!')
                time.sleep(0.1)
                continue

            im0 = cv2.resize(im0, (640, 480))
            viz_img = im0.copy()

            detections = detector(im0)

            f_dets = []; f_confs = []; f_genders = []; f_ages = []
            p_dets = []; p_confs = []

            for poly in polygons:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(viz_img, [pts], True, (0, 255, 255), 3, lineType=cv2.LINE_AA)

            timestamp_str = get_datetime_str()

            for det in detections:
                p_box = det['box']
                p_score = det['confidence']
                class_id = det['class_index']

                if class_id != 0:
                    continue

                ROI_Incl = False
                tip_point = None

                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32)
                    tip_offset = min(config2.get('tip_offset', 1.0), 1.0)
                    tip_point = Point(p_box[0] + (p_box[2] - p_box[0]) * tip_offset, p_box[3] - (p_box[3] - p_box[1]) / 5)
                    zone = Polygon(pts)
                    if zone.contains(tip_point):
                        ROI_Incl = True
                        break

                if ROI_Incl:
                    p_dets.append(p_box)
                    p_confs.append(p_score)

                    cv2.rectangle(viz_img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 255, 0), 1)
                    if tip_point:
                        cv2.circle(viz_img, ((int(tip_point.x), int(tip_point.y))), 1, (0, 0, 255), -1, cv2.LINE_AA)

            if len(p_dets) > 0:
                faces = face_analyzer.analyze(im0)
                f_dets = [f.bbox for f in faces]
                f_confs = [f.confidence for f in faces]
                f_genders = [f.sex for f in faces]
                f_ages = [f.age for f in faces]

            min_iou = config2.get('min_iou', 0.5)

            persons = match_boxes(p_dets, f_dets, p_confs, f_confs, f_genders, f_ages, min_iou)

            norfair_detections = []
            for box, conf_val, face in persons:
                if len(face) > 0:
                    data_payload = {
                        "faces": face,
                        "confidence": float(conf_val)
                    }
                    pts = np.array(box).reshape(2, 2)
                    det = Detection(points=pts, data=data_payload)
                    norfair_detections.append(det)

                if debug_mode:
                    for f_box, _, _, _, _, gender, age in face:
                        label = f"{gender[0]} - {age}"
                        plot_one_box(f_box, viz_img, label=label, color=(255, 0, 0), line_thickness=1)

            tracked_objects = tracker.update(detections=norfair_detections)

            current_time = time.time()

            for obj in tracked_objects:
                track_id = obj.global_id

                if track_id not in track_start_times:
                    track_start_times[track_id] = current_time

                time_elapsed = float(obj.age / expect_fps)
                track_dur = current_time - track_start_times[track_id]

                person_conf = 0.0
                if hasattr(obj.last_detection, "data") and obj.last_detection.data:
                    person_conf = obj.last_detection.data.get("confidence", 0.0)

                if track_id not in person_loggers:
                    p_logger = PersonDetectionLogger(detection_logger, camID, person_conf)
                    p_logger.person_id = track_id
                    p_logger.start_block()
                    p_logger.log_tracking_new(track_id, track_dur, time_elapsed, person_conf)
                    person_loggers[track_id] = p_logger

                    # ── MODIFICATION 3: Inject new person into PostgreSQL ──
                    gender = "unknown"
                    age = None
                    if hasattr(obj.last_detection, "data") and obj.last_detection.data:
                        faces = obj.last_detection.data.get("faces", [])
                        if len(faces) > 0:
                            gender = faces[0][5] if faces[0][5] else "unknown"
                            age = float(faces[0][6]) if faces[0][6] else None
                    db.insert_entrance(track_id, gender, age, person_conf, camID)
                    print(f"[DB] Inserted track_id={track_id} | gender={gender} | age={age} | conf={person_conf:.2f}")

                    if debug_mode:
                        debug_image_path = os.path.join(snapshot_path, f'snapshot_{no_snapshots}.jpg')
                        cv2.imwrite(debug_image_path, viz_img)
                        no_snapshots += 1

                elif int(track_dur) % 2 == 0:
                    person_loggers[track_id].log_tracking_update(track_id, track_dur, time_elapsed, person_conf)
                    # Update dwell time in DB every 2 seconds
                    db.update_dwell(track_id, track_dur)

            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / thread_fps) - elapsed_time)
            time.sleep(sleep_time)

            track_label = f"Persons: {len(person_loggers)}"
            write_text(viz_img, track_label, position="top_right")

            end_time = time.time()

            if args.save_demo:
                video_writer.write(viz_img)

            FPS = max(round(1 / (end_time - start_time)), 1)
            write_text(viz_img, f"FPS: {FPS}", position="top_left")

            if int(time.time()) % 10 == 0:
                print(f"INFO: Performance - FPS: {FPS:.1f} | Latency: {(end_time - start_time) * 1000:.0f}ms")

            if args.view_img:
                im_show = cv2.resize(viz_img, (800, 600), interpolation=cv2.INTER_AREA)
                cv2.imshow('Queue Management System', im_show)
                if cv2.waitKey(1) in {ord("q"), ord("Q"), 27}:
                    break

    except Exception as e:
        print(f"{get_datetime_str()} ERROR: An exception occurred - {str(e)}")
        systems_logger.error(f"An exception occurred - {str(e)}", exc_info=True)

    finally:
        db.close()
        if 'video_writer' in locals(): video_writer.release()
        if 'thread' in locals(): thread.stop()
        if args.view_img:
            cv2.destroyAllWindows()
        print(f"{get_datetime_str()} INFO: Cleanup complete. Exiting.")
        os._exit(0)


if __name__ == '__main__':
    main()