import os
import logging
import sys
from collections import deque
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
from shapely.geometry import Polygon, Point
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


def draw_text_lines(frame, lines, position="bottom_left", padding_ratio=0.02, color=(0, 0, 0)):
    if not lines:
        return

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(w, h) / 1000
    thickness = max(1, int(font_scale * 2))
    padding = int(padding_ratio * min(w, h))
    line_gap = max(6, int(padding * 0.6))

    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_w = max(size[0] for size in line_sizes)
    text_h = max(size[1] for size in line_sizes)
    block_h = len(lines) * text_h + (len(lines) - 1) * line_gap

    if position == "bottom_left":
        x = padding
        y = h - padding - block_h + text_h
    elif position == "bottom_right":
        x = w - text_w - padding
        y = h - padding - block_h + text_h
    elif position == "top_left":
        x = padding
        y = padding + text_h
    elif position == "top_right":
        x = w - text_w - padding
        y = padding + text_h
    else:
        raise ValueError("position must be: top_left, top_right, bottom_left, bottom_right")

    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + block_h - text_h + 5),
        (255, 255, 255),
        -1
    )

    for index, line in enumerate(lines):
        line_y = y + index * (text_h + line_gap)
        cv2.putText(frame, line, (x, line_y), font, font_scale, color, thickness, cv2.LINE_AA)


def person_anchor_from_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_zones(point_xy, zones):
    point = Point(point_xy[0], point_xy[1])
    return any(zone.contains(point) or zone.touches(point) for zone in zones)


def zone_index_for_point(point_xy, zones):
    if not zones:
        return 0

    point = Point(point_xy[0], point_xy[1])
    for index, zone in enumerate(zones):
        if zone.contains(point) or zone.touches(point):
            return index
    return None


def signed_line_side(point_xy, line_start, line_end):
    return (
        (line_end[0] - line_start[0]) * (point_xy[1] - line_start[1])
        - (line_end[1] - line_start[1]) * (point_xy[0] - line_start[0])
    )


def crossed_entry_line(previous_point, current_point, entry_line_points, direction):
    if previous_point is None or current_point is None or len(entry_line_points) != 2:
        return False

    line_start = tuple(entry_line_points[0])
    line_end = tuple(entry_line_points[1])
    previous_side = signed_line_side(previous_point, line_start, line_end)
    current_side = signed_line_side(current_point, line_start, line_end)
    epsilon = 1e-6

    if abs(previous_side) <= epsilon:
        previous_side = 0.0
    if abs(current_side) <= epsilon:
        current_side = 0.0

    if direction == "forward":
        return previous_side < 0 <= current_side
    if direction == "reverse":
        return previous_side > 0 >= current_side
    return (
        (previous_side < 0 <= current_side)
        or (previous_side > 0 >= current_side)
    )


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
    active_lanes = stored_config.get('active_lanes', 2)

    endpoint = stored_config.get('endpoint', 'http://127.0.0.1:5000/vsens')
    mode = stored_config.get('mode', 'restAPI')
    auth_type = stored_config.get('auth_type', 'none')

    username = stored_config.get('username', 'admin')
    password = stored_config.get('password', 'Wt@5651%')

    pretrained_model = stored_config.get('pretrained_model', 'models/yolov9t.onnx')

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
    SNAPSHOT_INTERVAL = config2.get('snapshot_interval', 10)
    entry_confirmation_seconds = float(config2.get('entry_confirmation_seconds', 1.5))
    counting_mode = str(config2.get('counting_mode', 'roi')).lower()
    entry_line_points = config2.get('entry_line_points', [])
    entry_line_direction = str(config2.get('entry_line_direction', 'forward')).lower()

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

    zones = [Polygon(np.array(poly, dtype=np.int32)) for poly in polygons]

    if counting_mode not in {"roi", "line_crossing"}:
        print(f"[COUNT] WARNING: unsupported counting_mode={counting_mode}. Falling back to roi mode.")
        counting_mode = "roi"

    if entry_line_direction not in {"forward", "reverse", "any"}:
        print(f"[COUNT] WARNING: unsupported entry_line_direction={entry_line_direction}. Falling back to forward.")
        entry_line_direction = "forward"

    if counting_mode == "line_crossing" and len(entry_line_points) != 2:
        print("[COUNT] WARNING: counting_mode=line_crossing but entry_line_points is invalid. Falling back to roi mode.")
        counting_mode = "roi"

    face_analyzer = FaceAnalyzer(
        detector=RetinaFace(confidence_threshold=face_score),
        age_gender=AgeGender(),
    )

    # Initialize tracker (ByteTrack or SORT)
    tracker_backend = config2.get('tracker', 'bytetrack').lower()
    if tracker_backend == 'sort':
        from utils.tracker_sort import SortTracker
        tracker = SortTracker(
            max_age=max_age,
            min_hits=max_age,
            iou_threshold=1.0 / max_distance_between_points,
        )
        systems_logger.info(f'[TRACKER] Using SORT (max_age={max_age})')
    else:  # bytetrack (default)
        from utils.tracker_bytetrack import ByteTrackTracker
        iou_threshold = 0.25  # ByteTrack default
        tracker = ByteTrackTracker(
            track_thresh=0.5,
            track_buffer=max_age,
            match_thresh=iou_threshold,
        )
        systems_logger.info(f'[TRACKER] Using ByteTrack (max_age={max_age})')

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
            camID = 'SIM_live_test_video'

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
    last_snapshot_time = 0.0
    track_data = {}        # track_id -> {gender, age, confidence, best_conf} accumulated while alive
    prev_track_ids = set() # track_ids active in the previous frame
    counted_entry_times = deque()
    track_first_seen_times = {}
    track_anchor_points = {}
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
            # COCO 24=backpack, 26=handbag
            BAG_CLASSES = {24, 26}
            bag_boxes = []

            for poly in polygons:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(viz_img, [pts], True, (0, 255, 255), 3, lineType=cv2.LINE_AA)
            if counting_mode == "line_crossing":
                p1 = tuple(int(v) for v in entry_line_points[0])
                p2 = tuple(int(v) for v in entry_line_points[1])
                cv2.line(viz_img, p1, p2, (255, 140, 0), 2, cv2.LINE_AA)

            timestamp_str = get_datetime_str()

            for det in detections:
                p_box = det['box']
                p_score = det['confidence']
                class_id = det['class_index']

                # Collect bag detections anywhere in frame
                if class_id in BAG_CLASSES:
                    bag_boxes.append(p_box)
                    if debug_mode:
                        cv2.rectangle(viz_img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 165, 255), 1)
                        cv2.putText(viz_img, 'bag', (p_box[0], p_box[1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                    continue

                if class_id != 0:
                    continue

                anchor_point = person_anchor_from_box(p_box)
                in_zone = point_in_zones(anchor_point, zones) if zones else False
                keep_detection = in_zone or counting_mode == "line_crossing"

                if keep_detection:
                    p_dets.append(p_box)
                    p_confs.append(p_score)

                    box_color = (0, 255, 0) if in_zone else (0, 200, 255)
                    cv2.rectangle(viz_img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), box_color, 1)
                    cv2.circle(viz_img, (int(anchor_point[0]), int(anchor_point[1])), 2, (0, 0, 255), -1, cv2.LINE_AA)

            if len(p_dets) > 0:
                faces = face_analyzer.analyze(im0)
                f_dets = [f.bbox for f in faces]
                f_confs = [f.confidence for f in faces]
                f_genders = [f.sex for f in faces]
                f_ages = [f.age for f in faces]

            min_iou = config2.get('min_iou', 0.5)

            persons = match_boxes(p_dets, f_dets, p_confs, f_confs, f_genders, f_ages, min_iou)

            # Associate bag detections to persons: a bag belongs to a person if its
            # box overlaps or is directly below/beside the person box (expanded by 30%)
            def _person_has_bag(p_box, bags):
                px1, py1, px2, py2 = p_box
                pw = px2 - px1
                ph = py2 - py1
                # Expand person box by 30% in each direction to catch carried bags
                ex1, ey1 = px1 - pw * 0.3, py1 - ph * 0.3
                ex2, ey2 = px2 + pw * 0.3, py2 + ph * 0.3
                for bx1, by1, bx2, by2 in bags:
                    # Check if bag centre falls inside expanded person box
                    bcx = (bx1 + bx2) / 2
                    bcy = (by1 + by2) / 2
                    if ex1 <= bcx <= ex2 and ey1 <= bcy <= ey2:
                        return True
                return False

            frame_detections = []
            for box, conf_val, face in persons:
                has_bag_detected = _person_has_bag(box, bag_boxes)
                data_payload = {
                    "faces":    face,
                    "confidence": float(conf_val),
                    "has_bag":  has_bag_detected,
                }
                if tracker_backend == 'sort':
                    from utils.tracker_base import TrackerDetection
                    frame_detections.append(TrackerDetection(bbox=box, score=float(conf_val), data=data_payload))
                else:  # bytetrack
                    from utils.tracker_base import TrackerDetection
                    frame_detections.append(TrackerDetection(bbox=box, score=float(conf_val), data=data_payload))

                if debug_mode:
                    for f_box, _, _, _, _, gender, age in face:
                        label = f"{gender[0]} - {age}"
                        plot_one_box(f_box, viz_img, label=label, color=(255, 0, 0), line_thickness=1)
                    if has_bag_detected:
                        cv2.putText(viz_img, 'BAG', (int(box[0]), int(box[1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            tracked_objects = tracker.update(detections=frame_detections)

            current_time = time.time()
            current_track_ids = set()

            for obj in tracked_objects:
                track_id = obj.global_id
                current_track_ids.add(track_id)

                estimate = obj.estimate.astype(int)
                bx1, by1 = estimate[0]
                bx2, by2 = estimate[1]
                anchor_point = person_anchor_from_box([bx1, by1, bx2, by2])
                current_zone_id = zone_index_for_point(anchor_point, zones)
                previous_anchor = track_anchor_points.get(track_id)
                track_anchor_points[track_id] = anchor_point

                if track_id not in track_first_seen_times:
                    track_first_seen_times[track_id] = current_time  # first appearance, not yet confirmed
                    track_data[track_id] = {
                        "gender_votes": {"male": 0.0, "female": 0.0, "unknown": 0.0},
                        "age_sum": 0.0,
                        "age_weight": 0.0,
                        "best_conf": 0.0,
                        "has_bag": False,
                        "entry_dt": None,
                        "zone_id": current_zone_id,
                    }

                person_conf = 0.0
                if hasattr(obj.last_detection, "data") and obj.last_detection.data:
                    person_conf = obj.last_detection.data.get("confidence", 0.0)
                    # Bag: latch True — once detected with a bag, stays True
                    if obj.last_detection.data.get("has_bag", False):
                        track_data[track_id]["has_bag"] = True
                    faces = obj.last_detection.data.get("faces", [])
                    if faces:
                        face_conf = float(faces[0][3]) if faces[0][3] else person_conf
                        gender = (faces[0][5] or "unknown").lower()
                        age_val = float(faces[0][6]) if faces[0][6] else None

                        # Accumulate confidence-weighted gender vote
                        if gender in track_data[track_id]["gender_votes"]:
                            track_data[track_id]["gender_votes"][gender] += face_conf
                        else:
                            track_data[track_id]["gender_votes"]["unknown"] += face_conf

                        # Accumulate confidence-weighted age
                        if age_val is not None:
                            track_data[track_id]["age_sum"]    += age_val * face_conf
                            track_data[track_id]["age_weight"] += face_conf

                        track_data[track_id]["best_conf"] = max(
                            track_data[track_id]["best_conf"], person_conf
                        )

                track_age = current_time - track_first_seen_times[track_id]
                is_confirmed = track_id in track_start_times
                if not is_confirmed and track_age >= entry_confirmation_seconds:
                    if counting_mode == "line_crossing":
                        should_confirm = crossed_entry_line(
                            previous_anchor,
                            anchor_point,
                            entry_line_points,
                            entry_line_direction,
                        )
                    else:
                        should_confirm = True

                    if should_confirm:
                        track_start_times[track_id] = current_time
                        counted_entry_times.append(current_time)
                        track_data[track_id]["entry_dt"] = datetime.now()
                        track_data[track_id]["zone_id"] = current_zone_id

                        is_confirmed = True
                        print(f"[COUNT] Confirmed entry track_id={track_id} via {counting_mode} after {track_age:.1f}s")

                if not is_confirmed:
                    continue

                if current_zone_id is not None:
                    track_data[track_id]["zone_id"] = current_zone_id

                track_dur = current_time - track_start_times[track_id]
                time_elapsed = float(obj.age / expect_fps)

                if track_id not in person_loggers:
                    p_logger = PersonDetectionLogger(detection_logger, camID, person_conf)
                    p_logger.person_id = track_id
                    person_loggers[track_id] = p_logger

                    if debug_mode:
                        debug_image_path = os.path.join(snapshot_path, f'snapshot_{no_snapshots}.jpg')
                        cv2.imwrite(debug_image_path, viz_img)
                        no_snapshots += 1
                else:
                    person_loggers[track_id].log_tracking_update(track_id, track_dur, time_elapsed, person_conf)

            # ── Insert died tracks into DB ────────────────────────────────────
            died_ids = prev_track_ids - current_track_ids
            for track_id in died_ids:
                if track_id not in track_start_times:
                    track_first_seen_times.pop(track_id, None)
                    track_anchor_points.pop(track_id, None)
                    track_data.pop(track_id, None)
                    person_loggers.pop(track_id, None)
                    continue
                dwell = current_time - track_start_times[track_id]
                if dwell >= min_elapsed_time and dwell > (max_age * expect_fps):
                    td = track_data.get(track_id, {})

                    # Gender: highest confidence-weighted vote
                    votes = td.get("gender_votes", {})
                    gender = max(votes, key=votes.get) if any(votes.values()) else "unknown"
                    if gender == "unknown" and (votes.get("male", 0) + votes.get("female", 0)) > 0:
                        gender = "male" if votes.get("male", 0) >= votes.get("female", 0) else "female"

                    # Age: confidence-weighted average
                    age = None
                    if td.get("age_weight", 0) > 0:
                        age = round(td["age_sum"] / td["age_weight"], 1)

                    conf     = td.get("best_conf", 0.0)
                    has_bag  = td.get("has_bag", False)
                    entry_dt = td.get("entry_dt")
                    zone_id = td.get("zone_id")

                    if zone_id is None and zones:
                        print(f"[SUPPRESS] Skipped insert for track_id={track_id}, no valid zone assigned.")
                    else:
                        eff_zone = zone_id if zone_id is not None else 0
                        active_zone_dwells = [
                            current_time - track_start_times[active_track_id]
                            for active_track_id in current_track_ids
                            if active_track_id in track_start_times
                            and active_track_id != track_id
                            and track_data.get(active_track_id, {}).get("zone_id", 0 if not zones else None) == eff_zone
                        ]
                        active_zone_max_dwell = max(active_zone_dwells) if active_zone_dwells else 0.0

                        if dwell > active_zone_max_dwell:
                            db.insert_entrance(track_id, gender, age, conf, camID,
                                               dwell_seconds=round(dwell, 2),
                                               entry_time=entry_dt,
                                               has_bag=has_bag)
                            zone_str = f"zone {eff_zone}" if zones else "global zone 0"
                            detection_logger.info(f"[DB] Track died  track_id={track_id} | dwell={dwell:.1f}s "
                                                  f"| {zone_str} | gender={gender} | age={age} | bag={has_bag} | conf={conf:.2f}")
                        else:
                            zone_str = f"zone {eff_zone}" if zones else "global zone 0"
                            detection_logger.info(f"[SUPPRESS] Skipped insert for track_id={track_id} in {zone_str}, "
                                                  f"dwell={dwell:.1f}s <= active_max={active_zone_max_dwell:.1f}s")
                track_start_times.pop(track_id, None)
                track_first_seen_times.pop(track_id, None)
                track_anchor_points.pop(track_id, None)
                track_data.pop(track_id, None)
                person_loggers.pop(track_id, None)
            prev_track_ids = current_track_ids

            # ── Periodic queue-state snapshot ────────────────────────────────
            if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
                active_ids = [obj.global_id for obj in tracked_objects if obj.global_id in track_start_times]
                queue_count = len(active_ids)
                dwells = [current_time - track_start_times[tid]
                          for tid in active_ids if tid in track_start_times]
                avg_dwell = float(sum(dwells) / len(dwells)) if dwells else 0.0
                max_dwell = float(max(dwells)) if dwells else 0.0
                db.log_queue_snapshot(camID, queue_count, avg_dwell, max_dwell, active_lanes)
                last_snapshot_time = current_time

            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / thread_fps) - elapsed_time)
            time.sleep(sleep_time)

            while counted_entry_times and current_time - counted_entry_times[0] > 86400:
                counted_entry_times.popleft()

            count_3_min = sum(1 for ts in counted_entry_times if current_time - ts <= 180)
            count_1_hour = sum(1 for ts in counted_entry_times if current_time - ts <= 3600)
            count_1_day = len(counted_entry_times)

            active_person_count = len([track_id for track_id in current_track_ids if track_id in track_start_times])
            track_label = f"Persons: {active_person_count}"
            write_text(viz_img, track_label, position="top_right")
            draw_text_lines(
                viz_img,
                [
                    f"3 min: {count_3_min}",
                    f"1 hour: {count_1_hour}",
                    f"1 day: {count_1_day}",
                    f"Mode: {'line' if counting_mode == 'line_crossing' else 'roi'}",
                ],
                position="bottom_left",
            )

            end_time = time.time()

            if args.save_demo:
                video_writer.write(viz_img)

            FPS = max(round(1 / (end_time - start_time)), 1)
            write_text(viz_img, f"FPS: {FPS}", position="top_left")

            if int(time.time()) % 10 == 0:
                print(f"INFO: Performance - FPS: {FPS:.1f} | Latency: {(end_time - start_time) * 1000:.0f}ms")

            if args.view_img:
                im_show = cv2.resize(viz_img, (600, 400), interpolation=cv2.INTER_AREA)
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
