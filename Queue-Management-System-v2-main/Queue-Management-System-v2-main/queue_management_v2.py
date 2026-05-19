import os
import logging
import sys
from collections import deque
from tracemalloc import start
import cv2
import time
import numpy as np
import requests
import argparse
from datetime import datetime
from threading import Thread, Lock, Event
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
        boxes = self.model.detect(image, h, w)
        return boxes


class FaceWorker:
    """Runs face analysis in a dedicated background thread.

    Main loop calls submit(frame) — non-blocking, always returns immediately.
    The worker processes the most recent frame at its own pace (~1 fps given
    the cost of face analysis). results() always returns the latest cached output.
    """

    def __init__(self, analyzer):
        self._analyzer  = analyzer
        self._frame     = None
        self._results   = []
        self._lock      = Lock()
        self._trigger   = Event()
        self._total_ms  = 0.0
        self._calls     = 0
        self._thread    = Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Pass the latest frame to the worker. Non-blocking; drops old frame if busy."""
        with self._lock:
            self._frame = frame.copy()
        self._trigger.set()

    def results(self):
        """Return the latest cached face results. Always instant."""
        with self._lock:
            return list(self._results)

    def take_stats(self):
        """Return (avg_ms_per_call, n_calls) since the last take_stats, then reset."""
        with self._lock:
            n  = self._calls
            ms = self._total_ms
            self._calls    = 0
            self._total_ms = 0.0
        return (ms / n if n else 0.0), n

    def _run(self):
        while True:
            self._trigger.wait()
            self._trigger.clear()
            with self._lock:
                frame = self._frame
            if frame is None:
                continue
            t0    = time.time()
            faces = self._analyzer.analyze(frame)
            elapsed_ms = (time.time() - t0) * 1000
            with self._lock:
                self._results   = faces
                self._total_ms += elapsed_ms
                self._calls    += 1


def _line_signed_dist(cx: float, cy: float, p1: list, p2: list) -> float:
    """Return signed perpendicular distance from (cx,cy) to the infinite line through p1→p2.
    Positive = side +1, negative = side -1. Magnitude = pixels from the line."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return 0.0
    return (dy * cx - dx * cy + p2[0] * p1[1] - p2[1] * p1[0]) / length


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
    save_snapshots = config2.get('save_snapshots', True)
    max_distance_between_points = config2.get('max_distance_between_points', 80)
    max_age = config2.get('max_age', 10)
    expect_fps = config2.get('expect_fps', 3)
    min_elapsed_time = config2.get('min_elapsed_time', 1)
    SNAPSHOT_INTERVAL = config2.get('snapshot_interval', 10)
    counting_mode        = config2.get('counting_mode', 'roi').lower()
    entry_line_points    = config2.get('entry_line_points', [])
    entry_line_direction = config2.get('entry_line_direction', 'forward').lower()
    entry_line_width     = config2.get('entry_line_width', 0)
    entry_line_min_disp  = config2.get('entry_line_min_displacement', 10)
    avg_person_width     = config2.get('avg_person_width', 80)
    mask_zones           = config2.get('mask_zones', [])
    active_lanes         = config2.get('active_lanes', 2)

    # Validate line crossing config
    line_p1 = line_p2 = None
    if counting_mode == 'line_crossing':
        if len(entry_line_points) >= 2:
            line_p1 = entry_line_points[0]
            line_p2 = entry_line_points[1]
            print(f"[COUNTER] Line crossing mode | line={line_p1}→{line_p2} | direction={entry_line_direction}")
        else:
            print("[COUNTER] WARNING: counting_mode=line_crossing but entry_line_points not set — falling back to roi")
            counting_mode = 'roi'
    else:
        print("[COUNTER] ROI counting mode")

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
    face_worker = FaceWorker(face_analyzer)

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

    # ── Inference timing accumulators ─────────────────────────────────────────
    _t_detect = _t_track = _t_total = 0.0
    _t_frames_counted = 0
    _t_last_report = time.time()
    _T_REPORT_EVERY = 30.0     # seconds between timing reports
    track_data = {}        # track_id -> {gender, age, confidence, best_conf} accumulated while alive
    prev_track_ids = set() # track_ids active in the previous frame
    counted_entry_times = deque(db.get_today_entry_timestamps())
    track_crossed:  set[int]        = set()   # tracks that already triggered a count
    band_candidate: dict[int, list] = {}      # track_id → [dist1, dist2, ...] for 3-frame confirm

    try:
        while True:
            start_time = time.time()
            im0 = thread.read()

            if im0 is None:
                systems_logger.warning('Received empty frame from video stream!')
                time.sleep(0.1)
                continue

            im0 = cv2.resize(im0, (640, 480))
            for _mz in mask_zones:
                cv2.fillPoly(im0, [np.array(_mz, dtype=np.int32)], (0, 0, 0))
            viz_img = im0.copy()

            _t0 = time.time()
            detections = detector(im0)
            _t_detect += time.time() - _t0

            f_dets = []; f_confs = []; f_genders = []; f_ages = []
            p_dets = []; p_confs = []
            # COCO 24=backpack, 26=handbag
            BAG_CLASSES = {24, 26}
            bag_boxes = []

            for poly in polygons:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(viz_img, [pts], True, (0, 255, 255), 3, lineType=cv2.LINE_AA)

            if counting_mode == 'line_crossing' and line_p1 and line_p2:
                # Middle line — faded grey
                cv2.line(viz_img, tuple(line_p1), tuple(line_p2), (120, 120, 120), 1, cv2.LINE_AA)

                # Band lines (offset perpendicular to the entry line)
                if entry_line_width > 0:
                    _dx = line_p2[0] - line_p1[0]
                    _dy = line_p2[1] - line_p1[1]
                    _len = (_dx * _dx + _dy * _dy) ** 0.5
                    if _len > 0:
                        _hb = entry_line_width / 2
                        _nx = int(_dy / _len * _hb)
                        _ny = int(-_dx / _len * _hb)
                        _p1_top = (line_p1[0] + _nx, line_p1[1] + _ny)
                        _p2_top = (line_p2[0] + _nx, line_p2[1] + _ny)
                        _p1_bot = (line_p1[0] - _nx, line_p1[1] - _ny)
                        _p2_bot = (line_p2[0] - _nx, line_p2[1] - _ny)
                        cv2.line(viz_img, _p1_top, _p2_top, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.line(viz_img, _p1_bot, _p2_bot, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(viz_img, "TOP BAND", (_p1_top[0], _p1_top[1] - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(viz_img, "BOTTOM BAND", (_p1_bot[0], _p1_bot[1] + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            timestamp_str = get_datetime_str()

            for det in detections:
                p_box = det['box']
                p_score = det['confidence']
                class_id = det['class_index']

                # Collect bag detections anywhere in frame
                if class_id in BAG_CLASSES:
                    bag_boxes.append(p_box)
                    continue

                if class_id != 0:
                    continue

                # In line_crossing mode track everyone in frame — the line acts as the gate
                if counting_mode == 'line_crossing':
                    p_dets.append(p_box)
                    p_confs.append(p_score)
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

            # Submit current frame to face worker (non-blocking).
            # Worker runs in background; results() returns latest cached output.
            if len(p_dets) > 0:
                face_worker.submit(im0)
            min_face_size = config2.get('min_face_size', 40)
            faces     = [f for f in face_worker.results()
                         if (f.bbox[2] - f.bbox[0]) >= min_face_size
                         and (f.bbox[3] - f.bbox[1]) >= min_face_size]
            f_dets    = [f.bbox       for f in faces]
            f_confs   = [f.confidence for f in faces]
            f_genders = [f.sex        for f in faces]
            f_ages    = [f.age        for f in faces]

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

            norfair_detections = []
            for box, conf_val, face in persons:
                has_bag_detected = _person_has_bag(box, bag_boxes)
                data_payload = {
                    "faces":    face,
                    "confidence": float(conf_val),
                    "has_bag":  has_bag_detected,
                }
                pts = np.array(box).reshape(2, 2)
                det = Detection(points=pts, data=data_payload)
                norfair_detections.append(det)

                if debug_mode:
                    for f_box, _, _, _, _, gender, age in face:
                        label = f"{gender[0]} - {age}"
                        plot_one_box(f_box, viz_img, label=label, color=(255, 0, 0), line_thickness=1)
                    if has_bag_detected:
                        cv2.putText(viz_img, 'BAG', (int(box[0]), int(box[1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            _t0 = time.time()
            tracked_objects = tracker.update(detections=norfair_detections)
            _t_track += time.time() - _t0

            current_time = time.time()
            current_track_ids = set()


            for obj in tracked_objects:
                track_id = obj.global_id
                current_track_ids.add(track_id)

                # Compute centroid from tracker estimate
                _raw = obj.estimate
                _cx = int((_raw[0][0] + _raw[1][0]) / 2)
                _cy = int((_raw[0][1] + _raw[1][1]) / 2)
                _cy_bottom = int(_raw[1][1])  # feet position — more accurate for line crossing

                if obj.hit_counter >= max_age:
                    _x1, _y1 = int(_raw[0][0]), int(_raw[0][1])
                    _x2, _y2 = int(_raw[1][0]), int(_raw[1][1])
                    _id_color = (0, 200, 0) if track_id in track_crossed else (0, 140, 255)
                    cv2.rectangle(viz_img, (_x1, _y1), (_x2, _y2), _id_color, 2)

                if track_id not in track_start_times:
                    track_start_times[track_id] = current_time
                    # ROI mode: count immediately on first appearance
                    if counting_mode == 'roi':
                        counted_entry_times.append(current_time)
                    track_data[track_id] = {
                        "gender_votes": {"male": 0.0, "female": 0.0, "unknown": 0.0},
                        "age_sum": 0.0,
                        "age_weight": 0.0,
                        "best_conf": 0.0,
                        "has_bag": False,
                        "entry_dt": datetime.now(),
                        "crossing_history": [],  # (cx, cy_bottom) — last 5 positions for smoothing
                    }

                # ── Line crossing check ───────────────────────────────────────
                if counting_mode == 'line_crossing' and line_p1 and line_p2:
                    # Smooth position over recent frames to suppress jitter
                    _hist = track_data[track_id]["crossing_history"]
                    _hist.append((_cx, _cy_bottom))
                    if len(_hist) > 5:
                        _hist.pop(0)
                    _sx = int(sum(h[0] for h in _hist) / len(_hist))
                    _sy = int(sum(h[1] for h in _hist) / len(_hist))

                    signed_dist = _line_signed_dist(_sx, _sy, line_p1, line_p2)
                    systems_logger.debug(
                        f"[LINE] id={track_id} pos=({_sx},{_sy}) feet=({_cx},{_cy_bottom}) "
                        f"dist={signed_dist:.1f}"
                    )

                    # ── 3-frame confirmation + direction validation ────────────
                    if track_id not in track_crossed:
                        _prox = max(entry_line_width * 3, 60)
                        if track_id in band_candidate:
                            # Candidate already started — keep accumulating as person
                            # walks deeper. Only clear if they walk back far outside.
                            if signed_dist > _prox:
                                band_candidate.pop(track_id, None)
                            else:
                                band_candidate[track_id].append(signed_dist)
                                dists = band_candidate[track_id]
                                if len(dists) >= 3:
                                    _was_outside = any(d > 0 for d in dists)
                                    _is_inside = dists[-1] < -(entry_line_width / 2)
                                    _crossed_undetected = (
                                        abs(dists[0]) <= entry_line_width * 3 and
                                        _is_inside and
                                        dists[-1] < dists[0]
                                    )
                                    if (_was_outside and _is_inside) or _crossed_undetected:
                                        _box_w = abs(_raw[1][0] - _raw[0][0])
                                        _n = max(1, round(_box_w / avg_person_width))
                                        systems_logger.info(
                                            f"[COUNTER] 3-frame confirm id={track_id} "
                                            f"dist: {dists[0]:.1f}→{dists[-1]:.1f} "
                                            f"box_w={_box_w:.0f}px → {_n} person(s)"
                                        )
                                        track_crossed.add(track_id)
                                        for _ in range(_n):
                                            counted_entry_times.append(current_time)
                                        band_candidate.pop(track_id, None)
                        elif abs(signed_dist) <= _prox:
                            # Start candidate if close enough to the line
                            if signed_dist >= -(entry_line_width * 3):
                                band_candidate[track_id] = [signed_dist]

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

                track_dur = current_time - track_start_times[track_id]
                time_elapsed = float(obj.age / expect_fps)

                if track_id not in person_loggers:
                    p_logger = PersonDetectionLogger(detection_logger, camID, person_conf)
                    p_logger.person_id = track_id
                    person_loggers[track_id] = p_logger

                    if debug_mode and save_snapshots:
                        debug_image_path = os.path.join(snapshot_path, f'snapshot_{no_snapshots}.jpg')
                        cv2.imwrite(debug_image_path, viz_img)
                        no_snapshots += 1
                else:
                    person_loggers[track_id].log_tracking_update(track_id, track_dur, time_elapsed, person_conf)

            # ── Insert died tracks into DB ────────────────────────────────────
            died_ids = prev_track_ids - current_track_ids
            for track_id in died_ids:
                if track_id not in track_start_times:
                    continue
                dwell = current_time - track_start_times[track_id]
                # In line_crossing mode only insert if person actually crossed the line
                if counting_mode == 'line_crossing' and track_id not in track_crossed:
                    systems_logger.debug(f"[DB] Skipped id={track_id} — never crossed entry line")
                    track_start_times.pop(track_id, None)
                    track_data.pop(track_id, None)
                    person_loggers.pop(track_id, None)
                    band_candidate.pop(track_id, None)
                    continue
                if dwell >= min_elapsed_time:
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
                    db.insert_entrance(track_id, gender, age, conf, camID,
                                       dwell_seconds=round(dwell, 2),
                                       entry_time=entry_dt,
                                       has_bag=has_bag)
                    detection_logger.info(f"[DB] Track died  track_id={track_id} | dwell={dwell:.1f}s "
                                          f"| gender={gender} | age={age} | bag={has_bag} | conf={conf:.2f}")
                track_start_times.pop(track_id, None)
                track_data.pop(track_id, None)
                person_loggers.pop(track_id, None)
                track_crossed.discard(track_id)
                band_candidate.pop(track_id, None)

            prev_track_ids = current_track_ids

            # ── Periodic queue-state snapshot ────────────────────────────────
            if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
                active_ids = [obj.global_id for obj in tracked_objects]
                queue_count = len(active_ids)
                dwells = [current_time - track_start_times[tid]
                          for tid in active_ids if tid in track_start_times]
                avg_dwell = float(sum(dwells) / len(dwells)) if dwells else 0.0
                max_dwell = float(max(dwells)) if dwells else 0.0
                db.log_queue_snapshot(camID, queue_count, avg_dwell, max_dwell, active_lanes)
                last_snapshot_time = current_time

            while counted_entry_times and current_time - counted_entry_times[0] > 86400:
                counted_entry_times.popleft()

            count_3_min = sum(1 for ts in counted_entry_times if current_time - ts <= 180)
            count_1_hour = sum(1 for ts in counted_entry_times if current_time - ts <= 3600)
            count_1_day = len(counted_entry_times)

            write_text(viz_img, f"At Door: {len(current_track_ids)}", position="top_right")
            draw_text_lines(
                viz_img,
                [
                    f"Last 3 min:  {count_3_min}",
                    f"Last hour:   {count_1_hour}",
                    f"Today:       {count_1_day}",
                ],
                position="bottom_left",
            )

            end_time = time.time()
            _t_total += end_time - start_time
            _t_frames_counted += 1

            if args.save_demo:
                video_writer.write(viz_img)

            FPS = max(round(1 / (end_time - start_time)), 1)
            write_text(viz_img, f"FPS: {FPS}", position="top_left")

            # ── Periodic per-stage timing report ──────────────────────────────
            if end_time - _t_last_report >= _T_REPORT_EVERY and _t_frames_counted > 0:
                n = _t_frames_counted
                avg_total     = _t_total  / n * 1000
                avg_detect    = _t_detect / n * 1000
                avg_track     = _t_track  / n * 1000
                avg_face, face_calls = face_worker.take_stats()
                effective_fps = n / _t_total if _t_total > 0 else 0.0
                report = (
                    f"[TIMING] {n} frames over {_T_REPORT_EVERY:.0f}s | "
                    f"effective FPS={effective_fps:.2f} | "
                    f"loop={avg_total:.0f}ms | "
                    f"YOLO={avg_detect:.0f}ms | "
                    f"tracker={avg_track:.0f}ms | "
                    f"face={avg_face:.0f}ms/call ({face_calls} calls, async background thread)"
                )
                print(report)
                systems_logger.info(report)
                _t_detect = _t_track = _t_total = 0.0
                _t_frames_counted = 0
                _t_last_report = end_time

            if args.view_img:
                im_show = cv2.resize(viz_img, (800, 600), interpolation=cv2.INTER_AREA)
                cv2.imshow('Queue Management System', im_show)
                if cv2.waitKey(1) in {ord("q"), ord("Q"), 27}:
                    break

            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / thread_fps) - elapsed_time)
            time.sleep(sleep_time)

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
