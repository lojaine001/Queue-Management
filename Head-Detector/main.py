import os
import sys
from typing import List, Optional, Union
from pathlib import Path
from importlib import metadata
import importlib.util

# ── Windows: keep DLL search-path handles alive for ORT/OpenVINO ─────────────
# Python 3.8+ does not search arbitrary site-packages subdirs for dependent
# DLLs, and add_dll_directory() only remains active while its return handle is
# still referenced.
_WINDOWS_DLL_DIR_HANDLES = []


def _register_windows_dll_dir(path: Optional[Union[str, os.PathLike]]) -> None:
    if sys.platform != "win32" or not path:
        return

    dll_dir = Path(path)
    if not dll_dir.is_dir():
        return

    try:
        handle = os.add_dll_directory(str(dll_dir))
    except (AttributeError, FileNotFoundError, OSError):
        return

    _WINDOWS_DLL_DIR_HANDLES.append(handle)


def _register_windows_runtime_dirs() -> None:
    if sys.platform != "win32":
        return

    # Python runtime DLLs
    _register_windows_dll_dir(Path(sys.executable).resolve().parent)
    _register_windows_dll_dir(Path(sys.base_prefix) if sys.base_prefix else None)

    # ONNX Runtime provider DLLs live in onnxruntime/capi
    ort_spec = importlib.util.find_spec("onnxruntime")
    if ort_spec and ort_spec.origin:
        _register_windows_dll_dir(Path(ort_spec.origin).resolve().parent / "capi")

    # OpenVINO runtime DLLs live in openvino/libs
    ov_spec = importlib.util.find_spec("openvino")
    if ov_spec and ov_spec.origin:
        _register_windows_dll_dir(Path(ov_spec.origin).resolve().parent / "libs")


_register_windows_runtime_dirs()
# ──────────────────────────────────────────────────────────────────────────────

import logging
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
# from flask import Flask, request, jsonify


from utils.queue_utils import get_config, create_logger, plot_one_box, iou, write_text
from utils.queue_utils import get_datetime_str, PersonDetectionLogger
from utils.db_logger import DBLogger
from utils.equipment_classifier import EquipmentClassifier, EQUIPMENT_COLOURS, EQUIPMENT_NONE
from utils.owlv2_detector import OWLv2Detector

from utils.yolo import Color, is_package_installed, YOLOv9


def _get_installed_version(package_name: str) -> Optional[str]:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _find_dist_info_dirs(distribution_name: str) -> List[str]:
    matches: List[str] = []
    pattern = f"{distribution_name.replace('-', '_')}-*.dist-info"
    for entry in sys.path:
        if not entry:
            continue
        try:
            path = Path(entry)
        except OSError:
            continue
        if not path.is_dir():
            continue
        matches.extend(sorted(item.name for item in path.glob(pattern)))
    return matches


def _print_norfair_dependency_help(exc: Exception) -> None:
    numpy_version = _get_installed_version('numpy')
    scipy_version = _get_installed_version('scipy')
    norfair_version = _get_installed_version('norfair')
    numpy_dist_infos = _find_dist_info_dirs('numpy')

    print(Color.RED('ERROR: Norfair failed to import because the NumPy/SciPy environment is inconsistent.'))
    print(Color.YELLOW(f'  numpy   : {numpy_version or "not installed"}'))
    print(Color.YELLOW(f'  scipy   : {scipy_version or "not installed"}'))
    print(Color.YELLOW(f'  norfair : {norfair_version or "not installed"}'))
    if len(numpy_dist_infos) > 1:
        print(Color.YELLOW(f'  detected multiple NumPy installs: {numpy_dist_infos}'))
    print(Color.YELLOW(f'  original import error: {exc}'))
    print(Color.CYAN('Norfair 2.3.0 requires NumPy < 2.0, but this environment appears to contain conflicting NumPy files.'))
    print(Color.CYAN('Repair the virtualenv with:'))
    print('  pip uninstall -y numpy scipy norfair')
    print('  pip install --no-cache-dir --force-reinstall "numpy>=1.23,<2.0" "scipy>=1.10,<1.16" "norfair==2.3.0"')
    print(Color.CYAN('If the problem persists, recreate the virtualenv and reinstall from requirements.txt.'))


def _normalize_openvino_device(device: str) -> str:
    normalized = device.upper()
    # Newer OpenVINO/ORT builds use NPU instead of MYRIAD.
    if normalized == 'MYRIAD':
        return 'NPU'
    return normalized


def _build_openvino_provider_options(device: str, precision: str) -> dict[str, str]:
    normalized_device = _normalize_openvino_device(device)
    normalized_precision = precision.upper()

    # OpenVINO CPU execution is the safe default for FP32. Asking for FP16 on
    # CPU can cause ORT to silently fall back to the plain CPU provider.
    if normalized_device == 'CPU' and normalized_precision != 'FP32':
        print(Color.YELLOW(
            f'[ORT] WARNING: OpenVINO {normalized_device} with precision {normalized_precision} '
            'is not reliable here; switching to FP32.'
        ))
        normalized_precision = 'FP32'

    ort_openvino_version = _get_installed_version('onnxruntime-openvino') or _get_installed_version('onnxruntime')
    ort_major_minor = tuple(int(part) for part in ort_openvino_version.split('.')[:2]) if ort_openvino_version else None

    # Older OpenVINO EP builds expect precision encoded in device_type, e.g.
    # CPU_FP32 / GPU_FP16, and reject a separate "precision" option.
    if ort_major_minor and ort_major_minor < (1, 17):
        return {
            'device_type': f'{normalized_device}_{normalized_precision}',
            'cache_dir': '.',
        }

    return {
        'device_type': normalized_device,
        'precision': normalized_precision,
        'cache_dir': '.',
    }


def _warn_if_openvino_runtime_mismatch() -> None:
    ort_openvino_version = _get_installed_version('onnxruntime-openvino')
    openvino_version = _get_installed_version('openvino')
    if not ort_openvino_version or not openvino_version:
        return

    # Official ONNX Runtime OpenVINO EP docs list these compatibility pairs.
    compatible_openvino_by_ort = {
        '1.21': '2025.0',
        '1.22': '2025.1',
        '1.23': '2025.3',
    }

    ort_key = '.'.join(ort_openvino_version.split('.')[:2])
    openvino_key = '.'.join(openvino_version.split('.')[:2])
    expected_openvino = compatible_openvino_by_ort.get(ort_key)
    if expected_openvino and openvino_key != expected_openvino:
        print(Color.YELLOW(
            '[ORT] WARNING: Detected a likely OpenVINO version mismatch: '
            f'onnxruntime-openvino {ort_openvino_version} with openvino {openvino_version}.'
        ))
        print(Color.YELLOW(
            f'[ORT] WARNING: For ORT {ort_key}.x, use OpenVINO {expected_openvino}.x to avoid silent fallback to CPU.'
        ))


_norfair_available = False
try:
    import norfair
    from norfair import Detection, Tracker
    _norfair_available = True
except Exception as exc:
    _print_norfair_dependency_help(exc)
    # Don't exit here — may be using SORT or ByteTrack instead


systems_logger = create_logger(name='Systems', log_dir='LOGs', file='systems.log')
detection_logger = create_logger(name='Detections', level='DEBUG', log_dir='LOGs', file='detections.log')



class VideoStream :
    def __init__(self, cap_url:str, cap_loop: bool) :
        self.stream = cv2.VideoCapture(cap_url)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.cap_loop = cap_loop
        self.cap_url = cap_url
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                print(get_datetime_str(), " WARNING: no frame grabbed!")
                systems_logger.debug('No frame grabbed!')
                if self.cap_loop:
                    self.stream.release()
                    self.stream.open(self.cap_url)
                    print(get_datetime_str(), " INFO: trying to re-open the video capture")
                    systems_logger.info('trying to re-open the video capture')
                    time.sleep(1) # Add sleep to prevent tight loop if camera/file is missing
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

    def read(self) :
        self.read_lock.acquire()
        if not isinstance(self.frame, type(None)):
            frame = self.frame.copy()
        else:
            frame = None
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        if self.thread.is_alive():
            self.thread.join()

    def get_fps(self) -> int:
        return int(self.stream.get(cv2.CAP_PROP_FPS))

    def get_width(self) -> int:
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> int:
        return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

def get_capture_thread(cap_url:str, cap_loop: bool):
    new_thread = VideoStream(cap_url=cap_url, cap_loop=cap_loop)
    return new_thread


def detect(image, model, disable_HI=False):
    debug_image = copy.deepcopy(image)

    boxes = model(
        image=debug_image,
        disable_headpose_identification_mode=disable_HI
    )

    return boxes, debug_image

def commandline_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pretrained_weights', type=str, default='yolov9-t.pt', help='model.pt path(s)')
    parser.add_argument('--inference_type', type=str, choices=['fp32','fp16', 'int8'], default='fp32', help='Inference type. Default: fp16')
    parser.add_argument('--execution_provider', type=str, choices=['cpu', 'cuda', 'tensorrt', 'openvino'], default='openvino', help='Execution provider for ONNXRuntime. Default: openvino.')
    parser.add_argument('--custom_weights', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--out', type=str, help='path to save demo') # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument('--view-img', dest='view_img', action='store_true', help='display results (default)')
    view_group.add_argument('--no-view-img', dest='view_img', action='store_false', help='disable display window')
    parser.set_defaults(view_img=True)
    # parser.add_argument('--save_demo', action='store_true', help='save_demos')
    args = parser.parse_args()
    return args

def main():
    args = commandline_args()

    # Loading config
    config_name = './config.yml'

    # Ensure LOGs directory exists
    os.makedirs('LOGs', exist_ok=True)

    stored_config = get_config(config_filepath=config_name)
    config2 = get_config(config_filepath='config2.yml')

    camID = stored_config.get('camID', 'camera1')
    ip_address = stored_config.get('ip_address', '192.168.1.136:1033/axis-media/media.amp')
    username = stored_config.get('username', 'admin')
    password = stored_config.get('password', 'Wt@5651%')

    if not is_package_installed('onnxruntime'):
        print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
        sys.exit(0)

    import onnxruntime as _ort
    _warn_if_openvino_runtime_mismatch()
    _available = _ort.get_available_providers()
    print(Color.CYAN(f'[ORT] Available execution providers: {_available}'))
    print(Color.CYAN(f'[ORT] Requested provider      : {args.execution_provider}'))
    if args.execution_provider == 'openvino' and 'OpenVINOExecutionProvider' not in _available:
        print(Color.YELLOW('[ORT] WARNING: OpenVINOExecutionProvider not available — falling back to CPU'))
    elif args.execution_provider == 'cuda' and 'CUDAExecutionProvider' not in _available:
        print(Color.YELLOW('[ORT] WARNING: CUDAExecutionProvider not available — falling back to CPU'))
    elif args.execution_provider == 'tensorrt' and 'TensorrtExecutionProvider' not in _available:
        print(Color.YELLOW('[ORT] WARNING: TensorrtExecutionProvider not available — falling back to CPU'))

    custom_model = stored_config.get('custom_model', 'models/yolov9_c_discrete_headpose_post_0100_1x3x480x640.onnx')
    score_threshold = stored_config.get('score', 0.3)
    min_delay = stored_config.get('min_delay', 0.0)
    debug_mode = config2.get('debug_mode', True)
    show_ids   = config2.get('show_ids', False)
    max_distance_between_points = config2.get('max_distance_between_points', 2)
    max_age = config2.get('max_age', 10)
    expect_fps = config2.get('expect_fps', 3)
    snapshot_class_only = config2.get('snapshot_classes', [])
    openvino_device     = config2.get('openvino_device', 'CPU').upper()     # CPU / GPU / MYRIAD
    openvino_precision  = config2.get('openvino_precision', 'FP32').upper() # FP32 / FP16
    save_one = config2.get('save_one', True)
    save_snapshots = config2.get('save_snapshots', False)
    show_rejected_track_overlay = config2.get('show_rejected_track_overlay', False)
    SNAPSHOT_INTERVAL = config2.get('snapshot_interval', 10)
    equipment_model_path  = config2.get('equipment_model', '')
    equipment_score_thresh = float(config2.get('equipment_score_thresh', 0.40))
    equipment_heuristic   = bool(config2.get('equipment_heuristic', False))
    show_equipment_labels = bool(config2.get('show_equipment_labels', True))
    owlv2_enabled         = bool(config2.get('owlv2_enabled', False))
    owlv2_interval_frames = int(config2.get('owlv2_interval_frames', 30))
    owlv2_score_thresh    = float(config2.get('owlv2_score_thresh', 0.15))
    disable_HI = stored_config.get('disable_headpose_identification_mode', False)
    tracker_backend = config2.get('tracker', 'norfair').lower()

    # h, w = im0.shape[:2]

    # points = [[0,0], [w,0], [w,h], [0,h]]
    # polygons = [points]

    headdirection_dict = {
                            -1: 'Unknown',
                            0: 'Front',
                            1: 'R-Front',
                            7: 'L-Front'
                        }

    # roi_polygons: list of (name, raw_points, shapely_Polygon)
    roi_polygons = []
    for roi in stored_config.get('rois', []):
        name = roi.get('name', f'lane{len(roi_polygons) + 1}')
        pts  = roi.get('points', [])
        if len(pts) >= 3:
            roi_polygons.append((name, pts, Polygon(np.array(pts))))
    if not roi_polygons:
        systems_logger.warning('No ROIs defined in config.yml — no detections will be tracked.')

    execution_provider: str = args.execution_provider
    inference_type: str = args.inference_type
    inference_type = inference_type.lower()

    providers_dict = {
        'cpu': [
            'CPUExecutionProvider',
        ],
        'cuda': [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
        'openvino': [
            (
                'OpenVINOExecutionProvider',
                _build_openvino_provider_options(openvino_device, openvino_precision),
            ),
            'CPUExecutionProvider',  # fallback
        ],
        'tensorrt': [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': f'{custom_model}',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                } | (
                    {
                        "trt_fp16_enable": True,
                    } if inference_type == 'fp16' else
                    {
                        "trt_fp16_enable": True,
                        "trt_int8_enable": True,
                        "trt_int8_calibration_table_name": "calibration.flatbuffers",
                    } if inference_type == 'int8' else
                    {
                        "trt_fp16_enable": True,
                    }
                ),
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ],
    }

    providers = providers_dict.get(execution_provider, None)
    preview_enabled = args.view_img
    preview_window_name = 'Detection Application YOLOv9'
    preview_window_created = False

    # Model initialization
    head_model = YOLOv9(
        model_path=custom_model,
        obj_class_score_th=0.35,
        attr_class_score_th=0.70,
        providers=providers,
    )

    # Wrap the distance function to collect per-frame pair distances for debug logging
    _frame_distances: list = []

    if tracker_backend == 'sort':
        from utils.tracker_sort import SORTTracker
        iou_threshold = 1.0 / max(max_distance_between_points, 1e-6)
        tracker = SORTTracker(
            max_age=max_age,
            min_hits=0,
            iou_threshold=iou_threshold,
        )
        print(Color.CYAN(f'[TRACKER] Using SORT (max_age={max_age}, iou_threshold={iou_threshold:.3f})'))
    elif tracker_backend == 'bytetrack':
        from utils.tracker_bytetrack import ByteTracker
        iou_threshold = 1.0 / max(max_distance_between_points, 1e-6)
        tracker = ByteTracker(
            max_age=max_age,
            min_hits=1,
            track_high_thresh=score_threshold,          # matches model score floor (config.yml)
            track_low_thresh=max(score_threshold - 0.15, 0.10),
            iou_threshold_high=iou_threshold,
            iou_threshold_low=max(iou_threshold - 0.10, 0.10),
            iou_threshold_lost=iou_threshold,
        )
        print(Color.CYAN(f'[TRACKER] Using ByteTrack (max_age={max_age}, iou_threshold_high={iou_threshold:.3f})'))
    else:
        # norfair (default)
        if not _norfair_available:
            print(Color.RED('ERROR: tracker=norfair selected but norfair failed to import. '
                            'Set tracker: sort or tracker: bytetrack in config2.yml.'))
            sys.exit(1)

        def _distance_fn(detection, tracked_object):
            dist = iou(detection, tracked_object)
            if debug_mode:
                _frame_distances.append((tracked_object.global_id, dist))
            return dist

        tracker = Tracker(
            distance_function=_distance_fn,
            distance_threshold=max_distance_between_points,
            hit_counter_max=max_age,
            past_detections_length=max_age,
            initialization_delay=0,
        )
        print(Color.CYAN(f'[TRACKER] Using Norfair (max_age={max_age}, distance_threshold={max_distance_between_points})'))

    equip_clf = EquipmentClassifier(
        model_path=equipment_model_path or None,
        score_thresh=equipment_score_thresh,
        heuristic_mode=equipment_heuristic,
    )

    owlv2 = OWLv2Detector(score_thresh=owlv2_score_thresh) if owlv2_enabled else None
    if owlv2_enabled:
        print(Color.CYAN(f'[OWLv2] Enabled — running every {owlv2_interval_frames} frames'))
    _owlv2_frame_count = 0


    # set video stream link
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


    ### Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('live_stream_yolov9_demo.mp4', fourcc, 12, (640, 480))

    ## detection results
    save_path = 'Detections_JSON'
    os.makedirs(save_path, exist_ok=True)

    snapshot_path = 'debug_snapshots'
    os.makedirs(snapshot_path, exist_ok=True)

    captured_dataset_folder = 'captured_dataset'
    os.makedirs(captured_dataset_folder, exist_ok=True)

    caddy_frames_path = 'caddy_frames'
    os.makedirs(caddy_frames_path, exist_ok=True)

    thread = get_capture_thread(cap_url=link, cap_loop=True)
    thread.start()

    db = DBLogger()

    thread_fps = 3

    no_dets = len(os.listdir(save_path)) + 1
    no_snapshots = len(os.listdir(snapshot_path)) + 1

    last_det_time = datetime.now()
    triggered_ids = set() # NEW: track IDs that already triggered alert in this session

    # Dictionary to track PersonDetectionLogger instances per person ID (Aligned with EV App)
    person_loggers = {}
    track_start_times = {}   # when we first see each track ID
    last_dwell_times = {}    # last time update_dwell was called per track ID
    track_conf = {}          # initial confidence per track ID (stored at first detection for DB insert at death)
    track_roi  = {}          # roi_idx per track ID (assigned at first confirmation)
    track_hits = {}          # count of successful detection associations
    track_last_det_id = {}   # handle for tracking new associations
    track_last_bbox = {}     # last estimated bbox per track ID
    prev_active_ids: set = set()
    track_equipment: dict[int, str] = {}  # track_id → equipment label (updated each frame)
    lane_last_insert: dict[str, tuple[int, float, int]] = {}  # roi_name → (track_id, dwell_seconds, active_heads_in_lane)
    recent_track_outcomes = {}  # track_id -> visual outcome overlay for 1 second after death
    last_snapshot_time = 0.0
    last_caddy_frame_time = 0.0
    last_summary_time = time.time()
    SUMMARY_INTERVAL_MIN = 15
    TRACK_RETENTION_SEC = 15 * 60  # keep dead tracks in memory for 15 min
    # Minimum seconds a track must persist to count as "in queue" vs passer-by noise.
    # Tracks shorter than this are still written to entrance_events (for counting)
    # but are excluded from service_events (wait time estimation) and the queue snapshot.
    QUEUE_MIN_DWELL_SEC = float(config2.get('queue_min_dwell_sec', 3.0))
    # Separate (lower) threshold used only for queue_state_snapshots live count.
    QUEUE_SNAPSHOT_MIN_DWELL_SEC = float(config2.get('queue_snapshot_min_dwell_sec', QUEUE_MIN_DWELL_SEC))
    # Caddy/basket image collection. Set interval to 0 to disable.
    CADDY_COLLECT_INTERVAL_SEC = float(config2.get('caddy_collect_interval_sec', 30.0))
    CADDY_COLLECT_MAX_GB       = float(config2.get('caddy_collect_max_gb', 2.0))

    try:
        while True:
            start_time = time.time()
            im0 = thread.read()

            if im0 is None:
                systems_logger.warning('Received empty frame from video stream!')
                time.sleep(0.1)
                continue

            # Resize frame to 640x480 for consistent processing
            im0 = cv2.resize(im0, (640, 480))

            # cv2.imwrite(os.path.join(f'frame_{get_datetime_str()}.jpg'), im0)

            # Run detection and head pose estimation
            boxes, img = detect(im0, head_model, disable_HI)

            h_dets = []; h_conf = []; h_classes = []; h_roi_idxs = []; h_head_dirs = []

            tip_offset = min(config2.get('tip_offset', 1.0), 1.0)

            for lane_number, (roi_name, roi_pts, roi_zone) in enumerate(roi_polygons, start=1):
                pts = np.array(roi_pts, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 255), 3, lineType=cv2.LINE_AA)

            # Get current timestamp for logging and filenames
            timestamp_str = get_datetime_str()

            for h_box in boxes:
                if h_box.classid != 0: # Only consider Head detections for ROI filtering
                    continue

                class_id = h_box.classid
                box_score = h_box.score
                if box_score < score_threshold:
                    continue
                x1, y1, x2, y2 = h_box.x1, h_box.y1, h_box.x2, h_box.y2

                tip_point = Point(x1 + (x2 - x1) * tip_offset, y2 - (y2 - y1) / 25)
                matched_roi_idx = None
                for roi_idx, (roi_name, roi_pts, roi_zone) in enumerate(roi_polygons):
                    if roi_zone.contains(tip_point):
                        matched_roi_idx = roi_idx
                        break

                if matched_roi_idx is not None:
                    h_dets.append([x1, y1, x2, y2])
                    h_conf.append(box_score)
                    h_classes.append(class_id)
                    h_roi_idxs.append(matched_roi_idx)
                    h_head_dirs.append(int(h_box.headdirection) if getattr(h_box, "headdirection", None) is not None else None)

            # Build detections for whichever tracker backend is active
            if tracker_backend == 'norfair':
                frame_detections = []
                for box, conf_val, class_, roi_idx, head_dir in zip(h_dets, h_conf, h_classes, h_roi_idxs, h_head_dirs):
                    data_payload = {
                        "class_id": int(class_),
                        "confidence": float(conf_val),
                        "roi_idx": int(roi_idx),
                        "head_direction": head_dir,
                    }
                    pts = np.array(box).reshape(2, 2)
                    frame_detections.append(Detection(points=pts, data=data_payload))
            else:
                from utils.tracker_base import TrackerDetection
                frame_detections = []
                for box, conf_val, class_, roi_idx, head_dir in zip(h_dets, h_conf, h_classes, h_roi_idxs, h_head_dirs):
                    data_payload = {
                        "class_id": int(class_),
                        "confidence": float(conf_val),
                        "roi_idx": int(roi_idx),
                        "head_direction": head_dir,
                    }
                    frame_detections.append(TrackerDetection(bbox=box, score=float(conf_val), data=data_payload))

            tracked_objects = tracker.update(detections=frame_detections)

            current_time = time.time()
            confirmed_visual_tracks = []
            snapshot_eligible_tracks = []  # in-ROI tracks counted by snapshot but not yet confirmed

            if debug_mode:
                new_ids  = [obj.global_id for obj in tracked_objects if obj.global_id not in track_start_times]
                lost_ids = [tid for tid in prev_active_ids if tid not in {o.global_id for o in tracked_objects}]
                if new_ids or lost_ids:
                    if tracker_backend == 'norfair' and _frame_distances:
                        best  = min(_frame_distances, key=lambda x: x[1])
                        worst = max(_frame_distances, key=lambda x: x[1])
                        dist_info = (f" | dist best={best[1]:.2f}(trk={best[0]})"
                                     f" worst={worst[1]:.2f}(trk={worst[0]})"
                                     f" threshold={max_distance_between_points}")
                    elif tracker_backend == 'norfair':
                        dist_info = " | no existing tracks to match against"
                    else:
                        dist_info = ""
                    systems_logger.debug(
                        f"[TRACK] dets={len(frame_detections)} active={len(tracked_objects)} "
                        f"new={len(new_ids)}{new_ids} lost={len(lost_ids)}{lost_ids}{dist_info}"
                    )
                if tracker_backend == 'norfair':
                    _frame_distances.clear()

            for obj in tracked_objects:
                track_id = obj.global_id

                curr_det_id = id(obj.last_detection) if obj.last_detection else None
                if track_id not in track_last_det_id:
                    track_last_det_id[track_id] = curr_det_id
                    track_hits[track_id] = 1 if curr_det_id else 0
                elif curr_det_id != track_last_det_id[track_id] and curr_det_id is not None:
                    track_last_det_id[track_id] = curr_det_id
                    track_hits[track_id] += 1


                # Always record start time so entry timestamp is correct
                if track_id not in track_start_times:
                    track_start_times[track_id] = current_time

                track_dur = current_time - track_start_times[track_id]

                # Skip logging and display until track has been alive for min_delay seconds
                if track_dur < min_delay:
                    continue

                time_elapsed = float(obj.age / expect_fps)

                person_conf = 0.0
                head_direction = None
                if hasattr(obj.last_detection, "data") and obj.last_detection.data:
                    person_conf = obj.last_detection.data.get("confidence", 0.0)
                    head_direction = obj.last_detection.data.get("head_direction")

                _raw = obj.estimate  # type: ignore[union-attr]
                bx1: int = int(_raw[0][0])  # type: ignore[arg-type]
                by1: int = int(_raw[0][1])  # type: ignore[arg-type]
                bx2: int = int(_raw[1][0])  # type: ignore[arg-type]
                by2: int = int(_raw[1][1])  # type: ignore[arg-type]
                track_last_bbox[track_id] = (bx1, by1, bx2, by2)

                active_tracked_sec = track_hits.get(track_id, 0) / expect_fps
                min_confirmed_track_sec = max_age / expect_fps
                if active_tracked_sec <= min_confirmed_track_sec:
                    if track_dur >= QUEUE_SNAPSHOT_MIN_DWELL_SEC:
                        snap_roi_idx = (obj.last_detection.data.get("roi_idx")
                                        if hasattr(obj.last_detection, "data") and obj.last_detection.data
                                        else None)
                        if snap_roi_idx is not None:
                            snapshot_eligible_tracks.append({
                                "track_id": track_id,
                                "roi_idx": snap_roi_idx,
                                "bbox": (bx1, by1, bx2, by2),
                                "track_dur": track_dur,
                            })
                    continue

                if track_id not in person_loggers:
                    roi_idx = None
                    if hasattr(obj.last_detection, "data") and obj.last_detection.data:
                        roi_idx = obj.last_detection.data.get("roi_idx")
                    track_roi[track_id] = roi_idx

                    roi_label = roi_polygons[roi_idx][0] if roi_idx is not None else "unknown"
                    p_logger = PersonDetectionLogger(detection_logger, camID, person_conf)
                    p_logger.person_id = track_id
                    p_logger.start_block()
                    p_logger.log_tracking_new(track_id, track_dur, time_elapsed, person_conf)
                    person_loggers[track_id] = p_logger
                    track_conf[track_id] = person_conf  # keep for DB insert at track death
                    last_dwell_times[track_id] = current_time
                    systems_logger.debug(f"[TRACK] confirmed id={track_id} lane={roi_label}")

                elif current_time - last_dwell_times.get(track_id, 0) >= 2.0:
                    last_dwell_times[track_id] = current_time

                if hasattr(obj.last_detection, "data") and obj.last_detection.data:
                    current_roi_idx = obj.last_detection.data.get("roi_idx")
                    if current_roi_idx is not None:
                        track_roi[track_id] = current_roi_idx

                confirmed_visual_tracks.append({
                    "track_id": track_id,
                    "roi_idx": track_roi.get(track_id),
                    "bbox": (bx1, by1, bx2, by2),
                    "tip": (
                        int(bx1 + (bx2 - bx1) * tip_offset),
                        int(by2 - (by2 - by1) / 5),
                    ),
                    "head_direction": head_direction,
                    "confidence": person_conf,
                    "track_dur": track_dur,
                })

            # ── Equipment classification (per confirmed track) ────────────────
            if confirmed_visual_tracks:
                frame_bboxes = {
                    t["track_id"]: t["bbox"]
                    for t in confirmed_visual_tracks
                }
                frame_equip = equip_clf.classify_tracks(img, frame_bboxes)
                for _tid, _label in frame_equip.items():
                    if _label != EQUIPMENT_NONE:
                        track_equipment[_tid] = _label

            # ── OWLv2 zero-shot caddy detection (low frame rate) ─────────────
            _owlv2_frame_count += 1
            if owlv2 is not None and confirmed_visual_tracks and _owlv2_frame_count % owlv2_interval_frames == 0:
                frame_bboxes = {
                    t["track_id"]: t["bbox"]
                    for t in confirmed_visual_tracks
                }
                owlv2_equip = owlv2.classify_tracks(im0, frame_bboxes)
                # OWLv2 overrides the heuristic only when it finds something
                for tid, label in owlv2_equip.items():
                    if label != EQUIPMENT_NONE:
                        track_equipment[tid] = label

            best_track_by_lane = {}
            for track_info in confirmed_visual_tracks:
                roi_idx = track_info["roi_idx"]
                if roi_idx is None:
                    continue
                current_best = best_track_by_lane.get(roi_idx)
                if current_best is None or track_info["track_dur"] > current_best["track_dur"]:
                    best_track_by_lane[roi_idx] = track_info

            lane_active_counts = {}
            for track_info in confirmed_visual_tracks:
                roi_idx = track_info["roi_idx"]
                if roi_idx is None:
                    continue
                lane_active_counts[roi_idx] = lane_active_counts.get(roi_idx, 0) + 1

            for track_info in confirmed_visual_tracks:
                track_id = track_info["track_id"]
                roi_idx = track_info["roi_idx"]
                bx1, by1, bx2, by2 = track_info["bbox"]
                tip_x, tip_y = track_info["tip"]
                head_direction = track_info["head_direction"]
                person_conf = track_info["confidence"]
                track_dur = track_info["track_dur"]

                is_best_lane_candidate = (
                    roi_idx is not None
                    and best_track_by_lane.get(roi_idx, {}).get("track_id") == track_id
                )
                track_color = (0, 255, 255) if is_best_lane_candidate else (255, 0, 0)

                PAD = 6
                rx1, ry1, rx2, ry2 = bx1 - PAD, by1 - PAD, bx2 + PAD, by2 + PAD
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), track_color, 2)
                cv2.circle(img, (tip_x, tip_y), 3, (0, 0, 255), -1, cv2.LINE_AA)

                if not disable_HI and head_direction is not None:
                    headpose_label = headdirection_dict.get(head_direction, str(head_direction))
                    cv2.putText(img, f'{headpose_label} {person_conf:.2f}', (rx1, ry1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, track_color, 1, cv2.LINE_AA)

                label_y: int = min(by2 + 20, img.shape[0] - 4)
                label_x: int = max(bx1, 2)

                dwell_label = f'ID:{track_id}  {track_dur:.1f}s' if show_ids else f'{track_dur:.1f}s'
                (tw, th), _ = cv2.getTextSize(dwell_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

                cv2.rectangle(img,
                              (label_x - 2, label_y - th - 6),
                              (label_x + tw + 4, label_y + 2),
                              (20, 20, 20), -1)
                cv2.rectangle(img,
                              (label_x - 2, label_y - th - 6),
                              (label_x + tw + 4, label_y + 2),
                              track_color, 1)
                cv2.putText(img, dwell_label, (label_x + 2, label_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, track_color, 2, cv2.LINE_AA)

                if show_equipment_labels:
                    eq_label = track_equipment.get(track_id, EQUIPMENT_NONE)
                    if eq_label != EQUIPMENT_NONE:
                        eq_color = EQUIPMENT_COLOURS[eq_label]
                        eq_y = label_y + 18
                        (ew, eh), _ = cv2.getTextSize(eq_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(img,
                                      (label_x - 2, eq_y - eh - 4),
                                      (label_x + ew + 4, eq_y + 2),
                                      (20, 20, 20), -1)
                        cv2.putText(img, eq_label, (label_x + 2, eq_y - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, eq_color, 1, cv2.LINE_AA)

            for track_info in snapshot_eligible_tracks:
                bx1, by1, bx2, by2 = track_info["bbox"]
                PAD = 6
                rx1, ry1, rx2, ry2 = bx1 - PAD, by1 - PAD, bx2 + PAD, by2 + PAD
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
                track_dur_s = track_info["track_dur"]
                dwell_label = (f'ID:{track_info["track_id"]}  {track_dur_s:.1f}s'
                               if show_ids else f'{track_dur_s:.1f}s')
                (tw, th), _ = cv2.getTextSize(dwell_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                label_y = min(by2 + 20, img.shape[0] - 4)
                label_x = max(bx1, 2)
                cv2.rectangle(img, (label_x - 2, label_y - th - 6), (label_x + tw + 4, label_y + 2), (20, 20, 20), -1)
                cv2.rectangle(img, (label_x - 2, label_y - th - 6), (label_x + tw + 4, label_y + 2), (255, 255, 255), 1)
                cv2.putText(img, dwell_label, (label_x + 2, label_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            lane_snapshot_counts = dict(lane_active_counts)
            for track_info in snapshot_eligible_tracks:
                roi_idx = track_info["roi_idx"]
                lane_snapshot_counts[roi_idx] = lane_snapshot_counts.get(roi_idx, 0) + 1

            for lane_number, (roi_name, roi_pts, roi_zone) in enumerate(roi_polygons, start=1):
                center_x = int(roi_zone.centroid.x)
                center_y = int(roi_zone.centroid.y)
                active_count = lane_snapshot_counts.get(lane_number - 1, 0)
                lane_label = f"L{lane_number}: {active_count}"
                (tw, th), _ = cv2.getTextSize(lane_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(
                    img,
                    (center_x - tw // 2 - 6, center_y - th - 6),
                    (center_x + tw // 2 + 6, center_y + 6),
                    (20, 20, 20),
                    -1,
                )
                cv2.putText(
                    img,
                    lane_label,
                    (center_x - tw // 2, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ── Track-death detection + stale-track cleanup ───────────────────
            active_ids_set = {obj.global_id for obj in tracked_objects}

            # Insert one DB row per track the moment the tracker drops it
            for tid in prev_active_ids - active_ids_set:
                if tid not in track_start_times:
                    continue
                dead_roi   = track_roi.get(tid)
                dead_dwell = current_time - track_start_times[tid]
                dead_bbox = track_last_bbox.get(tid)
                lane_name = roi_polygons[dead_roi][0] if dead_roi is not None else "unknown"
                confirmed_lane_active_count = 0
                if dead_roi is not None:
                    siblings = [aid for aid in active_ids_set if track_roi.get(aid) == dead_roi]
                    confirmed_lane_active_count = sum(
                        1
                        for aid in siblings
                        if aid in track_start_times
                        and (track_hits.get(aid, 0) / expect_fps) > (max_age / expect_fps)
                    )
                    if siblings:
                        sibling_dwells = [current_time - track_start_times[aid] for aid in siblings if aid in track_start_times]
                        max_sibling_dwell = max(sibling_dwells) if sibling_dwells else 0
                        if max_sibling_dwell > dead_dwell:
                            sibling_info = ", ".join(
                                f"{aid}({current_time - track_start_times[aid]:.1f}s)"
                                if aid in track_start_times else str(aid)
                                for aid in siblings
                            )
                            systems_logger.debug(
                                f"[DB] Suppressed id={tid}({dead_dwell:.1f}s) — "
                                f"{lane_name} has higher dwell [{sibling_info}]"
                            )
                            if show_rejected_track_overlay and dead_bbox is not None:
                                recent_track_outcomes[tid] = {
                                    "bbox": dead_bbox,
                                    "color": (0, 0, 255),
                                    "expires_at": current_time + 2.0,
                                }
                            continue
                entry_time = datetime.fromtimestamp(track_start_times[tid])
                final_dwell = current_time - track_start_times[tid]
                
                active_tracked_sec = track_hits.get(tid, 0) / expect_fps
                min_req_sec = max_age / expect_fps
                
                if active_tracked_sec <= min_req_sec:
                    systems_logger.debug(
                        f"[DB] Skipped id={tid} | lane={lane_name} | "
                        f"active_sec={active_tracked_sec:.1f}s <= threshold={min_req_sec:.1f}s"
                    )
                    if show_rejected_track_overlay and dead_bbox is not None:
                        recent_track_outcomes[tid] = {
                            "bbox": dead_bbox,
                            "color": (0, 0, 255),
                            "expires_at": current_time + 2.0,
                        }
                    continue
                conf = track_conf.get(tid, 0.0)
                equip_type = track_equipment.get(tid, EQUIPMENT_NONE)
                db.insert_entrance(tid, 'unknown', None, conf, camID,
                                   dwell_seconds=final_dwell, entry_time=entry_time,
                                   active_head_tracks_in_lane=confirmed_lane_active_count,
                                   equipment_type=equip_type)
                # Only log as a real wait event if the person stood long enough.
                # Short tracks (< QUEUE_MIN_DWELL_SEC) are passers-by — counting them
                # in service_events inflates the dwell average and distorts wait estimates.
                if final_dwell >= QUEUE_MIN_DWELL_SEC:
                    db.log_service_event(camID, tid, final_dwell, lane_id=dead_roi, equipment_type=equip_type)
                lane_last_insert[lane_name] = (tid, final_dwell, confirmed_lane_active_count)
                systems_logger.info(
                    f"[DB] Inserted id={tid} | lane={lane_name} | dwell={final_dwell:.1f}s "
                    f"| active_heads_in_lane={confirmed_lane_active_count} | cam={camID}"
                )
                if dead_bbox is not None:
                    recent_track_outcomes[tid] = {
                        "bbox": dead_bbox,
                        "color": (0, 180, 0),
                        "expires_at": current_time + 2.0,
                    }

            prev_active_ids = active_ids_set

            # Prune entries older than TRACK_RETENTION_SEC
            stale_ids = [
                tid for tid, t in track_start_times.items()
                if tid not in active_ids_set and current_time - t > TRACK_RETENTION_SEC
            ]
            for tid in stale_ids:
                if tid in person_loggers:
                    person_loggers[tid].end_block()
                    del person_loggers[tid]
                track_start_times.pop(tid, None)
                last_dwell_times.pop(tid, None)
                track_conf.pop(tid, None)
                track_roi.pop(tid, None)
                track_hits.pop(tid, None)
                track_last_det_id.pop(tid, None)
                track_last_bbox.pop(tid, None)
                recent_track_outcomes.pop(tid, None)
                track_equipment.pop(tid, None)

            # ── 15-min activity summary (queried from DB) ─────────────────────
            if current_time - last_summary_time >= SUMMARY_INTERVAL_MIN * 60:
                count, avg_dwell = db.get_period_summary(camID, SUMMARY_INTERVAL_MIN)
                sep = "=" * 60
                if count is not None:
                    systems_logger.info(sep)
                    systems_logger.info(f"  SUMMARY — last {SUMMARY_INTERVAL_MIN} min | cam: {camID}")
                    systems_logger.info(f"  Tracks logged to DB : {count}")
                    systems_logger.info(f"  Avg dwell time      : {avg_dwell:.1f}s")
                    systems_logger.info(sep)
                else:
                    systems_logger.warning("  SUMMARY — DB unavailable, could not retrieve stats.")
                last_summary_time = current_time

            # ── Periodic queue-state snapshot ────────────────────────────────
            if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
                active_ids = [obj.global_id for obj in tracked_objects]
                dwells = [current_time - track_start_times[tid]
                          for tid in active_ids if tid in track_start_times]
                long_dwells = [d for d in dwells if d >= QUEUE_SNAPSHOT_MIN_DWELL_SEC]
                # Use the same per-lane count that is shown on the overlay (0.25 s threshold).
                queue_count = sum(lane_snapshot_counts.values())
                avg_dwell = float(sum(long_dwells) / len(long_dwells)) if long_dwells else 0.0
                max_dwell = float(max(long_dwells)) if long_dwells else 0.0
                db.log_queue_snapshot(camID, queue_count, avg_dwell, max_dwell,
                                      lane_counts=lane_snapshot_counts)
                last_snapshot_time = current_time

            # ── Caddy dataset collection (raw frame, low framerate) ───────────
            if (CADDY_COLLECT_INTERVAL_SEC > 0
                    and current_time - last_caddy_frame_time >= CADDY_COLLECT_INTERVAL_SEC):
                folder_size_gb = sum(
                    os.path.getsize(os.path.join(caddy_frames_path, f))
                    for f in os.listdir(caddy_frames_path)
                    if os.path.isfile(os.path.join(caddy_frames_path, f))
                ) / 1e9
                if folder_size_gb < CADDY_COLLECT_MAX_GB:
                    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(
                        os.path.join(caddy_frames_path, f'{camID}_{ts_str}.jpg'),
                        im0,
                        [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )
                last_caddy_frame_time = current_time

            # Control loop timing to match desired FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / thread_fps) - elapsed_time)
            time.sleep(sleep_time)
            end_time = time.time()
            fps = max(round(1 / (end_time - start_time)), 1)
            write_text(img, f"FPS: {fps}", position="bottom_right")

            expired_outcome_ids = [
                tid for tid, overlay in recent_track_outcomes.items()
                if overlay["expires_at"] <= current_time
            ]
            for tid in expired_outcome_ids:
                recent_track_outcomes.pop(tid, None)

            for overlay in recent_track_outcomes.values():
                bx1, by1, bx2, by2 = overlay["bbox"]
                color = overlay["color"]
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 3)

            # ── Per-lane last-insert overlay (bottom-left) ────────────────────
            _line_h = 24
            _y = img.shape[0] - 10 - (len(roi_polygons) - 1) * _line_h
            for roi_name, _, _ in roi_polygons:
                if roi_name in lane_last_insert:
                    _lid, _last_dwell, _active_heads = lane_last_insert[roi_name]
                    _label = f"{roi_name}  ID={_lid}  {_last_dwell:.1f}s  heads={_active_heads}"
                    _color = (0, 200, 255)
                else:
                    _label = f"{roi_name}  --"
                    _color = (140, 140, 140)
                (_tw, _th), _ = cv2.getTextSize(_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (6, _y - _th - 4), (10 + _tw, _y + 4), (20, 20, 20), -1)
                cv2.putText(img, _label, (8, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _color, 1, cv2.LINE_AA)
                _y += _line_h

            # Save debug image with detections drawn (optional)
            if save_snapshots:
                debug_image_path = os.path.join(snapshot_path, f'debug_{no_snapshots}.jpg')
                cv2.imwrite(debug_image_path, img)
                no_snapshots += 1

            if preview_enabled:
                try:
                    if not preview_window_created:
                        cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
                        preview_window_created = True
                    im_show = cv2.resize(img, (600, 400), interpolation=cv2.INTER_AREA)
                    cv2.imshow(preview_window_name, im_show)
                    if cv2.waitKey(1) in {ord("q"), ord("Q"), 27}:
                        break
                except cv2.error as exc:
                    print(Color.YELLOW(
                        f'[VIEW] WARNING: Failed to open preview window, disabling --view-img. Reason: {exc}'
                    ))
                    preview_enabled = False

    except Exception as e:
        print(f"{get_datetime_str()} ERROR: An exception occurred - {str(e)}")
        systems_logger.error(f"An exception occurred - {str(e)}", exc_info=True)

    # except KeyboardInterrupt:
    #     # print("INFO: Keyboard interrupt received. Stopping video stream and exiting.")
    #     systems_logger.info('Keyboard interrupt received. Stopping video stream and exiting.')

    finally:
        if 'db' in locals(): db.close()
        if 'video_writer' in locals(): video_writer.release()
        if 'thread' in locals(): thread.stop()
        if preview_window_created:
            cv2.destroyAllWindows()
        print(f"{get_datetime_str()} INFO: Cleanup complete. Exiting.")
        os._exit(0)



if __name__ == '__main__':
    main()
