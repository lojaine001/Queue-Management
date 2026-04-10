import os
import logging
import sys
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
# from flask import Flask, request, jsonify


from utils.queue_utils import get_config, create_logger, plot_one_box, iou
from utils.queue_utils import get_datetime_str, PersonDetectionLogger
from utils.db_logger import DBLogger

from utils.yolo import Color, is_package_installed, YOLOv9


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
    parser.add_argument('--inference_type', type=str, choices=['fp16', 'int8'], default='fp16', help='Inference type. Default: fp16')
    parser.add_argument('--execution_provider', type=str, choices=['cpu', 'cuda', 'tensorrt'], default='tensorrt', help='Execution provider for ONNXRuntime.')
    parser.add_argument('--custom_weights', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--out', type=str, help='path to save demo') # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
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

    # New configurations for endpoints
    endpoint = stored_config.get('endpoint', 'http://127.0.0.1:5000/vsens')
    mode = stored_config.get('mode', 'restAPI') # [restAPI, digitalIO]
    auth_type = stored_config.get('auth_type', 'none') # [none, basic, digest]

    username = stored_config.get('username', 'admin')
    password = stored_config.get('password', 'Wt@5651%')

    if not is_package_installed('onnxruntime'):
        print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
        sys.exit(0)

    # YOLOv9 Model handles (assume stored locally in weights folder)
    # pretrained_model = stored_config.get('pretrained_model', 'models/yolov9-t.pt')
    custom_model = stored_config.get('custom_model', 'models/yolov9_c_discrete_headpose_post_0100_1x3x480x640.onnx')


    score = stored_config.get('score', 0.3)
    min_score = stored_config.get('min_score', 0.3)
    iou_score = stored_config.get('iou_score', 0.3)
    min_delay = stored_config.get('min_delay', 0.0)

    track_all = config2.get('track_all', False)
    pretrained_classes = config2.get('pretrained_classes', ['car', 'airplane', 'bus', 'train', 'truck', 'boat'])
    pattern_classes = stored_config.get('pattern_classes', ['TaxiSign'])
    debug_mode = config2.get('debug_mode', True)
    max_distance_between_points = config2.get('max_distance_between_points', 2)
    max_age = config2.get('max_age', 10)
    expect_fps = config2.get('expect_fps', 3)
    snapshot_class_only = config2.get('snapshot_classes', [])
    min_elapsed_time = config2.get('min_elapsed_time', 1)
    save_one = config2.get('save_one', True)
    disable_HI = stored_config.get('disable_headpose_identification_mode', False)

    # h, w = im0.shape[:2]

    # points = [[0,0], [w,0], [w,h], [0,h]]
    # polygons = [points]

    headdirection_dict = {
                            -1: 'Unknown',
                            0: 'Front',
                            1: 'R-Front',
                            7: 'L-Front'
                        }

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

    # Model initialization
    head_model = YOLOv9(
        model_path=custom_model,
        obj_class_score_th=0.35,
        attr_class_score_th=0.70,
        providers=providers,
    )

    tracker = Tracker(
        distance_function = iou,
        distance_threshold=max_distance_between_points,
        hit_counter_max = max_age,
        past_detections_length = max_age,
        initialization_delay = min_delay,
    )


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
            camID = 'live_test_video'


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
    track_start_times = {}  # To track when we first see each track ID

    try:
        while True:
            start_time = time.time()
            im0 = thread.read()

            # if im0 is None:
            #     systems_logger.warning('Received empty frame from video stream!')
            #     time.sleep(0.01) # Sleep briefly before trying to read again
            #     continue

            # Resize frame to 640x480 for consistent processing
            im0 = cv2.resize(im0, (640, 480))

            # cv2.imwrite(os.path.join(f'frame_{get_datetime_str()}.jpg'), im0)

            # Run detection and head pose estimation
            boxes, img = detect(im0, head_model, disable_HI)

            h_dets = []; h_conf = []; h_classes = []

            for poly in polygons:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(img,[pts],True,(0,255,255),3,lineType=cv2.LINE_AA)

            # Get current timestamp for logging and filenames
            timestamp_str = get_datetime_str()

            for h_box in boxes:
                if h_box.classid != 0: # Only consider Head detections for ROI filtering
                    continue

                class_id = h_box.classid
                score = h_box.score
                x1, y1, x2, y2 = h_box.x1, h_box.y1, h_box.x2, h_box.y2

                ROI_Incl = False
                tip_point = None
                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32)
                    tip_offset = min(config2.get('tip_offset', 1.0), 1.0)
                    tip_point = Point(x1 + (x2-x1)*tip_offset, y2 - (y2-y1)/5) # Tip point at 20% from the top of the head box
                    zone = Polygon(pts)
                    if zone.contains(tip_point):
                        ROI_Incl = True
                        break

                if ROI_Incl:
                    h_dets.append([x1, y1, x2, y2])
                    h_conf.append(score)
                    h_classes.append(class_id)

                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                    if tip_point:
                        cv2.circle(img, (int(tip_point.x), int(tip_point.y)), 1, (0, 0, 255), -1, cv2.LINE_AA)

                    if not disable_HI:
                        headpose_label = headdirection_dict[h_box.headdirection]
                        cv2.putText(img, f'{headpose_label} {score:.2f}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


                # Prepare Norfair detections
                norfair_detections = []
                for box, conf_val, class_ in zip(h_dets, h_conf, h_classes):
                    data_payload = {"class_id": int(class_), "confidence": float(conf_val)}
                    pts = np.array(box).reshape(2, 2)
                    det = Detection(points=pts, data=data_payload)
                    norfair_detections.append(det)

                tracked_objects = tracker.update(detections=norfair_detections)

                # norfair.draw_boxes(
                #     im0,
                #     tracked_objects,
                #     color=(0, 0, 255),
                #     text_size=1,
                #     draw_ids=True,
                #     thickness=2,
                #     draw_scores=False,
                #     draw_labels=False,
                #     text_color=(255, 0, 0)
                # )

                current_time = time.time()

                for obj in tracked_objects:
                    track_id = obj.global_id

                    # store first time we see this track
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

                        # Insert new checkout person into PostgreSQL
                        db.insert_queue_event(track_id, person_conf, camID)
                        print(f"[DB] Inserted track_id={track_id} | conf={person_conf:.2f} | cam={camID}")

                    # elif int(time_elapsed) % 2 == 0:  # update every ~2 seconds
                    elif int(track_dur) % 2 == 0:  # update every ~2 seconds
                        person_loggers[track_id].log_tracking_update(track_id, track_dur, time_elapsed, person_conf)
                        # Update dwell time in DB every 2 seconds
                        db.update_dwell(track_id, track_dur)

                    # Draw dwell time (always) and ID (debug only)
                    pts = obj.estimate.astype(int)
                    bx1, by1 = pts[0]
                    _,   _   = pts[1]

                    dwell_label = f'{track_dur:.1f}s' if not debug_mode else f'{track_dur:.1f}s  ID:{track_id}'
                    (tw, th), _ = cv2.getTextSize(dwell_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(img, (bx1, by1 - th - 10), (bx1 + tw + 4, by1), (0, 0, 0), -1)
                    cv2.putText(img, dwell_label, (bx1 + 2, by1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)


            # Save debug image with detections drawn (optional)
            debug_image_path = os.path.join(snapshot_path, f'debug_{no_snapshots}.jpg')
            cv2.imwrite(debug_image_path, img)
            no_snapshots += 1

            # Control loop timing to match desired FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / thread_fps) - elapsed_time)
            time.sleep(sleep_time)

            if args.view_img:
                im_show = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
                cv2.imshow('Detection Application YOLOv9', im_show)
                if cv2.waitKey(1) in {ord("q"), ord("Q"), 27}:
                    break

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
        if args.view_img: # Only destroy windows if they were created
            cv2.destroyAllWindows()
        print(f"{get_datetime_str()} INFO: Cleanup complete. Exiting.")
        os._exit(0)



if __name__ == '__main__':
    main()
