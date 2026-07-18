from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import copy
import ctypes
import cv2
import time
import onnx
import onnxruntime # type: ignore
from pprint import pprint
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod
from contextlib import redirect_stderr
from io import StringIO


class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)
    
@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    generation: int = -1 # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1 # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    headdirection: int = -1 # -1: Unknown, 0: front, 1: right-front, 2: right-side, 3: right-back, 4: back, 5: left-back, 6: left-side, 7: left-front
    is_used: bool = False



class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        # runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        # self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._providers = providers

        # Model loading — redirect OS-level stderr during session creation so
        # C++ provider-load errors (e.g. missing OpenVINO DLL) don't pollute output.
        onnxruntime.set_default_logger_severity(3)
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3

        _devnull_fd  = os.open(os.devnull, os.O_WRONLY)
        _old_stderr  = os.dup(2)
        provider_stderr = StringIO()
        self._provider_init_stderr = ""
        os.dup2(_devnull_fd, 2)
        try:
            self._preload_openvino_runtime()
            with redirect_stderr(provider_stderr):
                self._interpreter = self._create_inference_session(
                    model_path=model_path,
                    sess_options=session_option,
                    providers=providers,
                )
        finally:
            os.dup2(_old_stderr, 2)
            os.close(_old_stderr)
            os.close(_devnull_fd)
            self._provider_init_stderr = provider_stderr.getvalue().strip()

        requested_provider_names = [
            self._provider_name(provider) for provider in (providers or [])
        ]
        self._providers = self._interpreter.get_providers()
        print(f'YOLOv9 requested providers: {requested_provider_names}')
        print(f'YOLOv9 enabled providers: {self._providers}')

        if 'OpenVINOExecutionProvider' in requested_provider_names:
            if 'OpenVINOExecutionProvider' in self._providers:
                print('OpenVINO is active for this session.')
            else:
                print('OpenVINO was requested but is not active; ONNX Runtime fell back to CPU.')

        onnx_graph: onnx.ModelProto = onnx.load(model_path)
        if onnx_graph.graph.node[0].op_type == "Resize":
            first_resize_op: List[onnx.ValueInfoProto] = [i for i in onnx_graph.graph.value_info if i.name == "prep/Resize_output_0"]
            if first_resize_op:
                self._input_shapes = [[d.dim_value for d in first_resize_op[0].type.tensor_type.shape.dim]]
            else:
                self._input_shapes = [
                    input.shape for input in self._interpreter.get_inputs()
                ]
        else:
            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]


        self._input_names = [
            input.name for input in self._interpreter.get_inputs()
        ]
        self._input_dtypes = [
            self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
        ]
        self._output_shapes = [
            output.shape for output in self._interpreter.get_outputs()
        ]
        self._output_names = [
            output.name for output in self._interpreter.get_outputs()
        ]
        self._model = self._interpreter.run
        self._swap = (2, 0, 1)
        self._h_index = 2
        self._w_index = 3

    @staticmethod
    def _provider_name(provider) -> str:
        if isinstance(provider, tuple):
            return provider[0]
        return provider

    def _preload_openvino_runtime(self) -> None:
        if sys.platform != 'win32':
            return

        provider_names = [self._provider_name(provider) for provider in (self._providers or [])]
        if 'OpenVINOExecutionProvider' not in provider_names:
            return

        ov_spec = importlib.util.find_spec('openvino')
        ort_spec = importlib.util.find_spec('onnxruntime')
        if not ov_spec or not ov_spec.origin or not ort_spec or not ort_spec.origin:
            return

        ov_dir = Path(ov_spec.origin).resolve().parent
        ov_libs = ov_dir / 'libs'
        ort_capi = Path(ort_spec.origin).resolve().parent / 'capi'

        self._provider_dll_handles = []
        for dll_dir in [ov_libs, ort_capi]:
            if not dll_dir.is_dir():
                continue
            try:
                self._provider_dll_handles.append(os.add_dll_directory(str(dll_dir)))
            except (AttributeError, FileNotFoundError, OSError):
                continue

        preload_chain = [
            ov_libs / 'tbb12.dll',
            ov_libs / 'openvino.dll',
            ov_libs / 'openvino_c.dll',
            ort_capi / 'onnxruntime.dll',
            ort_capi / 'onnxruntime_providers_shared.dll',
        ]
        for dll_path in preload_chain:
            if not dll_path.exists():
                continue
            ctypes.WinDLL(str(dll_path))

    def _collect_openvino_debug_context(self) -> List[str]:
        details: List[str] = []

        try:
            import importlib.metadata as importlib_metadata
            for name in ['onnxruntime-openvino', 'openvino', 'openvino-telemetry']:
                try:
                    details.append(f'{name}={importlib_metadata.version(name)}')
                except importlib_metadata.PackageNotFoundError:
                    details.append(f'{name}=missing')
        except Exception as version_exc:
            details.append(f'version_probe_error={version_exc}')

        try:
            ov_spec = importlib.util.find_spec('openvino')
            if ov_spec and ov_spec.origin:
                ov_dir = Path(ov_spec.origin).resolve().parent
                ov_lib = ov_dir / 'libs' / 'openvino.dll'
                details.append(f'openvino_origin={ov_spec.origin}')
                details.append(f'openvino_dll_exists={ov_lib.exists()}')
        except Exception as ov_exc:
            details.append(f'openvino_probe_error={ov_exc}')

        if getattr(self, '_provider_init_stderr', ''):
            details.append(f'provider_stderr={self._provider_init_stderr}')

        return details

    def _create_inference_session(
        self,
        *,
        model_path: str,
        sess_options: onnxruntime.SessionOptions,
        providers: Optional[List],
    ):
        if not providers:
            return onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
                enable_fallback=False,
            )

        provider_names = [self._provider_name(provider) for provider in providers]
        has_openvino = 'OpenVINOExecutionProvider' in provider_names
        has_cpu_fallback = 'CPUExecutionProvider' in provider_names

        if not has_openvino or not has_cpu_fallback:
            return onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
                enable_fallback=False,
            )

        try:
            session = onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
                enable_fallback=False,
            )
            enabled = session.get_providers()
            if 'OpenVINOExecutionProvider' not in enabled:
                raise RuntimeError(
                    f'OpenVINOExecutionProvider was requested but the session enabled {enabled}.'
                )
            return session
        except Exception as exc:
            print(Color.YELLOW(
                f'[ORT] WARNING: OpenVINO session initialization failed, falling back to CPU. Reason: {exc}'
            ))
            debug_details = self._collect_openvino_debug_context()
            if debug_details:
                print(Color.YELLOW('[ORT] OpenVINO debug context:'))
                for detail in debug_details:
                    print(Color.YELLOW(f'  - {detail}'))
            return onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider'],
                enable_fallback=False,
            )


    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }

        outputs = [
            output for output in \
                self._model(
                    output_names=self._output_names,
                    input_feed=datas,
                )
        ]
        return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()


class YOLOv9(AbstractModel):
    def __init__(
        self,
        *,
        # runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolov9_s_discrete_headpose_post_0100_1x3x480x640.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOv9. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOv9

        obj_class_score_th: Optional[float]
            Object score threshold. Default: 0.35

        attr_class_score_th: Optional[float]
            Attributes score threshold. Default: 0.70

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            # runtime=runtime,
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9

    def __call__(
        self,
        image: np.ndarray,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, handedness, is_hand_used=False]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )

        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]. [instances, [batchno, classid, score, x1, y1, x2, y2]].

        disable_left_and_right_hand_identification_mode: bool

        disable_gender_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, handedness, is_hand_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._obj_class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                # Object filter
                for box, score in zip(boxes_keep, scores_keep):
                    classid = int(box[1])
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    result_boxes.append(
                        Box(
                            classid=classid,
                            # score=float(score),
                            score=float(score.item()),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                            generation=-1, # -1: Unknown, 0: Adult, 1: Child
                            gender=-1, # -1: Unknown, 0: Male, 1: Female
                            handedness=-1, # -1: Unknown, 0: Left, 1: Right
                            headdirection=-1, # -1: Unknown, 0: front, 1: right-front, 2: right-side, 3: right-back, 4: back, 5: left-back, 6: left-side, 7: left-front
                        )
                    )
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [0,1,2,3,4,5,6,7,8] and box.score >= self._attr_class_score_th) or box.classid not in [0,1,2,3,4,5,6,7,8]
                ]

                # Head-Pose merge
                # classid: 0 -> Head
                #   classid: 1 -> front
                #   classid: 2 -> right-front
                #   classid: 3 -> right-side
                #   classid: 4 -> right-back
                #   classid: 5 -> back
                #   classid: 6 -> left-back
                #   classid: 7 -> left-side
                #   classid: 8 -> left-front
                # =========================================================
                # 1. Calculate HeadPose IoUs for Head detection results
                # 2. Connect either the HeadPose with the highest score and the highest IoU with the Head.
                # 3. Exclude HeadPose from detection results
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 0]
                    headpose_boxes = [box for box in result_boxes if box.classid in [1,2,8]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1,2,3,4,5,6,7,8]]
        return result_boxes

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                # Process only unused objects with center Euclidean distance less than or equal to 10.0
                if not target_obj.is_used and distance <= 10.0:
                    # Prioritize high-score objects
                    if target_obj.score >= best_score:
                        # IoU Calculation
                        iou: float = \
                            self._calculate_iou(
                                base_obj=base_obj,
                                target_obj=target_obj,
                            )
                        # Adopt object with highest IoU
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                base_obj.headdirection = most_relevant_obj.classid - 1
                most_relevant_obj.is_used = True

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou


def list_image_files(dir_path: str) -> List[str]:
    path = Path(dir_path)
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(path.rglob(extension))
    return sorted([str(file) for file in image_files])

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)
