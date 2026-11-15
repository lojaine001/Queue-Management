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
import onnxruntime  # type: ignore
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Sequence, Union
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
    generation: int = -1     # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1         # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1     # -1: Unknown, 0: Left, 1: Right
    headdirection: int = -1  # -1: Unknown, 0: front, 1: right-front, ...
    is_used: bool = False
    rbox: Optional[Tuple[float, float, float, float, float]] = None  # (cx, cy, w, h, angle_deg) when from RAPiD


def rotated_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two rotated boxes (cx, cy, w, h, angle_deg)."""
    ra = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), float(a[4]))
    rb = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(b[4]))
    kind, pts = cv2.rotatedRectangleIntersection(ra, rb)
    if kind == cv2.INTERSECT_NONE or pts is None:
        return 0.0
    inter = cv2.contourArea(cv2.convexHull(pts))
    union = float(a[2]) * float(a[3]) + float(b[2]) * float(b[3]) - inter
    return float(inter / union) if union > 0 else 0.0


def rotated_nms(dts: np.ndarray, thres: float = 0.45) -> np.ndarray:
    """Greedy NMS on rows (cx, cy, w, h, angle_deg, score), highest score first."""
    if len(dts) == 0:
        return dts
    dts = dts[np.argsort(-dts[:, 5], kind='stable')]
    centres = dts[:, :2]
    radius = 0.5 * np.hypot(dts[:, 2], dts[:, 3])
    keep, suppressed = [], np.zeros(len(dts), dtype=bool)
    for i in range(len(dts)):
        if suppressed[i]:
            continue
        keep.append(i)
        near = (
            np.nonzero(
                ~suppressed[i + 1:]
                & (np.hypot(*(centres[i + 1:] - centres[i]).T) < radius[i + 1:] + radius[i])
            )[0]
            + i
            + 1
        )
        for j in near:
            if rotated_iou(dts[i], dts[j]) > thres:
                suppressed[j] = True
    return dts[keep]


def letterbox_image(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (1280, 1280),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Resize and pad image to `new_shape` (height, width) while preserving aspect ratio.
    Matches `letterbox(..., auto=False)` in UT_AICITY2025_Track4.
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def unscale_letterbox_coords(
    boxes_xyxy: np.ndarray,
    img_shape: Tuple[int, int],
    ratio: Tuple[float, float],
    pad: Tuple[float, float],
) -> np.ndarray:
    """Map letterboxed [x1, y1, x2, y2] coordinates back to original `img_shape` (H, W)."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    coords = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    coords[:, [0, 2]] = (coords[:, [0, 2]] - pad[0]) / max(ratio[0], 1e-9)
    coords[:, [1, 3]] = (coords[:, [1, 3]] - pad[1]) / max(ratio[1], 1e-9)
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img_shape[0])
    return coords


def nms_raw_outputs(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_th: float = 0.45,
    score_th: float = 0.25,
    max_per_class: int = 100,
) -> np.ndarray:
    """CPU greedy NMS per class.

    boxes: float32[1, A, 4] x1y1x2y2, scores: float32[1, C, A]
    returns float32[N, 7]: [batchno, classid, score, x1, y1, x2, y2]
    """
    boxes = boxes[0]
    results = []
    for classid, class_scores in enumerate(scores[0]):
        idxs = np.where(class_scores > score_th)[0]
        if len(idxs) == 0:
            continue
        idxs = idxs[np.argsort(-class_scores[idxs], kind='stable')]
        b = boxes[idxs]
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep: List[int] = []
        suppressed = np.zeros(len(idxs), dtype=bool)
        for i in range(len(idxs)):
            if suppressed[i]:
                continue
            keep.append(i)
            if len(keep) >= max_per_class:
                break
            xx1 = np.maximum(b[i, 0], b[i + 1:, 0])
            yy1 = np.maximum(b[i, 1], b[i + 1:, 1])
            xx2 = np.minimum(b[i, 2], b[i + 1:, 2])
            yy2 = np.minimum(b[i, 3], b[i + 1:, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            ious = inter / (areas[i] + areas[i + 1:] - inter + 1e-9)
            suppressed[i + 1:] |= ious > iou_th
        for i in keep:
            results.append([0, classid, class_scores[idxs[i]], *b[i]])
    if not results:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(results, dtype=np.float32)


def weighted_boxes_fusion_np(
    boxes_list: Sequence[Sequence[Sequence[float]]],
    scores_list: Sequence[Sequence[float]],
    labels_list: Sequence[Sequence[int]],
    weights: Optional[Sequence[float]] = None,
    iou_thr: float = 0.65,
    skip_box_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Boxes Fusion (WBF) matching UT_AICITY2025_Track4/dataprocessing/infer_YR_Y10_Y13_WBF.py.
    Uses `ensemble_boxes.weighted_boxes_fusion` when installed, or an equivalent pure-NumPy fallback.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion  # type: ignore
        b, s, l = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        )
        return np.asarray(b, dtype=np.float32), np.asarray(s, dtype=np.float32), np.asarray(l, dtype=np.int64)
    except ImportError:
        pass

    n_models = len(boxes_list)
    if n_models == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    w = np.ones(n_models, dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    w_sum = float(np.sum(w)) if np.sum(w) > 0 else 1.0

    # Collect per-label boxes: [x1, y1, x2, y2, weighted_score, raw_score, model_weight]
    by_label: Dict[int, List[np.ndarray]] = {}
    for m_idx in range(n_models):
        m_w = float(w[m_idx])
        for box, score, label in zip(boxes_list[m_idx], scores_list[m_idx], labels_list[m_idx]):
            sc = float(score)
            if sc < skip_box_thr:
                continue
            x1, y1, x2, y2 = (float(np.clip(v, 0.0, 1.0)) for v in box)
            if x2 <= x1 or y2 <= y1:
                continue
            lbl = int(label)
            by_label.setdefault(lbl, []).append(
                np.array([x1, y1, x2, y2, sc * m_w, sc, m_w], dtype=np.float32)
            )

    fused_boxes, fused_scores, fused_labels = [], [], []
    for lbl, entries in by_label.items():
        arr = np.stack(entries, axis=0)
        arr = arr[np.argsort(-arr[:, 5], kind='stable')]
        clusters: List[List[np.ndarray]] = []
        cluster_reps: List[np.ndarray] = []

        for item in arr:
            best_iou = -1.0
            best_c = -1
            for c_idx, rep in enumerate(cluster_reps):
                xx1 = max(float(item[0]), float(rep[0]))
                yy1 = max(float(item[1]), float(rep[1]))
                xx2 = min(float(item[2]), float(rep[2]))
                yy2 = min(float(item[3]), float(rep[3]))
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                if inter <= 0.0:
                    continue
                a1 = (item[2] - item[0]) * (item[3] - item[1])
                a2 = (rep[2] - rep[0]) * (rep[3] - rep[1])
                iou_val = inter / max(float(a1 + a2 - inter), 1e-9)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_c = c_idx

            if best_c >= 0 and best_iou >= iou_thr:
                clusters[best_c].append(item)
                c_arr = np.stack(clusters[best_c], axis=0)
                w_scores = c_arr[:, 4]
                rep_xyxy = np.sum(c_arr[:, :4] * w_scores[:, None], axis=0) / max(float(np.sum(w_scores)), 1e-9)
                cluster_reps[best_c] = rep_xyxy
            else:
                clusters.append([item])
                cluster_reps.append(item[:4].copy())

        for c_idx, c_items in enumerate(clusters):
            c_arr = np.stack(c_items, axis=0)
            rep_xyxy = cluster_reps[c_idx]
            # Standard WBF confidence: sum(score_i * weight_i) / sum(all_model_weights) * min(len(cluster), n_models) / n_models
            # Simplified standard WBF rescale: sum(score_i * weight_i) / sum(all_model_weights)
            conf = float(np.sum(c_arr[:, 4])) / w_sum
            fused_boxes.append(rep_xyxy)
            fused_scores.append(conf)
            fused_labels.append(lbl)

    if not fused_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    order = np.argsort(-np.asarray(fused_scores, dtype=np.float32), kind='stable')
    return (
        np.asarray(fused_boxes, dtype=np.float32)[order],
        np.asarray(fused_scores, dtype=np.float32)[order],
        np.asarray(fused_labels, dtype=np.int64)[order],
    )


class AbstractModel(ABC):
    """Base ONNX model wrapper compatible with Head-Detector/utils/yolo.py."""
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    def __init__(
        self,
        *,
        model_path: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        providers: Optional[List] = None,
    ):
        self._model_path = model_path or ''
        self._obj_class_score_th = float(obj_class_score_th if obj_class_score_th is not None else 0.35)
        self._attr_class_score_th = float(attr_class_score_th if attr_class_score_th is not None else 0.70)
        self._providers = providers

    @staticmethod
    def _provider_name(provider) -> str:
        if isinstance(provider, tuple):
            return provider[0]
        return provider

    def _preload_openvino_runtime(self) -> None:
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

        if sys.platform != 'win32':
            for so_dir in (ov_libs, ort_capi):
                if not so_dir.is_dir():
                    continue
                for pattern in ('libtbb.so*', 'libopenvino.so*', 'libopenvino_c.so*'):
                    for so_path in sorted(so_dir.glob(pattern)):
                        try:
                            ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            continue
            return

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
            f'{input_name}': input_data
            for input_name, input_data in zip(self._input_names, input_datas)
        }
        outputs = [
            output
            for output in self._model(
                output_names=self._output_names,
                input_feed=datas,
            )
        ]
        return outputs


class FisheyeDetector(AbstractModel):
    """Fisheye person/pedestrian detector with the exact same API as `Head-Detector/utils/yolo.py`'s `YOLOv9`.

    Supports:
    1. ONNX models (`.onnx`) via ONNX Runtime (`cpu`, `cuda`, `openvino`, `tensorrt`):
       - UT_AICITY2025_Track4 YOLOv10 end-to-end ONNX (`[1, N, 6]` -> `[x1, y1, x2, y2, score, class_id]`)
       - UT_AICITY2025_Track4 YOLOv13 / YOLOv8 / YOLOv9 ONNX (`[1, 4+C, A]` or `[1, A, 4+C]`)
       - UT_AICITY2025_Track4 YOLOR ONNX (`[1, A, 5+C]`)
       - PINTO `*_nonms_*` (`x1y1x2y2`, `main01_scores`) and `[N, 7]` post-NMS graphs
    2. Ultralytics PyTorch / TensorRT models (`.pt` / `.engine`) for YOLOv10 / YOLOv13
    3. YOLOR PyTorch models (`.pt`) via `UT_AICITY2025_Track4/train/YoloR`
    4. 3-model Weighted Boxes Fusion ensemble (`wbf_models={'yolor': ..., 'y10': ..., 'y13': ...}`)
       from `UT_AICITY2025_Track4/dataprocessing/infer_YR_Y10_Y13_WBF.py`
    5. RAPiD rotation-aware fisheye people detector (`.ckpt`, e.g. `pL1_MWHB1024_Mar11_4000.ckpt`)

    Matched person detections (e.g. `class_id == 3` Pedestrian on Fisheye8K/VisDrone, or `0` on COCO/Head/RAPiD)
    are returned as `Box(classid=0, score=..., x1=..., y1=..., x2=..., y2=..., cx=..., cy=...)` so the
    downstream tracking, ROI filtering, and DB logging work identically to `Head-Detector`.
    """

    # Fisheye8K / VisDrone 5-class mapping from UT_AICITY2025_Track4:
    # 0: Bus, 1: Bike, 2: Car, 3: Pedestrian, 4: Truck
    FISHEYE8K_PEDESTRIAN_CLASS_ID = 3

    def __init__(
        self,
        *,
        model_path: Optional[str] = 'models/fisheye_yolo.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        providers: Optional[List] = None,
        iou_th: float = 0.45,
        img_size: int = 1280,
        person_class_ids: Optional[Sequence[int]] = None,
        detector_backend: str = 'auto',
        wbf_models: Optional[Dict[str, str]] = None,
        wbf_weights: Sequence[float] = (9.0, 9.0, 9.0),
        wbf_iou: float = 0.65,
        wbf_confs: Tuple[float, float, float] = (0.50, 0.50, 0.495),
        wbf_ious: Tuple[float, float, float] = (0.55, 0.65, 0.45),
        rapid_dir: Optional[str] = None,
        rapid_backbone: str = 'dark53',
        device: str = 'cpu',
    ):
        super().__init__(
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            providers=providers,
        )
        self._iou_th = float(iou_th)
        self._default_img_size = int(img_size)
        self._explicit_person_class_ids = (
            {int(c) for c in person_class_ids} if person_class_ids is not None else None
        )
        self._backend = str(detector_backend or 'auto').strip().lower()
        if self._backend in ('yolov10', 'yolov13'):
            self._backend = 'ultralytics'
        self._wbf_models_cfg = dict(wbf_models) if wbf_models else {}
        self._wbf_weights = [float(w) for w in wbf_weights]
        self._wbf_iou = float(wbf_iou)
        self._wbf_confs = tuple(float(c) for c in wbf_confs)
        self._wbf_ious = tuple(float(i) for i in wbf_ious)
        self._rapid_dir = rapid_dir
        self._rapid_backbone = rapid_backbone
        self._device = self._infer_torch_device(device, providers)
        self._pinto_style_graph = False

        if self._backend == 'wbf' or (
            self._backend == 'auto'
            and self._wbf_models_cfg.get('yolor')
            and self._wbf_models_cfg.get('y10')
            and self._wbf_models_cfg.get('y13')
        ):
            self._backend = 'wbf'
            self._init_wbf_backend()
        elif self._backend == 'rapid' or (
            self._backend == 'auto' and str(self._model_path).lower().endswith('.ckpt')
        ):
            self._backend = 'rapid'
            self._init_rapid_backend()
        elif self._backend in ('ultralytics', 'yolor') or (
            self._backend == 'auto' and str(self._model_path).lower().endswith(('.pt', '.engine'))
        ):
            self._init_pytorch_or_trt_backend()
        else:
            self._backend = 'onnx'
            self._init_onnx_backend()

    @classmethod
    def _infer_torch_device(cls, device: str, providers: Optional[List]) -> str:
        if device and device.lower() not in ('auto', ''):
            return device
        provider_names = [cls._provider_name(p) for p in (providers or [])]
        if 'CUDAExecutionProvider' in provider_names or 'TensorrtExecutionProvider' in provider_names:
            return 'cuda:0'
        return 'cpu'

    def _init_onnx_backend(self) -> None:
        onnxruntime.set_default_logger_severity(3)
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3

        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _old_stderr = os.dup(2)
        provider_stderr = StringIO()
        self._provider_init_stderr = ""
        os.dup2(_devnull_fd, 2)
        try:
            self._preload_openvino_runtime()
            with redirect_stderr(provider_stderr):
                self._interpreter = self._create_inference_session(
                    model_path=self._model_path,
                    sess_options=session_option,
                    providers=self._providers,
                )
        finally:
            os.dup2(_old_stderr, 2)
            os.close(_old_stderr)
            os.close(_devnull_fd)
            self._provider_init_stderr = provider_stderr.getvalue().strip()

        requested_provider_names = [
            self._provider_name(provider) for provider in (self._providers or [])
        ]
        self._providers = self._interpreter.get_providers()
        print(f'FisheyeDetector requested providers: {requested_provider_names}')
        print(f'FisheyeDetector enabled providers: {self._providers}')

        if 'OpenVINOExecutionProvider' in requested_provider_names:
            if 'OpenVINOExecutionProvider' in self._providers:
                print('OpenVINO is active for this session.')
            else:
                print('OpenVINO was requested but is not active; ONNX Runtime fell back to CPU.')

        self._pinto_style_graph = False
        try:
            onnx_graph: onnx.ModelProto = onnx.load(self._model_path)
            if len(onnx_graph.graph.node) > 0 and onnx_graph.graph.node[0].op_type == "Resize":
                first_resize_op: List[onnx.ValueInfoProto] = [
                    i for i in onnx_graph.graph.value_info if i.name == "prep/Resize_output_0"
                ]
                if first_resize_op:
                    self._input_shapes = [[d.dim_value for d in first_resize_op[0].type.tensor_type.shape.dim]]
                    self._pinto_style_graph = True
                else:
                    self._input_shapes = [inp.shape for inp in self._interpreter.get_inputs()]
            else:
                self._input_shapes = [inp.shape for inp in self._interpreter.get_inputs()]
        except Exception:
            self._input_shapes = [inp.shape for inp in self._interpreter.get_inputs()]

        self._input_names = [inp.name for inp in self._interpreter.get_inputs()]
        self._input_dtypes = [
            self._onnx_dtypes_to_np_dtypes.get(inp.type, np.float32)
            for inp in self._interpreter.get_inputs()
        ]
        self._output_shapes = [out.shape for out in self._interpreter.get_outputs()]
        self._output_names = [out.name for out in self._interpreter.get_outputs()]
        if self._output_names == ['x1y1x2y2', 'main01_scores']:
            self._pinto_style_graph = True
        self._model = self._interpreter.run

    def _resolve_input_hw(self) -> Tuple[int, int]:
        if self._input_shapes and len(self._input_shapes[0]) >= 4:
            h_dim = self._input_shapes[0][self._h_index]
            w_dim = self._input_shapes[0][self._w_index]
            h = int(h_dim) if isinstance(h_dim, (int, np.integer)) and int(h_dim) > 0 else self._default_img_size
            w = int(w_dim) if isinstance(w_dim, (int, np.integer)) and int(w_dim) > 0 else self._default_img_size
            return h, w
        return self._default_img_size, self._default_img_size

    @staticmethod
    def _ensure_yolor_on_sys_path() -> None:
        env_dir = os.getenv('AICITY_YOLOR_DIR')
        candidates = []
        if env_dir:
            candidates.append(Path(env_dir))
        fisheye_dir = Path(__file__).resolve().parents[1]
        repo_root = fisheye_dir.parent
        candidates.extend([
            fisheye_dir / 'third_party' / 'YoloR',
            repo_root.parent / 'UT_AICITY2025_Track4' / 'train' / 'YoloR',
            repo_root / 'UT_AICITY2025_Track4' / 'train' / 'YoloR',
        ])
        for cand in candidates:
            if cand.is_dir():
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
                yolor_utils = str(cand / 'utils')
                if 'utils' in sys.modules and hasattr(sys.modules['utils'], '__path__'):
                    if yolor_utils not in sys.modules['utils'].__path__:
                        sys.modules['utils'].__path__.append(yolor_utils)

    @staticmethod
    def _ensure_ultralytics_on_sys_path(prefer_v10: bool = False) -> None:
        if importlib.util.find_spec('ultralytics') is not None:
            return
        fisheye_dir = Path(__file__).resolve().parents[1]
        repo_root = fisheye_dir.parent
        # Note: YoloV13 fork supports both YOLOv13 and YOLOv10 checkpoints
        sub = 'YoloV10' if prefer_v10 else 'YoloV13'
        candidates = [
            fisheye_dir / 'third_party' / sub,
            fisheye_dir / 'third_party' / 'YoloV13',
            repo_root.parent / 'UT_AICITY2025_Track4' / 'train' / sub,
            repo_root / 'UT_AICITY2025_Track4' / 'train' / sub,
        ]
        for cand in candidates:
            if (cand / 'ultralytics').is_dir() and str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
                break

    def _init_pytorch_or_trt_backend(self) -> None:
        model_name = Path(self._model_path).name.lower()
        if self._backend == 'yolor' or ('yolor' in model_name and self._backend == 'auto'):
            self._backend = 'yolor'
            self._ensure_yolor_on_sys_path()
            import torch
            from models.experimental import attempt_load as load_yolor  # type: ignore
            dev = torch.device(self._device)
            use_half = dev.type == 'cuda'
            self._yolor_dtype = torch.float16 if use_half else torch.float32
            self._yolor_device = dev
            self._yolor_model = load_yolor(self._model_path, map_location=dev).to(
                device=dev, dtype=self._yolor_dtype
            ).eval()
        else:
            self._backend = 'ultralytics'
            self._ensure_ultralytics_on_sys_path()
            from ultralytics import YOLO  # type: ignore
            self._ultra_model = YOLO(self._model_path)

    def _init_wbf_backend(self) -> None:
        self._ensure_yolor_on_sys_path()
        self._ensure_ultralytics_on_sys_path()
        import torch
        from models.experimental import attempt_load as load_yolor  # type: ignore
        from ultralytics import YOLO  # type: ignore

        dev = torch.device(self._device)
        use_half = dev.type == 'cuda'
        self._yolor_dtype = torch.float16 if use_half else torch.float32
        self._yolor_device = dev
        self._yolor_model = load_yolor(self._wbf_models_cfg['yolor'], map_location=dev).to(
            device=dev, dtype=self._yolor_dtype
        ).eval()
        self._y10_model = YOLO(self._wbf_models_cfg['y10'])
        self._y13_model = YOLO(self._wbf_models_cfg['y13'])

    def _init_rapid_backend(self) -> None:
        import types
        import torch

        fisheye_dir = Path(__file__).resolve().parents[1]
        repo_root = fisheye_dir.parent
        rapid_dir = Path(
            self._rapid_dir
            or os.getenv('RAPID_DIR')
            or (fisheye_dir / 'third_party' / 'RAPiD')
        )
        if not rapid_dir.is_dir() and (repo_root.parent / 'RAPiD').is_dir():
            rapid_dir = repo_root.parent / 'RAPiD'
        if not rapid_dir.is_dir() and (repo_root / 'RAPiD').is_dir():
            rapid_dir = repo_root / 'RAPiD'

        weights_path = Path(self._model_path)
        if not weights_path.exists():
            weights_path = rapid_dir / 'weights' / self._model_path

        # Stub pycocotools if not installed (we use exact polygon rotated_nms instead)
        if 'pycocotools' not in sys.modules:
            pc = types.ModuleType('pycocotools')
            pc.mask = types.ModuleType('pycocotools.mask')  # type: ignore[attr-defined]
            sys.modules['pycocotools'] = pc
            sys.modules['pycocotools.mask'] = pc.mask  # type: ignore[attr-defined]

        if str(rapid_dir) not in sys.path:
            sys.path.insert(0, str(rapid_dir))

        if self._default_img_size == 1280:
            self._default_img_size = 1024  # RAPiD default input size

        from models.rapid_export import RAPiD  # type: ignore

        dev_str = 'cuda' if str(self._device).startswith('cuda') else 'cpu'
        self._rapid_device = dev_str
        self._rapid_model = RAPiD(input_hw=(self._default_img_size, self._default_img_size))
        state = torch.load(str(weights_path), map_location='cpu', weights_only=True)['model']
        self._rapid_model.load_state_dict(state)
        self._rapid_model.eval().to(dev_str)

    def _resolve_person_class_ids(self, num_classes: Optional[int] = None) -> set[int]:
        if self._explicit_person_class_ids is not None:
            return self._explicit_person_class_ids
        if num_classes == 5:
            # Fisheye8K / VisDrone 5-class dataset: 3 = Pedestrian
            return {self.FISHEYE8K_PEDESTRIAN_CLASS_ID}
        if num_classes is not None and num_classes >= 1:
            return {0}
        return {0, self.FISHEYE8K_PEDESTRIAN_CLASS_ID}

    def _normalize_classid(self, raw_class_id: int, person_classes: set[int]) -> int:
        if raw_class_id in person_classes:
            return 0
        # Ensure non-person classes do not collide with classid=0
        return raw_class_id if raw_class_id != 0 else -1

    def __call__(
        self,
        image: np.ndarray,
        disable_headpose_identification_mode: bool = True,
    ) -> List[Box]:
        if self._backend == 'wbf':
            return self._call_wbf(image)
        if self._backend == 'rapid':
            return self._call_rapid(image)
        if self._backend == 'ultralytics':
            return self._call_ultralytics(image)
        if self._backend == 'yolor':
            return self._call_yolor(image)
        return self._call_onnx(image, disable_headpose_identification_mode=disable_headpose_identification_mode)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(self._swap)
        return np.ascontiguousarray(image, dtype=np.float32)

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_headpose_identification_mode: bool = True,
    ) -> List[Box]:
        image_height, image_width = image.shape[:2]
        in_h, in_w = self._resolve_input_hw()
        result_boxes: List[Box] = []
        if len(boxes) == 0:
            return result_boxes

        scores = boxes[:, 2:3]
        keep_idxs = scores[:, 0] > self._obj_class_score_th
        scores_keep = scores[keep_idxs, :]
        boxes_keep = boxes[keep_idxs, :]

        for box, score in zip(boxes_keep, scores_keep):
            classid = int(box[1])
            x_min = int(np.clip(box[3], 0, in_w) * image_width / max(in_w, 1))
            y_min = int(np.clip(box[4], 0, in_h) * image_height / max(in_h, 1))
            x_max = int(np.clip(box[5], 0, in_w) * image_width / max(in_w, 1))
            y_max = int(np.clip(box[6], 0, in_h) * image_height / max(in_h, 1))
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            result_boxes.append(
                Box(
                    classid=classid,
                    score=float(score.item()),
                    x1=x_min,
                    y1=y_min,
                    x2=x_max,
                    y2=y_max,
                    cx=cx,
                    cy=cy,
                )
            )
        return result_boxes

    def _call_onnx(
        self,
        image: np.ndarray,
        disable_headpose_identification_mode: bool = True,
    ) -> List[Box]:
        temp_image = copy.deepcopy(image)
        img_h, img_w = temp_image.shape[:2]
        in_h, in_w = self._resolve_input_hw()

        if self._pinto_style_graph:
            if (img_h, img_w) != (in_h, in_w):
                proc_img = cv2.resize(temp_image, (in_w, in_h))
            else:
                proc_img = temp_image
            chw = self._preprocess(proc_img)
            inp = np.asarray([chw], dtype=self._input_dtypes[0])
            outputs = super().__call__(input_datas=[inp])
            if self._output_names == ['x1y1x2y2', 'main01_scores']:
                raw_boxes = nms_raw_outputs(outputs[0], outputs[1], iou_th=self._iou_th, score_th=self._obj_class_score_th)
            else:
                raw_boxes = outputs[0]
            boxes = self._postprocess(
                image=temp_image,
                boxes=raw_boxes,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
            person_classes = self._resolve_person_class_ids(num_classes=None)
            for b in boxes:
                if b.classid != 0 and b.classid in person_classes:
                    b.classid = 0
            return [b for b in boxes if b.classid not in (1, 2, 3, 4, 5, 6, 7, 8) or b.classid == 0]

        # Standard YOLO / Fisheye letterbox + RGB [0, 1] preprocessing
        lb_img, ratio, pad = letterbox_image(temp_image, new_shape=(in_h, in_w))
        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        inp = np.asarray([chw], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inp])
        return self._decode_yolo_onnx_outputs(outputs, (img_h, img_w), (in_h, in_w), ratio, pad)

    def _decode_yolo_onnx_outputs(
        self,
        outputs: List[np.ndarray],
        orig_hw: Tuple[int, int],
        in_hw: Tuple[int, int],
        ratio: Tuple[float, float],
        pad: Tuple[float, float],
    ) -> List[Box]:
        if len(outputs) == 2 and outputs[0].ndim == 3 and outputs[0].shape[-1] == 4:
            # Separate boxes [1, A, 4] and scores [1, C, A]
            num_classes = int(outputs[1].shape[1])
            person_classes = self._resolve_person_class_ids(num_classes)
            nms_out = nms_raw_outputs(
                outputs[0], outputs[1],
                iou_th=self._iou_th,
                score_th=self._obj_class_score_th,
            )
            if len(nms_out) == 0:
                return []
            xyxy = unscale_letterbox_coords(nms_out[:, 3:7], orig_hw, ratio, pad)
            return self._build_boxes_from_arrays(xyxy, nms_out[:, 2], nms_out[:, 1].astype(int), person_classes)

        out0 = np.asarray(outputs[0])
        # Case 1: Post-NMS [N, 7] -> [batch_idx, class_id, score, x1, y1, x2, y2] or [batch_idx, x1, y1, x2, y2, class_id, score]
        if out0.ndim == 2 and out0.shape[1] == 7:
            keep = out0[out0[:, 2] >= self._obj_class_score_th]
            if len(keep) == 0:
                return []
            person_classes = self._resolve_person_class_ids(None)
            xyxy = unscale_letterbox_coords(keep[:, 3:7], orig_hw, ratio, pad)
            return self._build_boxes_from_arrays(xyxy, keep[:, 2], keep[:, 1].astype(int), person_classes)

        # Case 2: End-to-End YOLOv10 ONNX [1, N, 6] ([x1, y1, x2, y2, score, class_id])
        #         OR RAPiD ONNX [1, A, 6] ([cx, cy, w, h, angle_deg, score])
        if (out0.ndim == 3 and out0.shape[-1] == 6) or (out0.ndim == 2 and out0.shape[-1] == 6):
            preds = out0[0] if out0.ndim == 3 else out0
            is_rapid_onnx = (
                'rapid' in Path(self._model_path).name.lower()
                or preds.shape[0] > 1000
            )
            if is_rapid_onnx:
                keep = preds[preds[:, 5] >= self._obj_class_score_th]
                if len(keep) == 0:
                    return []
                if len(keep) > 1000:
                    keep = keep[np.argsort(-keep[:, 5], kind='stable')[:1000]]
                dts = rotated_nms(keep.astype(np.float32), thres=self._iou_th)
                if len(dts) == 0:
                    return []
                results: List[Box] = []
                rx, ry = max(ratio[0], 1e-9), max(ratio[1], 1e-9)
                dw, dh = pad
                for cx_lb, cy_lb, w_lb, h_lb, a_deg, sc in dts:
                    if float(sc) < self._obj_class_score_th:
                        continue
                    cx = (float(cx_lb) - dw) / rx
                    cy = (float(cy_lb) - dh) / ry
                    w = float(w_lb) / rx
                    h = float(h_lb) / ry
                    x1, y1, x2, y2 = self._rot_to_aabb(cx, cy, w, h, float(a_deg), orig_hw[1], orig_hw[0])
                    results.append(
                        Box(
                            classid=0,
                            score=float(sc),
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            cx=int(round(cx)),
                            cy=int(round(cy)),
                            rbox=(cx, cy, w, h, float(a_deg)),
                        )
                    )
                return results

            keep = preds[preds[:, 4] >= self._obj_class_score_th]
            if len(keep) == 0:
                return []
            person_classes = self._resolve_person_class_ids(None)
            xyxy = keep[:, :4].copy()
            # Handle normalized [0..1] vs pixel coordinates
            if np.max(xyxy) <= 1.5:
                xyxy[:, [0, 2]] *= in_hw[1]
                xyxy[:, [1, 3]] *= in_hw[0]
            xyxy = unscale_letterbox_coords(xyxy, orig_hw, ratio, pad)
            return self._build_boxes_from_arrays(xyxy, keep[:, 4], keep[:, 5].astype(int), person_classes)

        # Case 3: YOLOv8 / YOLOv9 / YOLOv13 ([1, 4+C, A] or [1, A, 4+C]) or YOLOR ([1, A, 5+C])
        if out0.ndim == 3:
            pred = out0[0]
            # Transpose [4+C, A] -> [A, 4+C] when channels come first
            if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 128:
                pred = pred.T

            n_cols = pred.shape[1]
            cxcywh = pred[:, :4]
            x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2.0
            y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2.0
            x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2.0
            y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2.0
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)[None, :, :]  # [1, A, 4]

            # Detect YOLOR [cx, cy, w, h, obj_conf, cls_0, ..., cls_C-1] vs Ultralytics [cx, cy, w, h, cls_0, ..., cls_C-1]
            # For Fisheye8K 5 classes: Ultralytics has 4+5=9 cols, YOLOR has 5+5=10 cols.
            # For COCO 80 classes: Ultralytics has 84 cols, YOLOR has 85 cols.
            if n_cols in (10, 85):
                obj_conf = pred[:, 4:5]
                cls_scores = (obj_conf * pred[:, 5:]).T[None, :, :]  # [1, C, A]
            else:
                cls_scores = pred[:, 4:].T[None, :, :]  # [1, C, A]

            num_classes = int(cls_scores.shape[1])
            person_classes = self._resolve_person_class_ids(num_classes)
            nms_out = nms_raw_outputs(
                boxes_xyxy.astype(np.float32),
                cls_scores.astype(np.float32),
                iou_th=self._iou_th,
                score_th=self._obj_class_score_th,
            )
            if len(nms_out) == 0:
                return []
            xyxy = unscale_letterbox_coords(nms_out[:, 3:7], orig_hw, ratio, pad)
            return self._build_boxes_from_arrays(xyxy, nms_out[:, 2], nms_out[:, 1].astype(int), person_classes)

        return []

    def _build_boxes_from_arrays(
        self,
        xyxy: np.ndarray,
        scores: np.ndarray,
        raw_classes: np.ndarray,
        person_classes: set[int],
    ) -> List[Box]:
        results: List[Box] = []
        for (x1, y1, x2, y2), sc, raw_cls in zip(xyxy, scores, raw_classes):
            if float(sc) < self._obj_class_score_th:
                continue
            ix1, iy1, ix2, iy2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
            cx = (ix1 + ix2) // 2
            cy = (iy1 + iy2) // 2
            norm_cls = self._normalize_classid(int(raw_cls), person_classes)
            results.append(
                Box(
                    classid=norm_cls,
                    score=float(sc),
                    x1=ix1,
                    y1=iy1,
                    x2=ix2,
                    y2=iy2,
                    cx=cx,
                    cy=cy,
                )
            )
        return results

    def _call_ultralytics(self, image: np.ndarray) -> List[Box]:
        preds = self._ultra_model.predict(
            source=image,
            imgsz=self._default_img_size,
            conf=self._obj_class_score_th,
            iou=self._iou_th,
            device=self._device,
            verbose=False,
        )
        if not preds:
            return []
        r = preds[0]
        names = getattr(r, 'names', None) or getattr(self._ultra_model, 'names', {}) or {}
        person_classes = self._resolve_person_class_ids(len(names) if names else None)
        boxes_obj = r.boxes.cpu().numpy()
        return self._build_boxes_from_arrays(
            boxes_obj.xyxy,
            boxes_obj.conf,
            boxes_obj.cls.astype(int),
            person_classes,
        )

    def _call_yolor(self, image: np.ndarray) -> List[Box]:
        import torch

        img_h, img_w = image.shape[:2]
        sz = self._default_img_size
        lb_img, ratio, pad = letterbox_image(image, new_shape=(sz, sz))
        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        tensor = torch.from_numpy(chw).to(device=self._yolor_device, dtype=self._yolor_dtype).unsqueeze(0)
        with torch.no_grad():
            pred = self._yolor_model(tensor)[0]
        return self._decode_yolo_onnx_outputs(
            [pred.float().cpu().numpy()],
            (img_h, img_w),
            (sz, sz),
            ratio,
            pad,
        )

    def _call_wbf(self, image: np.ndarray) -> List[Box]:
        import torch

        img_h, img_w = image.shape[:2]
        sz = self._default_img_size
        lb_img, ratio, pad = letterbox_image(image, new_shape=(sz, sz))
        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        tensor_f32 = torch.from_numpy(chw).to(device=self._yolor_device, dtype=torch.float32).unsqueeze(0)
        tensor_yr = tensor_f32.to(dtype=self._yolor_dtype)

        with torch.no_grad():
            yr_raw = self._yolor_model(tensor_yr)[0].float().cpu().numpy()
        yr_pred = yr_raw[0]
        cxcywh = yr_pred[:, :4]
        x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2.0
        y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2.0
        x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2.0
        y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2.0
        yr_boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)[None, :, :]
        yr_cls_scores = (yr_pred[:, 4:5] * yr_pred[:, 5:]).T[None, :, :]
        yr_nms = nms_raw_outputs(
            yr_boxes_xyxy.astype(np.float32),
            yr_cls_scores.astype(np.float32),
            iou_th=self._wbf_ious[0],
            score_th=self._wbf_confs[0],
        )
        yr_boxes, yr_scores, yr_classes = [], [], []
        if len(yr_nms) > 0:
            yr_boxes = unscale_letterbox_coords(yr_nms[:, 3:7], (img_h, img_w), ratio, pad).tolist()
            yr_scores = yr_nms[:, 2].astype(float).tolist()
            yr_classes = yr_nms[:, 1].astype(int).tolist()

        def _extract_ultra(model_obj, conf_t: float, iou_t: float):
            res = model_obj.predict(
                source=tensor_f32, imgsz=sz, conf=conf_t, iou=iou_t,
                device=self._device, verbose=False,
            )
            b_list, s_list, c_list = [], [], []
            for r in res:
                b_np = r.boxes.xyxy.cpu().numpy()
                s_np = r.boxes.conf.cpu().numpy()
                c_np = r.boxes.cls.cpu().numpy().astype(int)
                keep = s_np >= conf_t
                if np.any(keep):
                    scaled = unscale_letterbox_coords(b_np[keep], (img_h, img_w), ratio, pad)
                    b_list.extend(scaled.tolist())
                    s_list.extend(s_np[keep].astype(float).tolist())
                    c_list.extend(c_np[keep].tolist())
            return b_list, s_list, c_list

        y10_boxes, y10_scores, y10_classes = _extract_ultra(self._y10_model, self._wbf_confs[1], self._wbf_ious[1])
        y13_boxes, y13_scores, y13_classes = _extract_ultra(self._y13_model, self._wbf_confs[2], self._wbf_ious[2])

        all_boxes = [yr_boxes, y10_boxes, y13_boxes]
        all_scores = [yr_scores, y10_scores, y13_scores]
        all_classes = [yr_classes, y10_classes, y13_classes]
        norm_boxes = [
            [[b[0] / img_w, b[1] / img_h, b[2] / img_w, b[3] / img_h] for b in boxes]
            for boxes in all_boxes
        ]
        fused_norm, fused_scores, fused_labels = weighted_boxes_fusion_np(
            norm_boxes, all_scores, all_classes,
            weights=self._wbf_weights,
            iou_thr=self._wbf_iou,
        )
        if len(fused_norm) == 0:
            return []
        fused_xyxy = fused_norm * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        person_classes = self._resolve_person_class_ids(5)
        return self._build_boxes_from_arrays(fused_xyxy, fused_scores, fused_labels, person_classes)

    @staticmethod
    def _rot_to_aabb(
        cx: float, cy: float, w: float, h: float, angle_deg: float, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))
        x1 = int(round(float(np.clip(pts[:, 0].min(), 0, img_w))))
        y1 = int(round(float(np.clip(pts[:, 1].min(), 0, img_h))))
        x2 = int(round(float(np.clip(pts[:, 0].max(), 0, img_w))))
        y2 = int(round(float(np.clip(pts[:, 1].max(), 0, img_h))))
        return x1, y1, x2, y2

    def _call_rapid(self, image: np.ndarray) -> List[Box]:
        import torch

        img_h, img_w = image.shape[:2]
        sz = self._default_img_size
        lb_img, ratio, pad = letterbox_image(image, new_shape=(sz, sz), color=(0, 0, 0))
        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        x = torch.from_numpy(chw).unsqueeze(0).to(self._rapid_device)
        with torch.no_grad():
            out = self._rapid_model(x).cpu().squeeze(0)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        out = out[out[:, 5] >= self._obj_class_score_th]
        if len(out) > 1000:
            out = out[torch.argsort(-out[:, 5])[:1000]]
        dts = rotated_nms(out.numpy(), thres=self._iou_th)
        if len(dts) == 0:
            return []
        rx, ry = max(ratio[0], 1e-9), max(ratio[1], 1e-9)
        dw, dh = pad
        results: List[Box] = []
        for cx_lb, cy_lb, w_lb, h_lb, a, s in dts.reshape(-1, 6):
            if float(s) < self._obj_class_score_th:
                continue
            cx = (float(cx_lb) - dw) / rx
            cy = (float(cy_lb) - dh) / ry
            w = float(w_lb) / rx
            h = float(h_lb) / ry
            x1, y1, x2, y2 = self._rot_to_aabb(cx, cy, w, h, float(a), img_w, img_h)
            icx = int(round(cx))
            icy = int(round(cy))
            results.append(
                Box(
                    classid=0,
                    score=float(s),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    cx=icx,
                    cy=icy,
                    rbox=(cx, cy, w, h, float(a)),
                )
            )
        return results


# Alias so `from utils.yolo import YOLOv9, Box, Color, is_package_installed` works identically to Head-Detector
YOLOv9 = FisheyeDetector


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


def is_package_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = max(int(dist / dash_length), 1)
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
    dash_length: int = 10,
):
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)
