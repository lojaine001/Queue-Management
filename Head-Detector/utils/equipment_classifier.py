"""
equipment_classifier.py — Equipment type detection for checkout-camera tracks.

Associates one of four equipment labels with each tracked person at the
checkout lanes.  Two backends are available:

  model   — Runs a custom ONNX object-detection model (YOLOv8-style) trained
             on in-store equipment classes.  Set `equipment_model` in
             config2.yml to the .onnx file path to activate this backend.

  heuristic — Colour/saturation analysis on the region below the tracked
               head bbox.  No model required; works out-of-the-box but
               needs per-camera threshold tuning.  Enable with
               `equipment_heuristic: true` in config2.yml.

If neither backend is enabled the classifier always returns EQUIPMENT_NONE
so the rest of the pipeline (DB storage, overlay) is already wired in and
ready for when you add a trained model.

Equipment labels
────────────────
  "trolley"       — full-size store shopping trolley
  "store_basket"  — store hand basket (plastic, handle)
  "personal_bag"  — personal bag/basket brought from home
  "none"          — no visible equipment

ONNX model contract
───────────────────
The optional model must output bounding boxes in the same format as YOLOv8:
  output shape  : (1, num_detections, 6)  OR  (num_detections, 6)
  columns       : [x_center, y_center, width, height, confidence, class_id]
  class mapping : 0 → trolley, 1 → store_basket, 2 → personal_bag
  input         : (1, 3, H, W)  normalised float32

To train such a model, label images from the checkout camera with the three
classes above and export to ONNX with `yolo export format=onnx`.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

# ── Equipment type constants ──────────────────────────────────────────────────
EQUIPMENT_NONE          = "none"
EQUIPMENT_TROLLEY       = "trolley"
EQUIPMENT_STORE_BASKET  = "store_basket"
EQUIPMENT_PERSONAL_BAG  = "personal_bag"

_CLASS_ID_TO_LABEL = {
    0: EQUIPMENT_TROLLEY,
    1: EQUIPMENT_STORE_BASKET,
    2: EQUIPMENT_PERSONAL_BAG,
}

# Colour used on the overlay per equipment type
EQUIPMENT_COLOURS = {
    EQUIPMENT_NONE:         (100, 100, 100),   # grey
    EQUIPMENT_TROLLEY:      (0,   200, 255),   # amber / cyan
    EQUIPMENT_STORE_BASKET: (50,  255, 50),    # green
    EQUIPMENT_PERSONAL_BAG: (200, 100, 255),   # purple
}

# Dwell multiplier per equipment type (feeds the wait-time model)
# Values from the project roadmap: trolley ≈ 35 min, basket ≈ 15 min
EQUIPMENT_DWELL_MULTIPLIER = {
    EQUIPMENT_NONE:         1.0,
    EQUIPMENT_TROLLEY:      2.3,   # ~35 min vs ~15 min base
    EQUIPMENT_STORE_BASKET: 1.0,   # baseline
    EQUIPMENT_PERSONAL_BAG: 0.7,   # quicker than basket
}


class EquipmentClassifier:
    """Classify shopping equipment carried by tracked heads at checkout."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_thresh: float = 0.40,
        heuristic_mode: bool = False,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        """
        Parameters
        ----------
        model_path    : Path to a YOLOv8-style ONNX model (optional).
        score_thresh  : Minimum confidence to accept a model detection.
        heuristic_mode: Fall back to colour analysis if no model loaded.
        input_size    : (W, H) expected by the ONNX model.
        """
        self.score_thresh  = score_thresh
        self.heuristic_mode = heuristic_mode
        self.input_size    = input_size
        self._session      = None

        if model_path and os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                print(f"[EQUIP] ONNX model loaded: {model_path}")
            except Exception as exc:
                print(f"[EQUIP] WARNING: Could not load equipment model — {exc}")
        elif model_path:
            print(f"[EQUIP] WARNING: equipment_model path not found: {model_path}")

        if self._session:
            self._backend = "model"
        elif heuristic_mode:
            self._backend = "heuristic"
            print("[EQUIP] Using colour-heuristic backend (threshold tuning may be needed)")
        else:
            self._backend = "none"
            print("[EQUIP] No equipment model configured — storing 'none' for all tracks")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def classify_tracks(
        self,
        frame: np.ndarray,
        track_bboxes: dict[int, tuple[int, int, int, int]],
    ) -> dict[int, str]:
        """Return an equipment label for each tracked person.

        Parameters
        ----------
        frame        : BGR image (H, W, 3).
        track_bboxes : Mapping of track_id → (x1, y1, x2, y2) head bbox.

        Returns
        -------
        dict mapping track_id → equipment label string.
        """
        if not track_bboxes:
            return {}

        if self._backend == "model":
            return self._classify_model(frame, track_bboxes)
        if self._backend == "heuristic":
            return self._classify_heuristic(frame, track_bboxes)
        return {tid: EQUIPMENT_NONE for tid in track_bboxes}

    # ─────────────────────────────────────────────────────────────────────────
    # Model backend
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_model(
        self,
        frame: np.ndarray,
        track_bboxes: dict[int, tuple[int, int, int, int]],
    ) -> dict[int, str]:
        """Run ONNX model on the full frame, then associate each detection
        to the nearest track by IoU overlap of the expanded person region."""
        w_in, h_in = self.input_size
        resized = cv2.resize(frame, (w_in, h_in))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception as exc:
            print(f"[EQUIP] Model inference error: {exc}")
            return {tid: EQUIPMENT_NONE for tid in track_bboxes}

        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]  # (num_dets, 6)

        fh, fw = frame.shape[:2]
        sx, sy = fw / w_in, fh / h_in

        detections: list[tuple[int, int, int, int, str]] = []  # x1,y1,x2,y2,label
        for det in raw:
            cx, cy, dw, dh, conf, cls_id = det[:6]
            if conf < self.score_thresh:
                continue
            label = _CLASS_ID_TO_LABEL.get(int(cls_id), EQUIPMENT_NONE)
            dx1 = int((cx - dw / 2) * sx)
            dy1 = int((cy - dh / 2) * sy)
            dx2 = int((cx + dw / 2) * sx)
            dy2 = int((cy + dh / 2) * sy)
            detections.append((dx1, dy1, dx2, dy2, label))

        result: dict[int, str] = {}
        for tid, (hx1, hy1, hx2, hy2) in track_bboxes.items():
            person_box = _expand_person_region(hx1, hy1, hx2, hy2, fh, fw)
            best_label = EQUIPMENT_NONE
            best_iou   = 0.0
            for dx1, dy1, dx2, dy2, label in detections:
                overlap = _box_iou(person_box, (dx1, dy1, dx2, dy2))
                if overlap > best_iou:
                    best_iou   = overlap
                    best_label = label
            result[tid] = best_label
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Heuristic backend
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_heuristic(
        self,
        frame: np.ndarray,
        track_bboxes: dict[int, tuple[int, int, int, int]],
    ) -> dict[int, str]:
        """Colour-saturation heuristic.

        Examines the region below the head bbox for metallic grey (trolley)
        or brightly-coloured plastic (store basket).  Thresholds are tuned
        for typical overhead checkout cameras — adjust via config if needed.

        NOTE: This is a starting heuristic.  For reliable classification,
        train a dedicated ONNX model on images from your specific camera.
        """
        fh, fw = frame.shape[:2]
        result: dict[int, str] = {}

        for tid, (hx1, hy1, hx2, hy2) in track_bboxes.items():
            head_h = max(hy2 - hy1, 1)
            head_w = max(hx2 - hx1, 1)

            # Region directly below the head — 3× head height, 2× head width
            rx1 = max(0,  hx1 - head_w // 2)
            ry1 = hy2
            rx2 = min(fw, hx2 + head_w // 2)
            ry2 = min(fh, hy2 + head_h * 3)

            if ry2 <= ry1 or rx2 <= rx1:
                result[tid] = EQUIPMENT_NONE
                continue

            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                result[tid] = EQUIPMENT_NONE
                continue

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            total = float(hsv.shape[0] * hsv.shape[1])

            # Metallic grey → trolley (low saturation, mid-to-high value)
            grey_mask    = cv2.inRange(hsv, (0, 0, 70), (180, 45, 210))
            grey_ratio   = float(np.count_nonzero(grey_mask)) / total

            # Bright coloured plastic → store basket (high saturation)
            colour_mask  = cv2.inRange(hsv, (0, 90, 70), (180, 255, 255))
            colour_ratio = float(np.count_nonzero(colour_mask)) / total

            if grey_ratio > 0.28:
                result[tid] = EQUIPMENT_TROLLEY
            elif colour_ratio > 0.18:
                result[tid] = EQUIPMENT_STORE_BASKET
            else:
                result[tid] = EQUIPMENT_NONE

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _expand_person_region(
    hx1: int, hy1: int, hx2: int, hy2: int,
    fh: int, fw: int,
) -> tuple[int, int, int, int]:
    """Expand head bbox to approximate the full person + equipment region."""
    head_h = hy2 - hy1
    head_w = hx2 - hx1
    px1 = max(0,  hx1 - head_w)
    py1 = hy1
    px2 = min(fw, hx2 + head_w)
    py2 = min(fh, hy2 + head_h * 4)
    return px1, py1, px2, py2


def _box_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0
