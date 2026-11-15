"""
owlv2_detector.py — Zero-shot caddy/basket detection via OWLv2.

Runs Google's OWLv2 vision-language model on checkout camera frames to
detect shopping equipment without any training data.  Designed to run at
very low frame rate (every N frames) so it does not impact the main
YOLOv9 head-detection pipeline.

The public API mirrors EquipmentClassifier.classify_tracks() so it can
be used as a drop-in upgrade: same input (frame + track bboxes), same
output (dict of track_id → equipment label string).

Text queries → equipment labels
────────────────────────────────
  "shopping trolley" / "supermarket cart"  → "trolley"
  "shopping basket"  / "hand basket"       → "store_basket"
  "handbag" / "backpack" / "tote bag"      → "personal_bag"

Usage
─────
    detector = OWLv2Detector()
    # call every N frames — NOT every frame
    if frame_counter % owlv2_interval == 0:
        track_equipment.update(detector.classify_tracks(frame, track_bboxes))
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from .equipment_classifier import (
    EQUIPMENT_NONE,
    EQUIPMENT_PERSONAL_BAG,
    EQUIPMENT_STORE_BASKET,
    EQUIPMENT_TROLLEY,
    _box_iou,
    _expand_person_region,
)

# ── Text queries grouped by target label ─────────────────────────────────────
# OWLv2 scores each query independently; we take the highest-scoring query
# per detection and map it to the equipment label for that group.
_QUERY_GROUPS: dict[str, list[str]] = {
    EQUIPMENT_TROLLEY:      ["shopping trolley", "supermarket trolley", "shopping cart",
                             "grocery cart", "metal trolley"],
    EQUIPMENT_STORE_BASKET: ["shopping basket", "hand basket", "wire basket",
                             "plastic basket", "carry basket"],
    EQUIPMENT_PERSONAL_BAG: ["handbag", "backpack", "shoulder bag", "tote bag",
                             "crossbody bag", "canvas bag", "leather bag", "purse",
                             "satchel", "cloth bag", "personal bag", "reusable bag"],
}

# Flat list of all queries (order matters — index used for label lookup)
_ALL_QUERIES: list[str] = []
_QUERY_TO_LABEL: dict[int, str] = {}
for _label, _queries in _QUERY_GROUPS.items():
    for _q in _queries:
        _QUERY_TO_LABEL[len(_ALL_QUERIES)] = _label
        _ALL_QUERIES.append(_q)

_MODEL_ID = "google/owlv2-base-patch16-ensemble"


class OWLv2Detector:
    """Zero-shot equipment detector using OWLv2.

    Parameters
    ----------
    score_thresh  : Minimum OWLv2 confidence to accept a detection (0–1).
                    Lower = more detections but more false positives.
                    Recommended range: 0.10 – 0.25 for checkout cameras.
    device        : "cuda" (GPU) or "cpu".  GPU strongly recommended.
    model_id      : HuggingFace model identifier.
    """

    def __init__(
        self,
        score_thresh: float = 0.15,
        device: Optional[str] = None,
        model_id: str = _MODEL_ID,
    ) -> None:
        self.score_thresh = score_thresh
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._model_id = model_id
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> bool:
        if self._loaded:
            return True
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            print(f"[OWLv2] Loading {self._model_id} on {self.device} ...")
            self._processor = Owlv2Processor.from_pretrained(self._model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self._model_id)
            self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            print(f"[OWLv2] Model ready on {self.device}")
            return True
        except Exception as exc:
            print(f"[OWLv2] WARNING: Could not load model — {exc}")
            return False

    # ── Public API (mirrors EquipmentClassifier) ──────────────────────────────

    def classify_tracks(
        self,
        frame: np.ndarray,
        track_bboxes: dict[int, tuple[int, int, int, int]],
    ) -> dict[int, str]:
        """Detect equipment in `frame` and assign labels to each tracked person.

        Parameters
        ----------
        frame        : BGR image (H, W, 3) — use the raw frame, not the
                       annotated one, for cleaner detection.
        track_bboxes : Mapping of track_id → (x1, y1, x2, y2) head bbox.

        Returns
        -------
        dict mapping track_id → equipment label string.
        Returns {track_id: EQUIPMENT_NONE} for all tracks on any error.
        """
        if not track_bboxes:
            return {}

        if not self._load():
            return {tid: EQUIPMENT_NONE for tid in track_bboxes}

        detections = self._run_owlv2(frame)
        return self._assign_to_tracks(frame, track_bboxes, detections)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_owlv2(
        self,
        frame: np.ndarray,
    ) -> list[tuple[int, int, int, int, str, float]]:
        """Run OWLv2 and return (x1, y1, x2, y2, label, score) detections."""
        try:
            # OWLv2 expects RGB PIL image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image as _PILImage
            pil_img = _PILImage.fromarray(rgb)

            inputs = self._processor(
                text=[_ALL_QUERIES],
                images=pil_img,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process: scale boxes back to pixel coordinates
            target_sizes = torch.tensor([pil_img.size[::-1]])  # (H, W)
            results = self._processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                score_threshold=self.score_thresh,
            )[0]

            fh, fw = frame.shape[:2]
            detections: list[tuple[int, int, int, int, str, float]] = []
            for box, score, label_idx in zip(
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy(),
            ):
                x1, y1, x2, y2 = (
                    int(max(0, box[0])),
                    int(max(0, box[1])),
                    int(min(fw, box[2])),
                    int(min(fh, box[3])),
                )
                label = _QUERY_TO_LABEL.get(int(label_idx), EQUIPMENT_NONE)
                detections.append((x1, y1, x2, y2, label, float(score)))

            return detections

        except Exception as exc:
            print(f"[OWLv2] Inference error: {exc}")
            return []

    def _assign_to_tracks(
        self,
        frame: np.ndarray,
        track_bboxes: dict[int, tuple[int, int, int, int]],
        detections: list[tuple[int, int, int, int, str, float]],
    ) -> dict[int, str]:
        """Associate each detection with the nearest tracked person via IoU."""
        fh, fw = frame.shape[:2]
        result: dict[int, str] = {}

        for tid, (hx1, hy1, hx2, hy2) in track_bboxes.items():
            person_box = _expand_person_region(hx1, hy1, hx2, hy2, fh, fw)
            best_label = EQUIPMENT_NONE
            best_score = 0.0
            for x1, y1, x2, y2, label, score in detections:
                overlap = _box_iou(person_box, (x1, y1, x2, y2))
                if overlap > 0.05 and score > best_score:
                    best_score = score
                    best_label = label
            result[tid] = best_label

        return result