"""
tracker_base.py
───────────────
Shared Kalman filter, coordinate utilities, and unified output types used by
both the SORT and ByteTrack tracker backends.

No external dependencies beyond NumPy (already required by the project).
SciPy is only used in tracker_sort.py / tracker_bytetrack.py for the
Hungarian algorithm (scipy.optimize.linear_sum_assignment).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ── Coordinate utilities ───────────────────────────────────────────────────────

def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to Kalman measurement [cx, cy, s, r].

    cx, cy = centre coordinates
    s      = area  (w * h), clamped to ≥ 1 to avoid log(0) issues
    r      = aspect ratio (w / h)

    Returns shape (4, 1).
    """
    w  = float(bbox[2] - bbox[0])
    h  = float(bbox[3] - bbox[1])
    cx = float(bbox[0]) + w / 2.0
    cy = float(bbox[1]) + h / 2.0
    s  = max(w * h, 1.0)
    r  = w / max(h, 1e-6)
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert Kalman state [cx, cy, s, r, ...] back to [x1, y1, x2, y2]."""
    cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    s = max(s, 1.0)
    r = max(r, 1e-6)
    w  = float(np.sqrt(s * r))
    h  = s / w
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bounding boxes.

    Args:
        boxes_a : (N, 4)  [x1, y1, x2, y2]
        boxes_b : (M, 4)  [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix — values in [0, 1].
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ── Kalman filter ──────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """Single bounding-box track modelled with a constant-velocity Kalman filter.

    This is the same model used in the original SORT paper (Bewley 2016).

    State vector  x = [cx, cy, s, r, vx, vy, vs, vr]ᵀ
    Measurement   z = [cx, cy, s, r]ᵀ

    cx, cy = centre; s = area; r = aspect ratio; v* = their velocities.

    Class-level `_global_count` acts as a monotonically-increasing ID source
    shared across ALL KalmanBoxTracker instances (same semantics as Norfair's
    global_id counter).
    """

    _global_count: int = 0

    @classmethod
    def reset_count(cls) -> None:
        """Reset the global ID counter (useful between test runs)."""
        cls._global_count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.global_id: int = KalmanBoxTracker._global_count
        KalmanBoxTracker._global_count += 1

        self.data: Optional[Dict[str, Any]] = data
        self.hits: int = 0             # total matched detections
        self.hit_streak: int = 0       # consecutive frames with a match
        self.age: int = 0              # total frames since creation
        self.time_since_update: int = 0  # frames since last matched detection

        n, m = 8, 4   # state dim, measurement dim

        # ── State transition F (constant velocity: position += velocity * dt) ─
        self.F = np.eye(n, dtype=np.float64)
        self.F[:m, m:] = np.eye(m, dtype=np.float64)

        # ── Measurement matrix H ──────────────────────────────────────────────
        self.H = np.zeros((m, n), dtype=np.float64)
        self.H[:m, :m] = np.eye(m, dtype=np.float64)

        # ── Measurement noise R ───────────────────────────────────────────────
        # Scale/aspect-ratio noise is higher than position noise.
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        # ── Initial error covariance P ────────────────────────────────────────
        # Velocity components start with large uncertainty.
        self.P = np.eye(n, dtype=np.float64) * 10.0
        self.P[m:, m:] *= 100.0   # high velocity uncertainty at birth

        # ── Process noise Q ───────────────────────────────────────────────────
        self.Q = np.eye(n, dtype=np.float64)
        self.Q[m:, m:] *= 0.01    # smooth velocity evolution

        # ── Initial state (zero velocity, position from first detection) ──────
        self.x = np.zeros((n, 1), dtype=np.float64)
        self.x[:m] = bbox_to_z(bbox)

    # ── Kalman steps ───────────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Advance the filter one time-step.

        Clamps area velocity so area never goes negative, then applies
        F and updates P.  Returns the predicted bbox [x1, y1, x2, y2].
        """
        if self.x[2, 0] + self.x[6, 0] <= 0.0:
            self.x[6, 0] = 0.0           # stop area from going negative

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0          # streak breaks on any missed frame
        return z_to_bbox(self.x.flatten())

    def update(
        self,
        bbox: np.ndarray,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Assimilate a new matched detection into the Kalman filter.

        Uses the Joseph-form covariance update for numerical stability.
        """
        z = bbox_to_z(bbox)
        S = self.H @ self.P @ self.H.T + self.R          # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)      # state update
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.data = data

    def get_state(self) -> np.ndarray:
        """Return the current bbox estimate [x1, y1, x2, y2]."""
        return z_to_bbox(self.x.flatten())


# ── Unified tracker I/O types ──────────────────────────────────────────────────

class _LastDetection:
    """Minimal stand-in for norfair.Detection exposing only .data.

    main.py accesses tracked objects via ``obj.last_detection.data``; this
    class satisfies that contract without depending on norfair.
    """
    __slots__ = ("data",)

    def __init__(self, data: Optional[Dict[str, Any]]) -> None:
        self.data = data


class TrackedObject:
    """Output produced by SORT / ByteTrack trackers.

    Exposes the same attributes that main.py reads from norfair TrackedObject
    instances so that the rest of the pipeline is completely tracker-agnostic:

        .global_id       — unique monotonic integer ID  (same as norfair)
        .estimate        — np.ndarray shape (2, 2): [[x1,y1],[x2,y2]]
        .last_detection  — object with .data dict (or None)
        .age             — frames alive  (used as age / expect_fps)
    """
    __slots__ = ("global_id", "estimate", "last_detection", "age")

    def __init__(
        self,
        global_id: int,
        bbox: np.ndarray,
        age: int,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.global_id      = global_id
        self.estimate       = np.array(
            [[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float64
        )
        self.last_detection = _LastDetection(data)
        self.age            = age


class TrackerDetection:
    """Detection fed into SORT / ByteTrack.

    Analogous to norfair.Detection.  Carries the same ``data`` payload dict
    that main.py attaches (class_id, confidence, roi_idx, head_direction).
    """
    __slots__ = ("bbox", "score", "data")

    def __init__(
        self,
        bbox,
        score: float,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bbox  = np.asarray(bbox, dtype=np.float64)   # [x1, y1, x2, y2]
        self.score = float(score)
        self.data  = data
