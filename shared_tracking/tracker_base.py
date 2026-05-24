from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    cx = float(bbox[0]) + w / 2.0
    cy = float(bbox[1]) + h / 2.0
    s = max(w * h, 1.0)
    r = w / max(h, 1e-6)
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    s = max(s, 1.0)
    r = max(r, 1e-6)
    w = float(np.sqrt(s * r))
    h = s / w
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class KalmanBoxTracker:
    _global_count: int = 0

    @classmethod
    def reset_count(cls) -> None:
        cls._global_count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.global_id: int = KalmanBoxTracker._global_count
        KalmanBoxTracker._global_count += 1

        self.data: Optional[Dict[str, Any]] = data
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0
        self.time_since_update: int = 0

        n, m = 8, 4

        self.F = np.eye(n, dtype=np.float64)
        self.F[:m, m:] = np.eye(m, dtype=np.float64)

        self.H = np.zeros((m, n), dtype=np.float64)
        self.H[:m, :m] = np.eye(m, dtype=np.float64)

        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        self.P = np.eye(n, dtype=np.float64) * 10.0
        self.P[m:, m:] *= 100.0

        self.Q = np.eye(n, dtype=np.float64)
        self.Q[m:, m:] *= 0.01

        self.x = np.zeros((n, 1), dtype=np.float64)
        self.x[:m] = bbox_to_z(bbox)

    def predict(self) -> np.ndarray:
        if self.x[2, 0] + self.x[6, 0] <= 0.0:
            self.x[6, 0] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        return z_to_bbox(self.x.flatten())

    def update(
        self,
        bbox: np.ndarray,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        z = bbox_to_z(bbox)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        i_kh = np.eye(8) - K @ self.H
        self.P = i_kh @ self.P @ i_kh.T + K @ self.R @ K.T

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.data = data

    def get_state(self) -> np.ndarray:
        return z_to_bbox(self.x.flatten())


class _LastDetection:
    __slots__ = ('data',)

    def __init__(self, data: Optional[Dict[str, Any]]) -> None:
        self.data = data


class TrackedObject:
    __slots__ = ('global_id', 'estimate', 'last_detection', 'age')

    def __init__(
        self,
        global_id: int,
        bbox: np.ndarray,
        age: int,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.global_id = global_id
        self.estimate = np.array(
            [[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float64
        )
        self.last_detection = _LastDetection(data)
        self.age = age


class TrackerDetection:
    __slots__ = ('bbox', 'score', 'data')

    def __init__(
        self,
        bbox,
        score: float,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float64)
        self.score = float(score)
        self.data = data