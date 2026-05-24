from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker_base import KalmanBoxTracker, TrackerDetection, TrackedObject, iou_batch


def _associate_clean(
    tracks: List[KalmanBoxTracker],
    detections: List[TrackerDetection],
    iou_threshold: float,
) -> tuple[List[tuple[int, int]], List[int], List[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    trk_boxes = np.array([t.get_state() for t in tracks])
    det_boxes = np.array([d.bbox for d in detections])

    iou_mat = iou_batch(trk_boxes, det_boxes)
    cost_mat = 1.0 - iou_mat

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched: List[tuple[int, int]] = []
    rejected_trk: List[int] = []
    rejected_det: List[int] = []

    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            matched.append((r, c))
        else:
            rejected_trk.append(r)
            rejected_det.append(c)

    matched_trk_set = {r for r, _ in matched}
    matched_det_set = {c for _, c in matched}

    unmatched_trk = rejected_trk + [
        i for i in range(len(tracks)) if i not in matched_trk_set and i not in rejected_trk
    ]
    unmatched_det = rejected_det + [
        j for j in range(len(detections)) if j not in matched_det_set and j not in rejected_det
    ]

    return matched, unmatched_trk, unmatched_det


class SORTTracker:
    def __init__(
        self,
        max_age: int = 9,
        min_hits: int = 0,
        iou_threshold: float = 0.33,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[KalmanBoxTracker] = []
        self._frame_count: int = 0

    def update(
        self,
        detections: Optional[List[TrackerDetection]] = None,
    ) -> List[TrackedObject]:
        if detections is None:
            detections = []

        self._frame_count += 1

        for trk in self._tracks:
            trk.predict()

        matched, _, unmatched_det = _associate_clean(
            self._tracks, detections, self.iou_threshold
        )

        for trk_idx, det_idx in matched:
            self._tracks[trk_idx].update(
                detections[det_idx].bbox,
                data=detections[det_idx].data,
            )

        for det_idx in unmatched_det:
            detection = detections[det_idx]
            self._tracks.append(KalmanBoxTracker(detection.bbox, data=detection.data))

        self._tracks = [
            track for track in self._tracks if track.time_since_update <= self.max_age
        ]

        output: List[TrackedObject] = []
        for track in self._tracks:
            if track.hits > self.min_hits:
                output.append(
                    TrackedObject(
                        global_id=track.global_id,
                        bbox=track.get_state(),
                        age=track.age,
                        data=track.data,
                    )
                )

        return output


SortTracker = SORTTracker