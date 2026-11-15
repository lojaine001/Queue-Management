"""
tracker_sort.py — SORT tracker backend
───────────────────────────────────────
SORT: Simple Online and Realtime Tracking
Paper: Bewley et al. (2016)  https://arxiv.org/abs/1602.00763

Algorithm summary
─────────────────
1. Predict all existing tracks one step forward with the Kalman filter.
2. Build a cost matrix: cost[i][j] = 1 − IoU(track_i, detection_j).
3. Solve the linear assignment problem (Hungarian algorithm) to find the
   globally optimal set of track–detection matches.
4. Accept matches only when IoU ≥ iou_threshold.
5. Update matched tracks with the Kalman measurement step.
6. Initialise a brand-new track for every unmatched detection.
7. Delete tracks that have not been matched for > max_age frames.
8. Return all tracks that were matched this frame (time_since_update == 0)
   and have been seen at least min_hits times.

Dependencies
────────────
    numpy  ≥ 1.23   (already in requirements.txt)
    scipy  ≥ 1.10   (already in requirements.txt — used only for
                     scipy.optimize.linear_sum_assignment)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.tracker_base import (
    KalmanBoxTracker,
    TrackedObject,
    TrackerDetection,
    iou_batch,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _associate(
    tracks: List[KalmanBoxTracker],
    detections: List[TrackerDetection],
    iou_threshold: float,
) -> tuple[List[tuple[int, int]], List[int], List[int]]:
    """Hungarian IoU matching between predicted tracks and new detections.

    Returns
    -------
    matched       : list of (track_index, detection_index) pairs
    unmatched_trk : track indices that have no matching detection
    unmatched_det : detection indices that have no matching track
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    trk_boxes = np.array([t.get_state() for t in tracks])   # (N, 4)
    det_boxes = np.array([d.bbox for d in detections])       # (M, 4)

    iou_mat  = iou_batch(trk_boxes, det_boxes)               # (N, M)
    cost_mat = 1.0 - iou_mat                                 # minimise → maximise IoU

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched: List[tuple[int, int]] = []
    unmatched_trk: List[int] = []
    unmatched_det: List[int] = []

    matched_trk_set: set = set()
    matched_det_set: set = set()

    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            matched.append((r, c))
            matched_trk_set.add(r)
            matched_det_set.add(c)
        else:
            # Hungarian found a pair but IoU is too low — treat as unmatched
            unmatched_trk.append(r)
            unmatched_det.append(c)

    # Add indices not touched by the assignment at all
    for i in range(len(tracks)):
        if i not in matched_trk_set and i not in {r for r, _ in [(x, y) for x, y in zip(row_ind, col_ind) if iou_mat[x, y] < iou_threshold]}:
            if i not in matched_trk_set:
                unmatched_trk.append(i)

    for j in range(len(detections)):
        if j not in matched_det_set and j not in {c for _, c in [(x, y) for x, y in zip(row_ind, col_ind) if iou_mat[x, y] < iou_threshold]}:
            if j not in matched_det_set:
                unmatched_det.append(j)

    return matched, unmatched_trk, unmatched_det


def _associate_clean(
    tracks: List[KalmanBoxTracker],
    detections: List[TrackerDetection],
    iou_threshold: float,
) -> tuple[List[tuple[int, int]], List[int], List[int]]:
    """Clean Hungarian IoU matching (replaces _associate above)."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    trk_boxes = np.array([t.get_state() for t in tracks])
    det_boxes = np.array([d.bbox for d in detections])

    iou_mat  = iou_batch(trk_boxes, det_boxes)
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

    unmatched_trk = rejected_trk + [i for i in range(len(tracks))     if i not in matched_trk_set and i not in rejected_trk]
    unmatched_det = rejected_det + [j for j in range(len(detections))  if j not in matched_det_set and j not in rejected_det]

    return matched, unmatched_trk, unmatched_det


# ── Public tracker class ───────────────────────────────────────────────────────

class SORTTracker:
    """SORT drop-in replacement for norfair.Tracker.

    Parameters
    ----------
    max_age : int
        Frames a track survives without a detection before it is deleted.
        Maps to norfair's ``hit_counter_max``.
    min_hits : int
        Minimum number of detection matches a new track must accumulate
        before it appears in the output.  0 = report from the very first frame.
        Maps to norfair's ``initialization_delay``.
    iou_threshold : float
        Minimum IoU required to accept a track–detection assignment.
        Derived from config2.yml ``max_distance_between_points`` as
        ``1 / max_distance_between_points`` (because norfair uses IoU-distance
        = 1/IoU as its distance metric).

    Usage in main.py
    ----------------
        tracker = SORTTracker(max_age=9, min_hits=0, iou_threshold=0.33)
        tracked_objects = tracker.update(detections=frame_detections)
        # tracked_objects[i].global_id  — int
        # tracked_objects[i].estimate   — np.ndarray [[x1,y1],[x2,y2]]
        # tracked_objects[i].last_detection.data — dict
        # tracked_objects[i].age        — int
    """

    def __init__(
        self,
        max_age: int = 9,
        min_hits: int = 0,
        iou_threshold: float = 0.33,
    ) -> None:
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[KalmanBoxTracker] = []
        self._frame_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        detections: Optional[List[TrackerDetection]] = None,
    ) -> List[TrackedObject]:
        """Process one frame and return the list of active tracked objects.

        Call signature intentionally mirrors ``norfair.Tracker.update``:

            tracked_objects = tracker.update(detections=frame_detections)

        Parameters
        ----------
        detections : list[TrackerDetection] | None
            Detections for the current frame.  Pass an empty list or None
            for frames with no detections (tracks will coast on Kalman prediction).

        Returns
        -------
        list[TrackedObject]
            Active confirmed tracks for this frame.  Each object exposes the
            same attributes as a norfair TrackedObject.
        """
        if detections is None:
            detections = []

        self._frame_count += 1

        # ── Step 1: Predict ────────────────────────────────────────────────────
        for trk in self._tracks:
            trk.predict()

        # ── Step 2: Associate ──────────────────────────────────────────────────
        matched, unmatched_trk, unmatched_det = _associate_clean(
            self._tracks, detections, self.iou_threshold
        )

        # ── Step 3: Update matched tracks ─────────────────────────────────────
        for trk_idx, det_idx in matched:
            self._tracks[trk_idx].update(
                detections[det_idx].bbox,
                data=detections[det_idx].data,
            )

        # ── Step 4: Initialise new tracks ─────────────────────────────────────
        for det_idx in unmatched_det:
            d = detections[det_idx]
            self._tracks.append(KalmanBoxTracker(d.bbox, data=d.data))

        # ── Step 5: Delete stale tracks ────────────────────────────────────────
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.max_age
        ]

        # ── Step 6: Build output ───────────────────────────────────────────────
        # Return all alive tracks that have enough hits (matched or coasting).
        # This mirrors norfair's behaviour: a track stays in the output for up
        # to max_age frames after its last match so the death-detection logic in
        # main.py only fires once, when the track truly expires.
        output: List[TrackedObject] = []
        for trk in self._tracks:
            if trk.hits > self.min_hits:
                output.append(
                    TrackedObject(
                        global_id=trk.global_id,
                        bbox=trk.get_state(),
                        age=trk.age,
                        data=trk.data,
                    )
                )

        return output
