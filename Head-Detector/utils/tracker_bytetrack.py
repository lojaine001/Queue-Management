"""
tracker_bytetrack.py — ByteTrack tracker backend
──────────────────────────────────────────────────
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Paper: Zhang et al. (2022)  https://arxiv.org/abs/2110.06864

Key idea vs SORT
─────────────────
SORT discards low-confidence detections before matching.  ByteTrack keeps
them and runs a *two-pass* association:

  Pass 1 — match TRACKED tracks  ↔  HIGH-confidence detections  (standard)
  Pass 2 — match remaining TRACKED tracks ↔ LOW-confidence detections
            (the "byte" insight: a real occluded object produces weak but
            real detections; using them prevents premature track death)
  Pass 3 — match LOST tracks ↔ remaining high-confidence detections
            (re-identify objects that recently went missing)

Track states
────────────
  TRACKED  — currently matched and active; returned in output
  LOST     — not matched recently; kept alive for up to max_age frames
  REMOVED  — expired; pruned from all lists

Dependencies
────────────
    numpy  ≥ 1.23
    scipy  ≥ 1.10
"""
from __future__ import annotations

from enum import IntEnum
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.tracker_base import (
    KalmanBoxTracker,
    TrackedObject,
    TrackerDetection,
    iou_batch,
)


# ── Track state ────────────────────────────────────────────────────────────────

class _State(IntEnum):
    TRACKED = 0
    LOST    = 1
    REMOVED = 2


# ── Extended single-track class ────────────────────────────────────────────────

class _BYTETrack(KalmanBoxTracker):
    """KalmanBoxTracker with ByteTrack state management."""

    def __init__(
        self,
        bbox: np.ndarray,
        score: float,
        data=None,
    ) -> None:
        super().__init__(bbox, data=data)
        self.score        = score
        self.state        = _State.TRACKED
        self.is_activated = False      # True once the track passes min_hits
        self.start_frame  = 0          # frame_id when activated

    # ── State transitions ──────────────────────────────────────────────────────

    def activate(self, frame_id: int) -> None:
        """Mark track as active (called when a brand-new track is created)."""
        self.is_activated = True
        self.start_frame  = frame_id
        self.state        = _State.TRACKED

    def re_activate(
        self,
        bbox: np.ndarray,
        score: float,
        data=None,
    ) -> None:
        """Re-activate a LOST track that was just re-matched."""
        self.update(bbox, data=data)   # Kalman update resets time_since_update
        self.score        = score
        self.state        = _State.TRACKED
        self.is_activated = True

    def mark_lost(self) -> None:
        self.state = _State.LOST

    def mark_removed(self) -> None:
        self.state = _State.REMOVED


# ── IoU matching helper ────────────────────────────────────────────────────────

def _iou_match(
    tracks: List[_BYTETrack],
    detections: List[TrackerDetection],
    iou_threshold: float,
) -> tuple[List[tuple[int, int]], List[int], List[int]]:
    """Hungarian IoU matching shared by all three passes.

    Returns
    -------
    matched       : (track_idx, det_idx) pairs
    unmatched_trk : track indices with no valid match
    unmatched_det : detection indices with no valid match
    """
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

    matched_trk = {r for r, _ in matched}
    matched_det = {c for _, c in matched}

    unmatched_trk = rejected_trk + [i for i in range(len(tracks))
                                     if i not in matched_trk and i not in rejected_trk]
    unmatched_det = rejected_det + [j for j in range(len(detections))
                                     if j not in matched_det and j not in rejected_det]

    return matched, unmatched_trk, unmatched_det


# ── Public tracker class ───────────────────────────────────────────────────────

class ByteTracker:
    """ByteTrack drop-in replacement for norfair.Tracker.

    Parameters
    ----------
    max_age : int
        Frames a LOST track survives before deletion.
        Maps to norfair's ``hit_counter_max``.
    min_hits : int
        Minimum hit count before a new track appears in output.
        1 = confirmed after first matched frame (ByteTrack default).
        0 = report immediately on creation.
    track_high_thresh : float
        Confidence threshold for the first association pass.
        Detections at or above this score are treated as high-confidence.
    track_low_thresh : float
        Minimum confidence for a detection to participate in the second
        (low-confidence) association pass.  Detections below this are ignored.
    iou_threshold_high : float
        IoU threshold for Pass 1 (high-conf detections vs tracked tracks).
    iou_threshold_low : float
        IoU threshold for Pass 2 (low-conf detections vs unmatched tracked tracks).
    iou_threshold_lost : float
        IoU threshold for Pass 3 (unmatched high-conf detections vs lost tracks).

    Deriving thresholds from config2.yml
    ────────────────────────────────────
    norfair uses ``iou_distance = 1/IoU``, so ``max_distance_between_points = 3``
    maps to ``min_iou ≈ 0.33``.  Use that as ``iou_threshold_high`` and adjust
    the other thresholds accordingly (lower for low-conf, same for lost).

    Usage in main.py
    ----------------
        tracker = ByteTracker(max_age=30, iou_threshold_high=0.33)
        tracked_objects = tracker.update(detections=frame_detections)
    """

    def __init__(
        self,
        max_age: int           = 30,
        min_hits: int          = 1,
        track_high_thresh: float = 0.50,
        track_low_thresh: float  = 0.10,
        iou_threshold_high: float = 0.30,
        iou_threshold_low: float  = 0.20,
        iou_threshold_lost: float = 0.30,
    ) -> None:
        self.max_age            = max_age
        self.min_hits           = min_hits
        self.track_high_thresh  = track_high_thresh
        self.track_low_thresh   = track_low_thresh
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low  = iou_threshold_low
        self.iou_threshold_lost = iou_threshold_lost

        self._tracked_tracks: List[_BYTETrack] = []   # currently TRACKED
        self._lost_tracks:    List[_BYTETrack] = []   # recently LOST
        self._frame_id: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        detections: Optional[List[TrackerDetection]] = None,
    ) -> List[TrackedObject]:
        """Process one frame and return active tracked objects.

        Call signature mirrors ``norfair.Tracker.update``:

            tracked_objects = tracker.update(detections=frame_detections)

        Parameters
        ----------
        detections : list[TrackerDetection] | None
            Detections for the current frame.

        Returns
        -------
        list[TrackedObject]
            Confirmed TRACKED objects for this frame.
        """
        if detections is None:
            detections = []

        self._frame_id += 1

        # ── Split detections by confidence ────────────────────────────────────
        dets_high: List[TrackerDetection] = [
            d for d in detections if d.score >= self.track_high_thresh
        ]
        dets_low: List[TrackerDetection] = [
            d for d in detections
            if self.track_low_thresh <= d.score < self.track_high_thresh
        ]

        # ── Predict all existing tracks one step forward ──────────────────────
        for t in self._tracked_tracks + self._lost_tracks:
            t.predict()

        # ── Pass 1: TRACKED tracks  ↔  HIGH-conf detections ──────────────────
        matched_1, unmatched_trk_1, unmatched_det_1 = _iou_match(
            self._tracked_tracks, dets_high, self.iou_threshold_high
        )
        for ti, di in matched_1:
            self._tracked_tracks[ti].update(dets_high[di].bbox, data=dets_high[di].data)
            self._tracked_tracks[ti].score = dets_high[di].score
            self._tracked_tracks[ti].state = _State.TRACKED

        # ── Pass 2: unmatched TRACKED  ↔  LOW-conf detections ────────────────
        remaining_tracked = [self._tracked_tracks[i] for i in unmatched_trk_1]
        matched_2, unmatched_trk_2_local, _ = _iou_match(
            remaining_tracked, dets_low, self.iou_threshold_low
        )
        for ti, di in matched_2:
            remaining_tracked[ti].update(dets_low[di].bbox, data=dets_low[di].data)
            remaining_tracked[ti].score = dets_low[di].score
            remaining_tracked[ti].state = _State.TRACKED

        # Tracks unmatched in both passes → mark LOST
        for ti in unmatched_trk_2_local:
            if remaining_tracked[ti].state == _State.TRACKED:
                remaining_tracked[ti].mark_lost()

        # ── Pass 3: LOST tracks  ↔  remaining HIGH-conf detections ───────────
        unmatched_high_dets = [dets_high[i] for i in unmatched_det_1]
        matched_3, _, unmatched_new = _iou_match(
            self._lost_tracks, unmatched_high_dets, self.iou_threshold_lost
        )
        for ti, di in matched_3:
            self._lost_tracks[ti].re_activate(
                unmatched_high_dets[di].bbox,
                unmatched_high_dets[di].score,
                data=unmatched_high_dets[di].data,
            )

        # ── Initialise new tracks ─────────────────────────────────────────────
        for di in unmatched_new:
            d = unmatched_high_dets[di]
            new_t = _BYTETrack(d.bbox, d.score, data=d.data)
            new_t.activate(self._frame_id)
            self._tracked_tracks.append(new_t)

        # ── Rebuild track lists ───────────────────────────────────────────────
        # Re-activated lost tracks (state flipped to TRACKED) → move to tracked
        re_activated = [t for t in self._lost_tracks if t.state == _State.TRACKED]

        # Lost tracks that have aged out → REMOVED
        self._lost_tracks = [
            t for t in self._lost_tracks
            if t.state == _State.LOST and t.time_since_update <= self.max_age
        ]

        # Newly lost tracks from the tracked list
        newly_lost = [t for t in self._tracked_tracks if t.state == _State.LOST]

        # Final tracked list = still-tracked + re-activated
        self._tracked_tracks = (
            [t for t in self._tracked_tracks if t.state == _State.TRACKED]
            + re_activated
        )

        # Final lost list = surviving old lost + newly lost
        self._lost_tracks += newly_lost

        # ── Build output ──────────────────────────────────────────────────────
        # Include both TRACKED and LOST tracks so that a track coasts in the
        # output until it truly expires (mirrors norfair behaviour and prevents
        # the death-detection logic in main.py from firing on every missed frame).
        output: List[TrackedObject] = []
        for t in self._tracked_tracks + self._lost_tracks:
            if t.is_activated and t.hits >= self.min_hits:
                output.append(
                    TrackedObject(
                        global_id=t.global_id,
                        bbox=t.get_state(),
                        age=t.age,
                        data=t.data,
                    )
                )

        return output
