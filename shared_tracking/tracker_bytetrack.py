from __future__ import annotations

from enum import IntEnum
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker_base import KalmanBoxTracker, TrackerDetection, TrackedObject, iou_batch


class _State(IntEnum):
    TRACKED = 0
    LOST = 1
    REMOVED = 2


class _BYTETrack(KalmanBoxTracker):
    def __init__(
        self,
        bbox: np.ndarray,
        score: float,
        data=None,
    ) -> None:
        super().__init__(bbox, data=data)
        self.score = score
        self.state = _State.TRACKED
        self.is_activated = False
        self.start_frame = 0

    def activate(self, frame_id: int) -> None:
        self.is_activated = True
        self.start_frame = frame_id
        self.state = _State.TRACKED

    def re_activate(
        self,
        bbox: np.ndarray,
        score: float,
        data=None,
    ) -> None:
        self.update(bbox, data=data)
        self.score = score
        self.state = _State.TRACKED
        self.is_activated = True

    def mark_lost(self) -> None:
        self.state = _State.LOST

    def mark_removed(self) -> None:
        self.state = _State.REMOVED


def _iou_match(
    tracks: List[_BYTETrack],
    detections: List[TrackerDetection],
    iou_threshold: float,
) -> tuple[List[tuple[int, int]], List[int], List[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    trk_boxes = np.array([track.get_state() for track in tracks])
    det_boxes = np.array([detection.bbox for detection in detections])

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

    matched_trk = {r for r, _ in matched}
    matched_det = {c for _, c in matched}

    unmatched_trk = rejected_trk + [
        i for i in range(len(tracks)) if i not in matched_trk and i not in rejected_trk
    ]
    unmatched_det = rejected_det + [
        j for j in range(len(detections)) if j not in matched_det and j not in rejected_det
    ]

    return matched, unmatched_trk, unmatched_det


class ByteTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.10,
        iou_threshold_high: float = 0.30,
        iou_threshold_low: float = 0.20,
        iou_threshold_lost: float = 0.30,
        **_: object,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low = iou_threshold_low
        self.iou_threshold_lost = iou_threshold_lost

        self._tracked_tracks: List[_BYTETrack] = []
        self._lost_tracks: List[_BYTETrack] = []
        self._frame_id: int = 0

    def update(
        self,
        detections: Optional[List[TrackerDetection]] = None,
    ) -> List[TrackedObject]:
        if detections is None:
            detections = []

        self._frame_id += 1

        dets_high: List[TrackerDetection] = [
            detection for detection in detections if detection.score >= self.track_high_thresh
        ]
        dets_low: List[TrackerDetection] = [
            detection for detection in detections if self.track_low_thresh <= detection.score < self.track_high_thresh
        ]

        for track in self._tracked_tracks + self._lost_tracks:
            track.predict()

        matched_1, unmatched_trk_1, unmatched_det_1 = _iou_match(
            self._tracked_tracks, dets_high, self.iou_threshold_high
        )
        for ti, di in matched_1:
            self._tracked_tracks[ti].update(dets_high[di].bbox, data=dets_high[di].data)
            self._tracked_tracks[ti].score = dets_high[di].score
            self._tracked_tracks[ti].state = _State.TRACKED

        remaining_tracked = [self._tracked_tracks[i] for i in unmatched_trk_1]
        matched_2, unmatched_trk_2_local, _ = _iou_match(
            remaining_tracked, dets_low, self.iou_threshold_low
        )
        for ti, di in matched_2:
            remaining_tracked[ti].update(dets_low[di].bbox, data=dets_low[di].data)
            remaining_tracked[ti].score = dets_low[di].score
            remaining_tracked[ti].state = _State.TRACKED

        for ti in unmatched_trk_2_local:
            if remaining_tracked[ti].state == _State.TRACKED:
                remaining_tracked[ti].mark_lost()

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

        for di in unmatched_new:
            detection = unmatched_high_dets[di]
            new_track = _BYTETrack(detection.bbox, detection.score, data=detection.data)
            new_track.activate(self._frame_id)
            self._tracked_tracks.append(new_track)

        re_activated = [track for track in self._lost_tracks if track.state == _State.TRACKED]

        self._lost_tracks = [
            track
            for track in self._lost_tracks
            if track.state == _State.LOST and track.time_since_update <= self.max_age
        ]

        newly_lost = [track for track in self._tracked_tracks if track.state == _State.LOST]

        self._tracked_tracks = [
            track for track in self._tracked_tracks if track.state == _State.TRACKED
        ] + re_activated
        self._lost_tracks += newly_lost

        output: List[TrackedObject] = []
        for track in self._tracked_tracks + self._lost_tracks:
            if track.is_activated and track.hits >= self.min_hits:
                output.append(
                    TrackedObject(
                        global_id=track.global_id,
                        bbox=track.get_state(),
                        age=track.age,
                        data=track.data,
                    )
                )

        return output


ByteTrackTracker = ByteTracker