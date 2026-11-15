"""Measured track ages in seconds.

Trackers count age in frames. Converting frames to seconds with a configured
FPS (the old `expect_fps`) is wrong whenever the real processing rate differs
from the configured one (NPU load, face worker, throttle, slow frames).

FrameClock measures how much wall time each processed frame accounts for, and
TrackClock accumulates that per track, so ages come out in seconds directly.

Kept identical in Head-Detector/utils and Queue-Management-System-v2-main/utils
(tests/test_track_clock.py checks this).
"""
import time


class FrameClock:
    """Wall time between successive processed frames (one tick per tracker.update)."""

    def __init__(self, max_gap_sec: float = 2.0, smoothing: float = 0.1):
        self.max_gap_sec = max_gap_sec  # longer gaps are stream stalls, not frame time
        self.smoothing = smoothing      # EMA weight of the newest frame in avg_dt
        self.dt = 0.0                   # seconds the current frame accounts for
        self.avg_dt = 0.0               # smoothed seconds per frame
        self._last = None

    def tick(self, now: float = None) -> float:
        """Call once per processed frame, right after tracker.update(). Returns dt."""
        now = time.perf_counter() if now is None else now
        if self._last is None:
            dt = 0.0
        else:
            dt = now - self._last
            # The tracker did not advance during a stall/reconnect, so one long
            # gap must not be credited to every live track as a single frame.
            if dt > self.max_gap_sec:
                dt = self.avg_dt
        self._last = now
        self.dt = dt
        if dt > 0:
            self.avg_dt = dt if self.avg_dt == 0 else self.avg_dt + self.smoothing * (dt - self.avg_dt)
        return dt

    @property
    def fps(self) -> float:
        return 1.0 / self.avg_dt if self.avg_dt > 0 else 0.0

    def frames_to_sec(self, frames: float) -> float:
        """Convert a frame count (e.g. tracker max_age) to seconds at the measured rate."""
        return frames * self.avg_dt


class TrackClock:
    """Per-track seconds, built from the measured duration of each frame."""

    def __init__(self):
        self.age_sec = {}  # track_id -> seconds since the tracker created the track
        self.hit_sec = {}  # track_id -> seconds of frames with a matched detection

    def update(self, track_id, frame_age: int, clock: FrameClock, hit: bool = False) -> float:
        """Call once per frame for every reported track. Returns its age in seconds."""
        if track_id not in self.age_sec:
            # Tracker age is 0 on the creation frame, but the tracker may have held
            # the track for a few frames before reporting it (initialization
            # delay); seed those at the measured rate.
            self.age_sec[track_id] = frame_age * clock.avg_dt
            self.hit_sec[track_id] = 0.0
        else:
            self.age_sec[track_id] += clock.dt
        if hit:
            self.hit_sec[track_id] += clock.dt
        return self.age_sec[track_id]

    def age(self, track_id) -> float:
        return self.age_sec.get(track_id, 0.0)

    def hits(self, track_id) -> float:
        return self.hit_sec.get(track_id, 0.0)

    def forget(self, track_id):
        self.age_sec.pop(track_id, None)
        self.hit_sec.pop(track_id, None)
