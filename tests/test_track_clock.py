"""
Unit tests for measured track ages (utils/track_clock.py).
Track age in seconds must come from measured frame durations, not from a
configured FPS, and the Head-Detector / v2 copies of the module must match.
"""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD_PATH = os.path.join(ROOT, "Head-Detector", "utils", "track_clock.py")
V2_PATH = os.path.join(ROOT, "Queue-Management-System-v2-main",
                       "Queue-Management-System-v2-main", "utils", "track_clock.py")

_spec = importlib.util.spec_from_file_location("track_clock", HEAD_PATH)
track_clock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(track_clock)
FrameClock, TrackClock = track_clock.FrameClock, track_clock.TrackClock


def _run(frame_times, clock=None):
    clock = clock or FrameClock()
    for t in frame_times:
        clock.tick(now=t)
    return clock


class TestFrameClock(unittest.TestCase):
    def test_first_frame_has_no_duration(self):
        clock = _run([100.0])
        self.assertEqual(clock.dt, 0.0)
        self.assertEqual(clock.fps, 0.0)

    def test_measures_frame_duration(self):
        clock = _run([0.0, 0.25, 0.5, 0.75])
        self.assertAlmostEqual(clock.dt, 0.25)
        self.assertAlmostEqual(clock.avg_dt, 0.25)
        self.assertAlmostEqual(clock.fps, 4.0)
        self.assertAlmostEqual(clock.frames_to_sec(4), 1.0)

    def test_stall_is_not_credited_as_one_frame(self):
        clock = _run([0.0, 0.1, 0.2, 30.2])  # 30 s reconnect gap
        self.assertAlmostEqual(clock.dt, 0.1)


class TestTrackClock(unittest.TestCase):
    def test_age_follows_measured_time_when_fps_varies(self):
        # 3 frames at 0.1 s then 3 frames at 0.5 s: a fixed FPS can't get this right
        times = [0.0, 0.1, 0.2, 0.3, 0.8, 1.3, 1.8]
        clock, tracks = FrameClock(), TrackClock()
        for frame_age, t in enumerate(times):
            clock.tick(now=t)
            age = tracks.update(7, frame_age, clock, hit=True)
        self.assertAlmostEqual(age, 1.8)
        self.assertAlmostEqual(tracks.hits(7), 1.8)

    def test_hit_seconds_only_count_matched_frames(self):
        clock, tracks = FrameClock(), TrackClock()
        for frame_age, (t, hit) in enumerate([(0.0, True), (0.2, True), (0.4, False), (0.6, True)]):
            clock.tick(now=t)
            tracks.update(1, frame_age, clock, hit=hit)
        self.assertAlmostEqual(tracks.age(1), 0.6)
        self.assertAlmostEqual(tracks.hits(1), 0.4)

    def test_track_reported_late_is_seeded_with_its_frame_age(self):
        clock = _run([0.0, 0.2, 0.4, 0.6])
        tracks = TrackClock()
        # Tracker held the track 3 frames before reporting it (initialization delay)
        self.assertAlmostEqual(tracks.update(5, 3, clock), 0.6)

    def test_forget_drops_track(self):
        clock, tracks = _run([0.0, 0.1]), TrackClock()
        tracks.update(3, 0, clock, hit=True)
        tracks.forget(3)
        self.assertEqual(tracks.age(3), 0.0)
        self.assertEqual(tracks.hits(3), 0.0)


class TestCopiesInSync(unittest.TestCase):
    def test_head_detector_and_v2_copies_identical(self):
        with open(HEAD_PATH, "rb") as a, open(V2_PATH, "rb") as b:
            self.assertEqual(a.read(), b.read())


if __name__ == "__main__":
    unittest.main()
