# Tracking evaluation, step 1: summary

Date: 2026-10-30. Scope: checkout camera (Head-Detector, YOLOv9-S head detector + Norfair), current Bosch camera.

## In short

- The current pipeline follows a single customer correctly (live test in front of the camera).
- The database shows that tracks are often split: in each lane, 23-30% of counted service rows start within 1.5 s of the previous row ending. One person counted as two rows means two shorter dwell times.
- The probable cause is a mismatch between the camera frame rate and the loop rate (details below). It is not specific to this camera and will affect the fisheye camera too.
- Since the target is the fisheye camera, we stop evaluating the current setup here and move to step 2 (top-view detectors). The fix below goes into step 3.

## What was measured

| Source | What it gives |
|---|---|
| 3 recorded clips, 30 min each, 2026-10-29: 14:31 (quiet), 15:01 (afternoon), 17:51 (evening peak) | Footage for replay, kept on the server / laptop only |
| Frame log (new config flag `frame_log_enabled`, off by default) | Real loop rate and camera frame rate |
| Database, last 14 days (`eval/db_check.py`, read-only) | Busy hours, rows per clip, split-track signal |
| Live test in front of the camera | Tracking of one person |

Manual labelling of arrivals and departures was tried on 5 minutes of footage and dropped: it is too slow for a 30-minute clip with many moving customers.

## Findings

**1. Each camera frame is processed about 14 times.**
The camera stream delivers about 2.3 frames per second (no missing frames, about 0.4-0.5 s apart). The loop runs at about 33.6 passes per second (median 20 ms per pass) and processes the latest frame each time, new or not.

**2. The tracker forgets a person after about 0.12 s.**
Norfair settings: `hit_counter_max = 4` loop passes, IoU distance, threshold 6, no re-identification. If the detector misses a person in a single camera frame, that frame is processed about 14 times without them, far more than the 4 passes the tracker waits, so the track ends. On the next frame a new track starts for the same person.

**3. The database shows that pattern.**

| Lane (DB `lane_id`) | Rows, 14 days | Share starting within 1.5 s of the previous row's end |
|---|---|---|
| 0, 1, 2, 3 | 5,752 / 6,569 / 9,951 / 3,961 | 23-30% per lane |
| Total | 26,233 | 6,944 (26%) |

Median dwell per lane is 5.8-9.2 s. During the 30-minute 14:31 clip the live system wrote 1,807 tracks, of which 292 reached the 3 s minimum and were counted (queue average 1.9, maximum 6). During the 15:01 clip: 1,603 tracks, 327 counted (queue average 2.3, maximum 11).

**4. The business rules hide part of the problem.**
The 3 s minimum dwell, sibling suppression (a shorter track in the same lane as a longer one is dropped) and the confirmation delay remove most short pieces. When both pieces of a split person are longer than 3 s, both are counted and both dwell times are too short.

**Not measured:** the dwell error in seconds against real arrival and departure times. That needs hand-made reference times, which were dropped (see above).

## Why we stop here

- The target is the fisheye camera. The detector (trained on side views), the lane polygons and the way people move in the image will all change, so tuning Norfair for the current view would mostly be lost.
- Public fisheye datasets come with labels, so step 2 can be measured automatically.

## Fix to carry over (step 3, behind a config flag, off by default)

- Run detection and tracking only when the camera delivers a new frame. The frame counter needed for this is already in `VideoStream` on the evaluation branch.
- Express the tracker's memory and the confirmation delay in seconds instead of loop passes, so that a missed frame does not end a track.
- Check the effect with the split-track query (`db_check.py fragments`) before and after.

## Tools ready for the next steps (branch `eval-tracking-step1-prod`)

- `record_clip.ps1`: records a clip without touching the live loop.
- `detect_cache.py`: runs the detector once per clip.
- `track_replay.py` + `pipeline_logic.py`: replays the tracker with the same business rules as production. Any new detector or tracker plugs in before those rules.
- `db_check.py`: read-only database checks.

## Next: step 2

Evaluate top-view detectors (RAPiD first, then a rotated-box YOLO) on public ceiling fisheye datasets (CEPDOF, WEPDTOF, HABBOF), which include labels. Generated videos (Gemini Flow) will be used for visual checks only, since they have no labels.
