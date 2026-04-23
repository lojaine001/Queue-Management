# Head-Detector — Pipeline A

YOLOv9 head-pose detector with Norfair tracking and TimescaleDB logging.

## What it does

- Detects heads inside one of the configured ROI polygons using a YOLOv9 ONNX model
- Tracks head detections across frames with Norfair (IoU-based)
- Hides short-lived tracks until they survive the confirmation window `max_age / expect_fps`
- Measures dwell from the first tracked frame until the track dies
- Inserts at most one DB row per dead track after applying rejection rules
- Writes periodic `queue_state_snapshots` (queue count, avg/max dwell, lanes)
- Stores `gender='unknown'`, `age=NULL` (no face analytics in this pipeline)

## DB Insert / Reject Strategy

`Head-Detector` writes to `entrance_events` only when a head track dies, not while it is still active.

For a dead track to be inserted into the DB, all of the following must be true:

- the track had been created and later disappeared from the active Norfair set
- the track had a valid ROI assignment
- there is no still-active track in the same ROI with a higher dwell time
- the track was observed for long enough:
  - `active_tracked_sec = track_hits / expect_fps`
  - the insert only happens when `active_tracked_sec > max_age / expect_fps`

If any of those checks fail, the dead track is rejected and no DB row is written.

Current rejection reasons in the code are:

- another active track in the same ROI already has a higher dwell time
- the dead track was tracked for too short a time to be considered reliable

When a row is inserted:

- `timestamp` is the track start time, not the death time
- `dwell_seconds` is the total tracked dwell until death
- `active_head_tracks_in_lane` is the number of currently active confirmed head tracks in the same lane at insert time

## On-Screen Feedback

Confirmed active tracks are shown lane-by-lane:

- yellow = current best active candidate in that lane
- blue = other confirmed active tracks in that lane

Each ROI polygon is labeled on screen as `L1: N`, `L2: N`, `L3: N`, ... where `N` is the current number of active confirmed head tracks in that lane.

After a head track dies, the last known bbox is shown for 2 seconds only when the track is inserted into the DB:

- green = the track was inserted into the DB
- rejected tracks do not show a red bbox by default

## Setup

```bash
cd Head-Detector
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Troubleshooting

If startup fails while importing `norfair`/`scipy`, the virtualenv usually has an incompatible NumPy install. This project requires `numpy < 2.0` for `norfair==2.3.0`.

```bash
pip uninstall -y numpy scipy norfair
pip install --no-cache-dir --force-reinstall "numpy>=1.23,<2.0" "scipy>=1.10,<1.16" "norfair==2.3.0"
```

If startup fails with errors like `numpy.core.multiarray failed to import` or `module compiled against ABI version ...`, rebuild the Python packages that depend on NumPy:

```bash
pip uninstall -y numpy scipy norfair opencv-python onnx protobuf onnxruntime onnxruntime-openvino openvino openvino-telemetry
pip install --no-cache-dir --force-reinstall -r requirements.txt
pip install --no-cache-dir --force-reinstall "numpy>=1.23,<2.0" "protobuf<=3.20.1" openvino onnxruntime-openvino
```

If `OpenVINOExecutionProvider` is listed as available but the session still enables only `CPUExecutionProvider`, check the OpenVINO version pairing. `onnxruntime-openvino 1.23.x` is intended for `openvino 2025.3.x`; newer `openvino 2026.x` builds can cause silent fallback to CPU.

```bash
pip uninstall -y openvino openvino-telemetry
pip install --no-cache-dir --force-reinstall "openvino==2025.3.0"
```

If that still fails, recreate the virtualenv and reinstall from `requirements.txt`.

## ROI configuration

Run once to define the queue zone polygon:

```bash
python pick_zone.py
```

Left-click to add points, `Z` to undo, `Enter`/`Space` to print the `points:` block, paste into `config.yml`.

## Run

```bash
python main.py
python main.py --execution_provider cuda
python main.py --execution_provider tensorrt --inference_type fp16
python main.py --source path/to/video.mp4 --view-img
```

## Configuration files

| File | Key settings |
|---|---|
| `config.yml` | `camID`, `ip_address`, RTSP credentials, ONNX model path, ROI polygon |
| `config2.yml` | `max_age`, `max_distance_between_points`, `expect_fps`, `snapshot_interval`, `debug_mode`, `show_rejected_track_overlay` |

## Camera ID note

Use `camID: SIM_live_test_video` when running against a local test video file so entries are excluded from real-data training queries.

## Database tables written

| Table | Written when |
|---|---|
| `entrance_events` | At head-track death if the insert rules pass |
| `queue_state_snapshots` | Every `snapshot_interval` seconds |

## Useful Config

- `max_age / expect_fps` defines the display and confirmation window for head tracks
- `show_rejected_track_overlay: False` keeps rejected dead tracks from flashing red
- set `show_rejected_track_overlay: True` if you want the red rejection bbox back for debugging
