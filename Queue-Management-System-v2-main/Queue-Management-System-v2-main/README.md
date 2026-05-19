# Queue-Management-System-v2 — Pipeline B

YOLOv9 body detector + RetinaFace/Uniface face analytics + ByteTrack/SORT tracking with TimescaleDB logging.

## What it does

- Detects people (bodies) in a configurable ROI polygon using YOLOv9 ONNX
- Detects handbags and backpacks (COCO classes 24 + 26) in the same pass
- Runs face attribute analysis (gender, age, confidence) via Uniface/RetinaFace
- Tracks individuals with ByteTrack or SORT (configurable via `config2.yml`)
- Per track, accumulates **confidence-weighted** gender votes and age readings
- Associates bags to persons via proximity check; `has_bag` is latched True once detected
- **Inserts one DB row per track at track death**
  - `timestamp` = actual entry time (first frame), not insert time
  - `gender` = confidence-weighted vote across all frames
  - `age_estimate` = confidence-weighted average across all frames
  - `has_bag` = True if a bag was detected near the person in any frame
  - Tracks shorter than `min_elapsed_time` are discarded without a DB insert
- Writes periodic `queue_state_snapshots` (queue count, avg/max dwell, lanes)

## Setup

Download ONNX models and place them in `models/`:
https://drive.google.com/drive/folders/1gxqqcMACrjvegS0_OQypeT_lDKarco5_?usp=sharing

```bash
cd Queue-Management-System-v2-main/Queue-Management-System-v2-main
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use this project-local `venv` when running `queue_management_v2.py`. Do not install these packages only into the workspace-root `.venv`, because this app has its own separate environment.

`requirements.txt` now includes the OpenVINO runtime used by this app:

- `onnxruntime-openvino==1.23.0`
- `openvino==2025.3.0`

Quick provider check:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `OpenVINOExecutionProvider` in the output after installation.

## Run

```bash
python queue_management_v2.py
python queue_management_v2.py --device openvino
python queue_management_v2.py --execution_provider cuda
python queue_management_v2.py --execution_provider openvino --view-img
```

Notes:

- `--device openvino` and `--execution_provider openvino` are both accepted.
- `--view_img` and `--view-img` are both accepted.

## Configuration files

| File | Key settings |
|---|---|
| `config.yml` | `camID`, `ip_address`, RTSP credentials, ONNX model path, ROI polygon, `active_lanes` |
| `config2.yml` | `max_age`, `min_iou` (face-body match), `snapshot_interval`, `min_elapsed_time`, `debug_mode` |

## Camera ID note

Use `camID: SIM_live_test_video` when running against a local test video file so entries are excluded from real-data training queries.

## Forecasting

```bash
python prophet_predict.py --source REAL
python ensemble_predict.py --source REAL
```

## Database tables written

| Table | Written when |
|---|---|
| `entrance_events` | At track death (one row per confirmed person, includes `has_bag`) |
| `queue_state_snapshots` | Every `snapshot_interval` seconds |
| `queue_predictions` | By `ensemble_predict.py` when run |
