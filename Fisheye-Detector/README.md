# Fisheye-Detector — Fisheye Checkout Pipeline

Standalone fisheye person/pedestrian detector and multi-lane queue tracking pipeline with the **exact same CLI, configuration, `Box` API, tracker backends (Norfair / SORT / ByteTrack), and TimescaleDB logging (`DBLogger`) as `Head-Detector/`**.

## Supported Detection Models (`detector_backend` in `config.yml` or `--detector_backend`)

All models from [UT_AICITY2025_Track4](https://github.com/arnaudbastide/UT_AICITY2025_Track4) plus **RAPiD** and **ONNX Runtime** models are supported behind the unified `FisheyeDetector` (`YOLOv9` alias) class in `utils/yolo.py`:

| `detector_backend` | Model / Architecture | Weights Format | Execution Providers / Devices |
|---|---|---|---|
| `onnx` (or `auto` on `.onnx`) | Exported YOLOv10, YOLOv13, YOLOR, YOLOv8/v9, or PINTO `*_nonms_*` | `.onnx` | `openvino` (Intel Xeon / NPU), `cuda` (RTX 3090), `tensorrt`, `cpu` |
| `yolov10` (`ultralytics`) | `UT_AICITY2025_Track4` YOLOv10-X | `.pt`, `.engine` | `cuda`, `cpu` |
| `yolov13` (`ultralytics`) | `UT_AICITY2025_Track4` YOLOv13-L | `.pt`, `.engine` | `cuda`, `cpu` |
| `yolor` | `UT_AICITY2025_Track4` YOLOR-D6 (`train/YoloR`) | `.pt` | `cuda`, `cpu` |
| `wbf` | `UT_AICITY2025_Track4` 3-model Weighted Boxes Fusion (`YOLOR + YOLOv10 + YOLOv13`) | 3 × `.pt` / `.engine` | `cuda`, `cpu` |
| `rapid` (or `auto` on `.ckpt`) | RAPiD rotation-aware overhead fisheye people detector | `.ckpt` (`pL1_MWHB1024_Mar11_4000.ckpt`) | `cuda`, `cpu` |

### Class Mapping
- For `UT_AICITY2025_Track4` models trained on Fisheye8K / VisDrone (`0: Bus, 1: Bike, 2: Car, 3: Pedestrian, 4: Truck`), **Pedestrian (`class_id == 3`)** is automatically normalized to `Box(classid=0, ...)`.
- For single-class person/head models, COCO models, and RAPiD, `class_id == 0` is emitted as `Box(classid=0, ...)`.
- You can also set `person_class_ids: [3, 0]` explicitly in `config.yml`.

## Running `Fisheye-Detector`

Same command-line flags as `Head-Detector/main.py`:

```powershell
cd Fisheye-Detector

# OpenVINO (Intel Xeon CPU or Intel NPU, configured via OPENVINO_DEVICE / config2.yml)
python main.py --execution_provider openvino --no-view-img

# NVIDIA GPU (CUDA, e.g. RTX 3090)
python main.py --execution_provider cuda --no-view-img

# Plain CPU
python main.py --execution_provider cpu --no-view-img

# Override model or backend on the command line:
python main.py --execution_provider cuda --detector_backend yolov13 --custom_weights models/yolov13_l_best_checkpoint.pt --img-size 1280
python main.py --execution_provider cuda --detector_backend rapid --custom_weights models/pL1_MWHB1024_Mar11_4000.ckpt --img-size 1024
python main.py --execution_provider cuda --detector_backend wbf
```

## Benchmarking on Fisheye Datasets (`eval/fisheye/`)

```powershell
python eval/fisheye/run_detector.py --detector fisheye --model Fisheye-Detector/models/<model>.onnx --dataset CEPDOF --step 10
python eval/fisheye/score_detections.py --dets-dir eval/data/fisheye/dets/fisheye/CEPDOF
python eval/fisheye/track_eval.py --dets-dir eval/data/fisheye/dets/fisheye/CEPDOF --fps 10 --step 10
```
