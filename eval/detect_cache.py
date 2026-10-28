"""Run the checkout head detector over a recorded clip and cache every frame's detections.

Run this on the production machine (it has the model and the NPU), outside store
hours: the NPU is shared with the live loop, and running both at once slows it down.

    python eval/detect_cache.py --video eval/data/raw/checkout_20261029_101500_peak.mkv

Output: eval/data/cache/<clip>.dets.npz. It holds only frame timestamps, box
coordinates and scores, no pixels, so it can be copied to another machine to rerun
tracker experiments without the footage. Every detection the model returns is kept
(no ROI or score filtering here), so those thresholds can still be varied in replay.

Use the same Head-Detector virtualenv as the live system so the model, ONNX Runtime
and OpenVINO versions match production.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAD_DIR = REPO_ROOT / 'Head-Detector'


# ── Copied from Head-Detector/main.py (importing main.py would open its live log files).
# Keep in sync with main.py if the provider setup there changes.
_WINDOWS_DLL_DIR_HANDLES = []


def _register_windows_dll_dir(path):
    if sys.platform != "win32" or not path:
        return
    dll_dir = Path(path)
    if not dll_dir.is_dir():
        return
    try:
        _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
    except (AttributeError, FileNotFoundError, OSError):
        return


def _register_windows_runtime_dirs():
    if sys.platform != "win32":
        return
    _register_windows_dll_dir(Path(sys.executable).resolve().parent)
    _register_windows_dll_dir(Path(sys.base_prefix) if sys.base_prefix else None)
    ort_spec = importlib.util.find_spec("onnxruntime")
    if ort_spec and ort_spec.origin:
        _register_windows_dll_dir(Path(ort_spec.origin).resolve().parent / "capi")
    ov_spec = importlib.util.find_spec("openvino")
    if ov_spec and ov_spec.origin:
        _register_windows_dll_dir(Path(ov_spec.origin).resolve().parent / "libs")


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _build_openvino_provider_options(device, precision, cache_dir):
    device = 'NPU' if device.upper() == 'MYRIAD' else device.upper()
    precision = precision.upper()
    if device == 'CPU' and precision != 'FP32':
        precision = 'FP32'
    ort_version = _version('onnxruntime-openvino') or _version('onnxruntime')
    major_minor = tuple(int(p) for p in ort_version.split('.')[:2]) if ort_version else None
    if major_minor and major_minor < (1, 17):
        return {'device_type': f'{device}_{precision}', 'cache_dir': cache_dir}
    return {'device_type': device, 'precision': precision, 'cache_dir': cache_dir}
# ──────────────────────────────────────────────────────────────────────────────


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', required=True, help='recorded clip (.mkv/.mp4)')
    parser.add_argument('--out-dir', default=str(REPO_ROOT / 'eval' / 'data' / 'cache'))
    parser.add_argument('--provider', choices=['openvino', 'cpu'], default='openvino',
                        help='openvino = same as production (default); cpu gives slightly different boxes')
    parser.add_argument('--max-frames', type=int, default=0, help='stop after N frames (0 = whole clip)')
    args = parser.parse_args()

    _register_windows_runtime_dirs()
    sys.path.insert(0, str(HEAD_DIR))
    import cv2
    import numpy as np
    from utils.queue_utils import get_config
    from utils.yolo import YOLOv9

    stored_config = get_config(str(HEAD_DIR / 'config.yml'))
    config2 = get_config(str(HEAD_DIR / 'config2.yml'))
    model_rel = stored_config.get('custom_model', 'models/yolov9_c_discrete_headpose_post_0100_1x3x480x640.onnx')
    model_path = HEAD_DIR / model_rel
    disable_HI = stored_config.get('disable_headpose_identification_mode', False)
    ov_device = config2.get('openvino_device', 'CPU')
    ov_precision = config2.get('openvino_precision', 'FP32')

    if args.provider == 'openvino':
        providers = [('OpenVINOExecutionProvider',
                      _build_openvino_provider_options(ov_device, ov_precision, str(HEAD_DIR / 'ov_cache'))),
                     'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # Same thresholds as main.py.
    model = YOLOv9(model_path=str(model_path), obj_class_score_th=0.35, attr_class_score_th=0.70,
                   providers=providers)
    active_providers = list(getattr(model, '_providers', []) or [])
    print(f'[CACHE] model={model_rel} providers={active_providers}')
    if args.provider == 'openvino' and 'OpenVINOExecutionProvider' not in active_providers:
        print('[CACHE] WARNING: OpenVINO provider not active, running on CPU. Boxes may differ from production.')

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f'Cannot open {video}')
    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    pts_list, det_counts, infer_ms, rows = [], [], [], []
    idx = 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pts <= 0 and idx > 0 and stream_fps > 0:
            pts = idx / stream_fps
        frame = cv2.resize(frame, (640, 480))  # same resize as the live loop
        t0 = time.perf_counter()
        boxes = model(image=frame, disable_headpose_identification_mode=disable_HI)
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        for b in boxes:
            hd = getattr(b, 'headdirection', None)
            rows.append([b.classid, b.score, b.x1, b.y1, b.x2, b.y2, -1 if hd is None else hd])
        pts_list.append(pts)
        det_counts.append(len(boxes))
        idx += 1
        if idx % 500 == 0:
            el = time.time() - t_start
            print(f'[CACHE] {idx} frames, {el:.0f}s, {idx / el:.1f} fps, median infer {np.median(infer_ms):.1f} ms')
        if args.max_frames and idx >= args.max_frames:
            break
    cap.release()

    if idx == 0:
        sys.exit('No frames decoded.')

    offsets = np.zeros(idx + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(det_counts)
    meta = {
        'video': video.name,
        'video_sha256': sha256_file(video),
        'n_frames': idx,
        'stream_fps': stream_fps,
        'frame_size': [640, 480],
        'model': model_rel,
        'model_sha256': sha256_file(model_path),
        'provider_requested': args.provider,
        'providers_active': active_providers,
        'openvino_device': ov_device,
        'openvino_precision': ov_precision,
        'disable_headpose_identification_mode': disable_HI,
        'obj_class_score_th': 0.35,
        'onnxruntime': _version('onnxruntime-openvino') or _version('onnxruntime'),
        'openvino': _version('openvino'),
        'git_commit': git_commit(),
        'created': datetime.now().isoformat(timespec='seconds'),
        'det_columns': ['classid', 'score', 'x1', 'y1', 'x2', 'y2', 'headdirection'],
        'median_infer_ms': float(np.median(infer_ms)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{video.stem}.dets.npz'
    np.savez_compressed(
        out_path,
        pts=np.asarray(pts_list, dtype=np.float64),
        offsets=offsets,
        dets=np.asarray(rows, dtype=np.float32).reshape(-1, 7),
        infer_ms=np.asarray(infer_ms, dtype=np.float32),
        meta=json.dumps(meta),
    )
    print(f'[CACHE] wrote {out_path} | {idx} frames | {len(rows)} detections | '
          f'median infer {meta["median_infer_ms"]:.1f} ms')


if __name__ == '__main__':
    main()
