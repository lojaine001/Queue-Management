"""Run one of today's detectors on the fisheye datasets and save what it finds.

    python eval/fisheye/run_detector.py --detector head --model "<path>\\yolov9_s_discrete_headpose_post_0100_1x3x480x640.onnx"
    python eval/fisheye/run_detector.py --detector entrance --model "<path>\\yolov9t.onnx"
    python eval/fisheye/run_detector.py --detector head --model ... --dataset CEPDOF --step 10

  head      checkout head detector (Head-Detector/utils/yolo.py), thresholds from Head-Detector/config.yml,
            heads only (classid 0), as in Head-Detector/main.py
  entrance  entrance person detector (utils/yolov9.py of the entrance app), thresholds from its config.yml,
            persons only (class 0), as in queue_management_v2.py

Each picture is resized to 640x480 first, as the live loops do; boxes are scaled back to the
picture's own size. One process runs one detector (the two apps both have a package named utils).

Output: eval/data/fisheye/dets/<detector>/<dataset>/<scene>.npz with the frame file names, boxes and
scores, no pixels. Score it with score_detections.py; look at it with view.py --dets.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAD_DIR = REPO_ROOT / 'Head-Detector'
ENTRANCE_DIR = REPO_ROOT / 'Queue-Management-System-v2-main' / 'Queue-Management-System-v2-main'
LIVE_SIZE = (640, 480)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def providers_for(device):
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']


def build_head(model_path, device):
    sys.path.insert(0, str(HEAD_DIR))
    from utils.queue_utils import get_config
    from utils.yolo import YOLOv9
    cfg = get_config(str(HEAD_DIR / 'config.yml'))
    score_th = float(cfg.get('score', 0.15))
    disable_hi = cfg.get('disable_headpose_identification_mode', False)
    # Same model thresholds as Head-Detector/main.py.
    model = YOLOv9(model_path=str(model_path), obj_class_score_th=0.35, attr_class_score_th=0.70,
                   providers=providers_for(device))

    def detect(img640):
        out = []
        for b in model(image=img640, disable_headpose_identification_mode=disable_hi):
            if b.classid != 0 or b.score < score_th:
                continue
            out.append([b.x1, b.y1, b.x2, b.y2, b.score])
        return out

    settings = {'score': score_th, 'obj_class_score_th': 0.35, 'attr_class_score_th': 0.70,
                'disable_headpose_identification_mode': disable_hi, 'classes_kept': [0]}
    return detect, settings


def build_entrance(model_path, device):
    sys.path.insert(0, str(ENTRANCE_DIR))
    from utils.queue_utils import get_config
    from utils.yolov9 import YOLOv9
    cfg = get_config(str(ENTRANCE_DIR / 'config.yml'))
    score = float(cfg.get('score', 0.3))
    min_score = float(cfg.get('min_score', 0.3))
    iou_score = float(cfg.get('iou_score', 0.3))
    # Same arguments as OnnxDetector in queue_management_v2.py.
    model = YOLOv9(model_path=str(model_path), score_threshold=score, conf_thresold=min_score,
                   iou_threshold=iou_score, device=device)

    def detect(img640):
        h, w = img640.shape[:2]
        out = []
        for d in model.detect(img640, h, w):
            if int(d['class_index']) != 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in d['box'])
            out.append([x1, y1, x2, y2, float(d['confidence'])])
        return out

    settings = {'score': score, 'min_score': min_score, 'iou_score': iou_score, 'classes_kept': [0]}
    return detect, settings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detector', choices=['head', 'entrance'], required=True)
    ap.add_argument('--model', required=True, help='.onnx file of that detector')
    ap.add_argument('--root', default=ds.DEFAULT_ROOT)
    ap.add_argument('--dataset', choices=ds.DATASETS, default='WEPDTOF')
    ap.add_argument('--scenes', nargs='*', help='default: every scene of the dataset')
    ap.add_argument('--step', type=int, default=1, help='use every Nth labelled frame')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--out-dir', help='default: eval/data/fisheye/dets/<detector>/<dataset>')
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f'Model not found: {model_path}')
    build = build_head if args.detector == 'head' else build_entrance
    detect, settings = build(model_path, args.device)

    scenes = ds.list_scenes(args.dataset, args.root)
    if args.scenes:
        wanted = {s.lower() for s in args.scenes}
        scenes = [s for s in scenes if s.name.lower() in wanted]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / 'eval' / 'data' / 'fisheye' / 'dets' / args.detector / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    model_sha = sha256_file(model_path)

    for scene in scenes:
        frames = ds.load_frames(scene)[::args.step]
        files, counts, rows, infer_ms = [], [], [], []
        t0 = time.time()
        for fr in frames:
            img = cv2.imread(str(fr.path))
            if img is None:
                continue
            H, W = img.shape[:2]
            small = cv2.resize(img, LIVE_SIZE)
            t = time.perf_counter()
            dets = detect(small)
            infer_ms.append((time.perf_counter() - t) * 1000)
            sx, sy = W / LIVE_SIZE[0], H / LIVE_SIZE[1]
            for x1, y1, x2, y2, s in dets:
                rows.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy, s])
            files.append(fr.file_name)
            counts.append(len(dets))
        offsets = np.zeros(len(files) + 1, np.int64)
        offsets[1:] = np.cumsum(counts)
        meta = {
            'detector': args.detector, 'dataset': args.dataset, 'scene': scene.name,
            'model': model_path.name, 'model_sha256': model_sha, 'device': args.device,
            'settings': settings, 'step': args.step, 'live_resize': list(LIVE_SIZE),
            'n_frames': len(files), 'median_infer_ms': float(np.median(infer_ms)) if infer_ms else None,
            'created': datetime.now().isoformat(timespec='seconds'),
            'det_columns': ['x1', 'y1', 'x2', 'y2', 'score'],
        }
        np.savez_compressed(out_dir / f'{scene.name}.npz', files=np.array(files), offsets=offsets,
                            dets=np.array(rows, np.float32).reshape(-1, 5), meta=json.dumps(meta))
        print(f'{scene.name:<20} {len(files):>5} frames  {sum(counts):>6} detections  '
              f'{time.time() - t0:5.0f}s  median {meta["median_infer_ms"] or 0:.0f} ms/frame')
    print(f'-> {out_dir}')


if __name__ == '__main__':
    main()
