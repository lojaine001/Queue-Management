"""Run RAPiD (rotation-aware people detector for overhead fisheye) on the fisheye datasets.

    python eval/fisheye/run_rapid.py --dataset WEPDTOF
    python eval/fisheye/run_rapid.py --dataset CEPDOF --step 10

Needs the RAPiD code (https://github.com/duanzhiihao/RAPiD) cloned next to backend-v3 (or --rapid-dir)
and the pretrained weights in its weights/ folder. Default weights: pL1_MWHB1024_Mar11_4000.ckpt,
trained on COCO + MW-R + HABBOF, so WEPDTOF and CEPDOF are unseen test data for it.
RAPiD code and weights are for non-commercial use only.

Unlike today's detectors, RAPiD sees the whole picture (padded to a square, input_size 1024).

Output has the same layout as run_detector.py (x1, y1, x2, y2, score = the upright box around each
rotated detection), plus `rboxes` (cx, cy, w, h, angle_deg, score), so score_detections.py and view.py
work unchanged.
"""
import argparse
import hashlib
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAPID_DIR = REPO_ROOT.parent / 'RAPiD'


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def rotated_iou(a, b):
    """IoU of two rotated boxes (cx, cy, w, h, angle_deg)."""
    ra = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), float(a[4]))
    rb = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(b[4]))
    kind, pts = cv2.rotatedRectangleIntersection(ra, rb)
    if kind == cv2.INTERSECT_NONE or pts is None:
        return 0.0
    inter = cv2.contourArea(cv2.convexHull(pts))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def rotated_nms(dts, thres):
    """Greedy NMS on rows (cx, cy, w, h, angle_deg, score), highest score first.
    Same procedure as RAPiD's utils.nms, but with exact polygon overlap instead of
    pycocotools rasterised masks, so it needs no extra package."""
    if len(dts) == 0:
        return dts
    dts = dts[np.argsort(-dts[:, 5])]
    centres = dts[:, :2]
    radius = 0.5 * np.hypot(dts[:, 2], dts[:, 3])  # boxes further apart than r_i + r_j cannot overlap
    keep, suppressed = [], np.zeros(len(dts), bool)
    for i in range(len(dts)):
        if suppressed[i]:
            continue
        keep.append(i)
        near = np.nonzero(~suppressed[i + 1:] &
                          (np.hypot(*(centres[i + 1:] - centres[i]).T) < radius[i + 1:] + radius[i]))[0] + i + 1
        for j in near:
            if rotated_iou(dts[i], dts[j]) > thres:
                suppressed[j] = True
    return dts[keep]


def load_rapid(rapid_dir, weights, device):
    # RAPiD imports pycocotools at module level for its mask-based IoU; we use rotated_nms
    # instead, so a stub avoids installing pycocotools.
    if 'pycocotools' not in sys.modules:
        pc = types.ModuleType('pycocotools')
        pc.mask = types.ModuleType('pycocotools.mask')
        sys.modules['pycocotools'] = pc
        sys.modules['pycocotools.mask'] = pc.mask
    sys.path.insert(0, str(rapid_dir))
    import torch
    from models.rapid import RAPiD
    from utils import utils as rutils

    model = RAPiD(backbone='dark53')
    # weights_only=True: load tensors only, never run code from the downloaded file.
    state = torch.load(weights, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    model.eval().to(device)
    return model, rutils, torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ds.DEFAULT_ROOT)
    ap.add_argument('--dataset', choices=ds.DATASETS, default='WEPDTOF')
    ap.add_argument('--scenes', nargs='*', help='default: every scene of the dataset')
    ap.add_argument('--step', type=int, default=1, help='use every Nth labelled frame')
    ap.add_argument('--rapid-dir', default=str(DEFAULT_RAPID_DIR))
    ap.add_argument('--weights', default='pL1_MWHB1024_Mar11_4000.ckpt', help='file name in <rapid-dir>/weights, or a path')
    ap.add_argument('--input-size', type=int, default=1024)
    ap.add_argument('--conf', type=float, default=0.3, help='score threshold (0.3 as in the RAPiD example)')
    ap.add_argument('--nms', type=float, default=0.45)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--out-dir', help='default: eval/data/fisheye/dets/rapid/<dataset>')
    args = ap.parse_args()

    rapid_dir = Path(args.rapid_dir)
    weights = Path(args.weights)
    if not weights.exists():
        weights = rapid_dir / 'weights' / args.weights
    if not weights.exists():
        sys.exit(f'Weights not found: {weights}')
    model, rutils, torch = load_rapid(rapid_dir, weights, args.device)
    import torchvision.transforms.functional as tvf
    from PIL import Image

    scenes = ds.list_scenes(args.dataset, args.root)
    if args.scenes:
        wanted = {s.lower() for s in args.scenes}
        scenes = [s for s in scenes if s.name.lower() in wanted]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / 'eval' / 'data' / 'fisheye' / 'dets' / 'rapid' / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_sha = sha256_file(weights)

    for scene in scenes:
        frames = ds.load_frames(scene)[::args.step]
        files, counts, rows, rrows, infer_ms = [], [], [], [], []
        t0 = time.time()
        for fr in frames:
            if not fr.path.exists():
                continue
            pil = Image.open(fr.path).convert('RGB')
            # Same steps as Detector._predict_pil in RAPiD's api.py.
            square, _, pad_info = rutils.rect_to_square(pil, None, args.input_size, 0)
            x = tvf.to_tensor(square).unsqueeze(0).to(args.device)
            t = time.perf_counter()
            with torch.no_grad():
                out = model(x).cpu().squeeze()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            infer_ms.append((time.perf_counter() - t) * 1000)
            out = out[out[:, 5] >= args.conf]
            if len(out) > 1000:
                _, idx = torch.topk(out[:, 5], k=1000)
                out = out[idx, :]
            out = torch.from_numpy(rotated_nms(out.numpy(), args.nms))
            out = rutils.detection2original(out, pad_info.squeeze())
            n = 0
            for cx, cy, w, h, a, s in out.numpy().reshape(-1, 6):
                pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(a)))
                x1, y1 = pts.min(0)
                x2, y2 = pts.max(0)
                rows.append([x1, y1, x2, y2, s])
                rrows.append([cx, cy, w, h, a, s])
                n += 1
            files.append(fr.file_name)
            counts.append(n)
        offsets = np.zeros(len(files) + 1, np.int64)
        offsets[1:] = np.cumsum(counts)
        meta = {
            'detector': 'rapid', 'dataset': args.dataset, 'scene': scene.name,
            'model': weights.name, 'model_sha256': weights_sha, 'device': args.device,
            'settings': {'input_size': args.input_size, 'conf': args.conf, 'nms': args.nms},
            'step': args.step, 'n_frames': len(files),
            'median_infer_ms': float(np.median(infer_ms)) if infer_ms else None,
            'torch': torch.__version__, 'created': datetime.now().isoformat(timespec='seconds'),
            'det_columns': ['x1', 'y1', 'x2', 'y2', 'score'],
            'rbox_columns': ['cx', 'cy', 'w', 'h', 'angle_deg', 'score'],
        }
        np.savez_compressed(out_dir / f'{scene.name}.npz', files=np.array(files), offsets=offsets,
                            dets=np.array(rows, np.float32).reshape(-1, 5),
                            rboxes=np.array(rrows, np.float32).reshape(-1, 6), meta=json.dumps(meta))
        print(f'{scene.name:<20} {len(files):>5} frames  {sum(counts):>6} detections  '
              f'{time.time() - t0:5.0f}s  median {meta["median_infer_ms"] or 0:.0f} ms/frame', flush=True)
    print(f'-> {out_dir}')


if __name__ == '__main__':
    main()
