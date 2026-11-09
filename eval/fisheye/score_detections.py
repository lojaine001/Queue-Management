"""Score a detector's output (run_detector.py) against the fisheye labels.

    python eval/fisheye/score_detections.py --dets-dir eval/data/fisheye/dets/head/WEPDTOF
    python eval/fisheye/score_detections.py --dets-dir eval/data/fisheye/dets/entrance/WEPDTOF --scenes convenience_store tech_store

Matching: a detection counts for a labelled person when the centre of the detection box lies
inside that person's rotated label box (one detection per person, nearest centre first). This
works the same for a head detector and a full-body detector, since the head is inside the body
box when seen from above. It is more lenient than an IoU test, so it measures "was this person
found", not "how tight is the box". Only people and detections inside the scene's ROI map count.

Per scene and overall it reports:
  recall      share of labelled people that were found
  precision   share of detections that are a real person
  count MAE   mean |detections - people| per frame: what a queue count would be off by
  count bias  mean (detections - people): negative = undercounting
  exact       share of frames where the count is exactly right
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dets(path):
    """{file_name: float32[N, 5] (x1, y1, x2, y2, score) in original image pixels}"""
    z = np.load(path, allow_pickle=False)
    files, offsets, dets = z['files'], z['offsets'], z['dets']
    return {str(f): dets[offsets[i]:offsets[i + 1]] for i, f in enumerate(files)}


def load_meta(path):
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z['meta']))


def match_frame(frame, dets, roi):
    """Returns (matches [(det_idx, gt_idx)], det indices in ROI, gt indices in ROI)."""
    gt_in = [g for g, b in enumerate(frame.boxes) if ds.inside_roi(roi, b[0], b[1])]
    centres = [((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in dets]
    det_in = [i for i, (x, y) in enumerate(centres) if ds.inside_roi(roi, x, y)]
    if not gt_in or not det_in:
        return [], det_in, gt_in
    big = 1e9
    cost = np.full((len(det_in), len(gt_in)), big)
    for a, i in enumerate(det_in):
        x, y = centres[i]
        for b, g in enumerate(gt_in):
            box = frame.boxes[g]
            if ds.inside_box(box, x, y):
                cost[a, b] = np.hypot(x - box[0], y - box[1])
    rows, cols = linear_sum_assignment(cost)
    matches = [(det_in[r], gt_in[c]) for r, c in zip(rows, cols) if cost[r, c] < big]
    return matches, det_in, gt_in


def score_scene(scene, det_path):
    frames = ds.load_frames(scene)
    by_file = load_dets(det_path)
    roi = None
    n_gt = n_det = n_match = 0
    count_err, frames_scored = [], 0
    for fr in frames:
        if fr.file_name not in by_file:
            continue  # frame skipped by run_detector --step
        if roi is None and scene.roi_path is not None:
            roi = ds.load_roi(scene, fr.width, fr.height)
        matches, det_in, gt_in = match_frame(fr, by_file[fr.file_name], roi)
        n_gt += len(gt_in)
        n_det += len(det_in)
        n_match += len(matches)
        count_err.append(len(det_in) - len(gt_in))
        frames_scored += 1
    e = np.array(count_err, float) if count_err else np.zeros(1)
    return {
        'scene': scene.name,
        'frames': frames_scored,
        'people': n_gt,
        'detections': n_det,
        'found': n_match,
        'recall': round(n_match / n_gt, 3) if n_gt else None,
        'precision': round(n_match / n_det, 3) if n_det else None,
        'count_mae': round(float(np.abs(e).mean()), 2),
        'count_bias': round(float(e.mean()), 2),
        'exact_share': round(float((e == 0).mean()), 3),
        'avg_people_per_frame': round(n_gt / frames_scored, 2) if frames_scored else None,
    }


def total(rows, name):
    n_gt = sum(r['people'] for r in rows)
    n_det = sum(r['detections'] for r in rows)
    n_m = sum(r['found'] for r in rows)
    nf = sum(r['frames'] for r in rows)
    # frame-weighted averages of the per-frame count metrics
    w = lambda k: round(sum(r[k] * r['frames'] for r in rows) / nf, 2 if k != 'exact_share' else 3) if nf else None
    return {'scene': name, 'frames': nf, 'people': n_gt, 'detections': n_det, 'found': n_m,
            'recall': round(n_m / n_gt, 3) if n_gt else None,
            'precision': round(n_m / n_det, 3) if n_det else None,
            'count_mae': w('count_mae'), 'count_bias': w('count_bias'), 'exact_share': w('exact_share'),
            'avg_people_per_frame': round(n_gt / nf, 2) if nf else None}


def fmt(v, pct=False):
    if v is None:
        return '-'
    return f'{100 * v:.0f}%' if pct else f'{v}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ds.DEFAULT_ROOT)
    ap.add_argument('--dets-dir', required=True, help='folder with <scene>.npz from run_detector.py')
    ap.add_argument('--scenes', nargs='*', help='only these scenes (default: every .npz in --dets-dir)')
    ap.add_argument('--out-dir', help='default: eval/runs/fisheye/<detector>/<dataset>')
    args = ap.parse_args()

    dets_dir = Path(args.dets_dir)
    files = sorted(p for p in dets_dir.glob('*.npz'))
    if args.scenes:
        wanted = {s.lower() for s in args.scenes}
        files = [p for p in files if p.stem.lower() in wanted]
    if not files:
        sys.exit(f'No .npz files in {dets_dir}')
    meta = load_meta(files[0])
    dataset = meta['dataset']

    rows = []
    for p in files:
        scene = ds.get_scene(dataset, p.stem, args.root)
        rows.append(score_scene(scene, p))
        print(f'  scored {p.stem}')
    shop = [r for r in rows if r['scene'] in ds.SHOP_SCENES]
    summary = [total(rows, 'ALL')]
    if shop and len(shop) < len(rows):
        summary.append(total(shop, 'SHOP SCENES'))

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / 'eval' / 'runs' / 'fisheye' / meta['detector'] / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_dir / 'scores.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows + summary:
            w.writerow(r)
    with open(out_dir / 'scores.json', 'w', encoding='utf-8') as f:
        json.dump({'detector_meta': meta, 'scenes': rows, 'summary': summary}, f, indent=2)

    print(f"\n{meta['detector']} on {dataset}  (model {meta['model']}, step {meta['step']})")
    head = f"{'scene':<20} {'frames':>6} {'people/fr':>9} {'recall':>7} {'precision':>9} {'count MAE':>9} {'bias':>6} {'exact':>6}"
    print(head)
    print('-' * len(head))
    for r in rows + summary:
        if r['scene'] == 'ALL':
            print('-' * len(head))
        print(f"{r['scene']:<20} {r['frames']:>6} {fmt(r['avg_people_per_frame']):>9} {fmt(r['recall'], True):>7} "
              f"{fmt(r['precision'], True):>9} {fmt(r['count_mae']):>9} {r['count_bias']:>+6.2f} {fmt(r['exact_share'], True):>6}")
    print(f'\n-> {out_dir}')


if __name__ == '__main__':
    main()
