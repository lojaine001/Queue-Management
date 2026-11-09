"""Look at a fisheye scene: labelled people (rotated boxes + person id) and, optionally,
what a detector found (from run_detector.py).

    python eval/fisheye/view.py --dataset WEPDTOF --scene convenience_store
    python eval/fisheye/view.py --dataset WEPDTOF --scene convenience_store --dets eval/data/fisheye/dets/head/WEPDTOF/convenience_store.npz
    python eval/fisheye/view.py --dataset WEPDTOF --scene convenience_store --frame 100 --png out.png   # save one frame

Keys: space play/pause, d / a next / previous frame, w / s jump 10 frames, + / - speed,
r show/hide the scored area, q quit.

Colours: one colour per labelled person. Detections: green dot = matched a person,
red dot = no person there (false detection). Labelled people with no detection get a red box.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402
from score_detections import load_dets, match_frame  # noqa: E402


def colour(pid):
    rng = np.random.default_rng(int(pid) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw(frame, img, roi, dets, show_roi, idx, total, scene):
    out = img.copy()
    if show_roi and roi is not None:
        shade = out.copy()
        shade[~roi] = (shade[~roi] * 0.35).astype(np.uint8)
        out = shade
    missed = set()
    det_ok = []
    if dets is not None:
        matches, det_in, gt_in = match_frame(frame, dets, roi)
        matched_gt = {g for _, g in matches}
        matched_det = {d for d, _ in matches}
        missed = {g for g in gt_in if g not in matched_gt}
        det_ok = [(d, d in matched_det) for d in det_in]
    scale = max(1, round(img.shape[1] / 960))
    for g, (box, pid) in enumerate(zip(frame.boxes, frame.person_ids)):
        c = (0, 0, 255) if g in missed else colour(pid)
        pts = ds.box_corners(box).astype(np.int32)
        cv2.polylines(out, [pts], True, c, 2 * scale, cv2.LINE_AA)
        cv2.putText(out, str(pid), (int(box[0]) - 8, int(box[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, c, 2 * scale, cv2.LINE_AA)
    if dets is not None:
        for d, ok in det_ok:
            x1, y1, x2, y2, _ = dets[d]
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1 * scale)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cv2.circle(out, (int(cx), int(cy)), 6 * scale, (0, 200, 0) if ok else (0, 0, 255), -1)
    n_gt = len(frame.boxes)
    txt = f'{scene.dataset}/{scene.name}  frame {idx + 1}/{total}  {frame.file_name}  people {n_gt}'
    if dets is not None:
        txt += f'  detections {len(det_ok)}  missed {len(missed)}  false {sum(1 for _, ok in det_ok if not ok)}'
    cv2.rectangle(out, (0, 0), (out.shape[1], 30 * scale), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, 21 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 1 * scale, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ds.DEFAULT_ROOT)
    ap.add_argument('--dataset', choices=ds.DATASETS, default='WEPDTOF')
    ap.add_argument('--scene', help='scene name; omit to list the scenes')
    ap.add_argument('--dets', help='detections .npz from run_detector.py')
    ap.add_argument('--frame', type=int, default=0, help='start frame (0-based)')
    ap.add_argument('--png', help='save --frame to this file and exit')
    ap.add_argument('--max-width', type=int, default=1400, help='shrink the window to this width')
    args = ap.parse_args()

    if not args.scene:
        for s in ds.list_scenes(args.dataset, args.root):
            print(s.name)
        return
    scene = ds.get_scene(args.dataset, args.scene, args.root)
    frames = ds.load_frames(scene)
    det_by_file = load_dets(args.dets) if args.dets else None
    roi = None
    idx, playing, delay, show_roi = max(0, min(args.frame, len(frames) - 1)), False, 100, True

    def render(i):
        nonlocal roi
        fr = frames[i]
        img = cv2.imread(str(fr.path))
        if img is None:
            img = np.zeros((fr.height, fr.width, 3), np.uint8)
        if roi is None:
            roi = ds.load_roi(scene, img.shape[1], img.shape[0])
        dets = None
        if det_by_file is not None:
            dets = det_by_file.get(fr.file_name, np.zeros((0, 5), np.float32))
        return draw(fr, img, roi, dets, show_roi, i, len(frames), scene)

    if args.png:
        cv2.imwrite(args.png, render(idx))
        print(f'saved {args.png}')
        return

    win = f'{scene.dataset} / {scene.name}'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        out = render(idx)
        if out.shape[1] > args.max_width:
            f = args.max_width / out.shape[1]
            out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        cv2.imshow(win, out)
        key = cv2.waitKey(delay if playing else 0) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('d') or (playing and key == 255):
            idx = min(idx + 1, len(frames) - 1)
        elif key == ord('a'):
            idx = max(idx - 1, 0)
        elif key == ord('w'):
            idx = min(idx + 10, len(frames) - 1)
        elif key == ord('s'):
            idx = max(idx - 10, 0)
        elif key in (ord('+'), ord('=')):
            delay = max(10, delay // 2)
        elif key == ord('-'):
            delay = min(2000, delay * 2)
        elif key == ord('r'):
            show_roi = not show_roi
        if playing and idx == len(frames) - 1:
            playing = False
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
