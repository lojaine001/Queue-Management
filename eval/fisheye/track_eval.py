"""Test the checkout trackers on fisheye detections and score them against the labelled person IDs.

    python eval/fisheye/track_eval.py --dets-dir eval/data/fisheye/dets/rapid_track/CEPDOF --fps 10 --step 4

Detections come from run_rapid.py (run with the same --step, and --conf 0.1 so ByteTrack gets its
low-score boxes). The trackers are the ones built into Head-Detector/main.py, created with the same
arguments as there, fed the upright box around each detection:

  norfair_live     Norfair, production settings, each picture processed 14 times (the live loop runs
                   ~33.6 times/s on a ~2.3 pictures/s stream; see eval/step1_summary.md)
  norfair          Norfair, production settings, once per picture
  norfair_2s       as norfair, but the tracker remembers a lost person for 2 s (one change)
  sort             SORT backend of main.py (shared IoU threshold 1/max_distance_between_points)
  bytetrack        ByteTrack backend of main.py (high/low score thresholds from the detector floor)
  bytetrack_2s     as bytetrack, memory 2 s

Norfair, SORT and ByteTrack get boxes with score >= --score (detector default 0.3); ByteTrack also
gets the boxes between its low threshold and --score, which is how it is meant to work.

Each tracker's output in every picture is matched to the labelled people with the same rule as
score_detections.py (box centre inside the person's rotated label box, one-to-one). A track's
duration runs from the first to the last picture it is reported in, which is what the live loop
logs as dwell (start to track death). People labelled -1 (unidentified) are left out, and a person
absent from the labels for more than --visit-gap seconds left and came back, so each visit is scored
separately. Per visit with a real stay >= --min-dwell:

  dwell error     duration of the visit's main track (the one matched most) minus the real stay
  tracks/visit    different track IDs the person was given (1 = perfect)
  ID switches     times the matched ID changes from one picture to the next
  ID kept         average time the person keeps one ID before it changes (s)
  coverage        share of the visit's pictures in which some track was on them

Overall: logged rows per visit = tracks lasting >= --min-dwell divided by visits lasting >= --min-dwell
(what the database would get; 1.0 = one row per person), and false tracks = such tracks that sit on
nobody for most of their life.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402
from score_detections import load_dets, match_frame  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAD_DIR = REPO_ROOT / 'Head-Detector'
LIVE_REPEATS = 14  # loop passes per camera picture measured on production (33.6 / 2.32)


def production_settings():
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HEAD_DIR))
    from utils.queue_utils import get_config
    cfg2 = get_config(str(HEAD_DIR / 'config2.yml'))
    return {'max_age': int(cfg2.get('max_age', 4)),
            'max_distance_between_points': float(cfg2.get('max_distance_between_points', 6))}


def make_tracker(kind, prod, score_floor, pictures_2s):
    """Same constructor arguments as Head-Detector/main.py."""
    max_age = pictures_2s if kind.endswith('_2s') else prod['max_age']
    dist = prod['max_distance_between_points']
    if kind.startswith('norfair'):
        from norfair import Detection, Tracker
        from utils.queue_utils import iou
        tracker = Tracker(distance_function=iou, distance_threshold=dist, hit_counter_max=max_age,
                          past_detections_length=max_age, initialization_delay=0)

        def to_dets(boxes):
            return [Detection(points=np.array(b[:4], float).reshape(2, 2), data={'confidence': float(b[4])})
                    for b in boxes if b[4] >= score_floor]
        return tracker, to_dets
    from utils.tracker_base import TrackerDetection
    if kind == 'sort':
        from utils.tracker_sort import SORTTracker
        tracker = SORTTracker(max_age=max_age, min_hits=0, iou_threshold=1.0 / max(dist, 1e-6))
        low = score_floor
    else:
        from utils.tracker_bytetrack import ByteTracker
        thr = 1.0 / max(dist, 1e-6)
        low = max(score_floor - 0.15, 0.10)
        tracker = ByteTracker(max_age=max_age, min_hits=1, track_high_thresh=score_floor, track_low_thresh=low,
                              iou_threshold_high=thr, iou_threshold_low=max(thr - 0.10, 0.10), iou_threshold_lost=thr)

    def to_dets(boxes):
        return [TrackerDetection(bbox=b[:4], score=float(b[4]), data={}) for b in boxes if b[4] >= low]
    return tracker, to_dets


def run_tracker(kind, frames, by_file, prod, score_floor, pictures_2s):
    """Per picture: list of (track_id, x1, y1, x2, y2) the tracker reports."""
    tracker, to_dets = make_tracker(kind, prod, score_floor, pictures_2s)
    repeats = LIVE_REPEATS if kind == 'norfair_live' else 1
    out = []
    for fr in frames:
        dets = to_dets(by_file.get(fr.file_name, np.zeros((0, 5), np.float32)))
        for _ in range(repeats):
            objs = tracker.update(detections=dets)
        rows = []
        for o in objs:
            e = np.asarray(o.estimate, float).reshape(-1)
            rows.append((int(o.global_id), e[0], e[1], e[2], e[3]))
        out.append(rows)
    return out


def score_tracks(frames, tracks_per_frame, dt, min_dwell, visit_gap):
    gt_seq = defaultdict(list)            # pid -> [(picture idx, track id or None)]
    track_frames = defaultdict(list)      # tid -> picture idxs reported
    track_on_person = defaultdict(int)    # tid -> pictures matched to someone
    for i, (fr, rows) in enumerate(zip(frames, tracks_per_frame)):
        boxes = np.array([[r[1], r[2], r[3], r[4], 1.0] for r in rows], np.float32).reshape(-1, 5)
        matches, _, _ = match_frame(fr, boxes, None)
        by_gt = {g: rows[d][0] for d, g in matches}
        for g, pid in enumerate(fr.person_ids):
            gt_seq[int(pid)].append((i, by_gt.get(g)))
        for r in rows:
            track_frames[r[0]].append(i)
        for d, _ in matches:
            track_on_person[rows[d][0]] += 1

    track_dur = {t: (f[-1] - f[0] + 1) * dt for t, f in track_frames.items()}
    people = []
    for pid, seq in gt_seq.items():
        if pid < 0:
            continue  # person_id -1 = unidentified person in the labels
        # A person who is out of the labels for longer than visit_gap pictures left and came back:
        # each stay is scored as its own visit.
        visits, cur = [], [seq[0]]
        for prev, item in zip(seq, seq[1:]):
            if item[0] - prev[0] > visit_gap:
                visits.append(cur)
                cur = []
            cur.append(item)
        visits.append(cur)
        for v_no, vis in enumerate(visits):
            real = (vis[-1][0] - vis[0][0] + 1) * dt
            ids = [t for _, t in vis if t is not None]
            counts = Counter(ids)
            switches = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
            runs = switches + 1 if ids else 0
            rec = {'person_id': pid, 'visit': v_no, 'real_dwell_s': round(real, 2), 'pictures': len(vis),
                   'coverage': round(len(ids) / len(vis), 3), 'n_track_ids': len(counts), 'id_switches': switches,
                   'id_hold_s': round(len(ids) * dt / runs, 2) if runs else None}
            if counts:
                main = counts.most_common(1)[0][0]
                rec['main_track_s'] = round(track_dur[main], 2)
                rec['dwell_error_s'] = round(track_dur[main] - real, 2)
            people.append(rec)

    scored = [p for p in people if p['real_dwell_s'] >= min_dwell]
    found = [p for p in scored if 'dwell_error_s' in p]
    long_tracks = [t for t, d in track_dur.items() if d >= min_dwell]
    false_tracks = [t for t in long_tracks if track_on_person[t] < 0.5 * len(track_frames[t])]
    err = np.array([p['dwell_error_s'] for p in found]) if found else np.zeros(0)
    summary = {
        'people_scored': len(scored),
        'people_without_track': len(scored) - len(found),
        'dwell_mae_s': round(float(np.abs(err).mean()), 2) if len(err) else None,
        'dwell_median_abs_s': round(float(np.median(np.abs(err))), 2) if len(err) else None,
        'dwell_bias_s': round(float(err.mean()), 2) if len(err) else None,
        'dwell_mae_pct': round(100 * float(np.mean([abs(p['dwell_error_s']) / p['real_dwell_s'] for p in found])), 1) if found else None,
        'track_ids_per_person': round(float(np.mean([p['n_track_ids'] for p in found])), 2) if found else None,
        'people_with_one_id': round(float(np.mean([p['n_track_ids'] == 1 for p in found])), 3) if found else None,
        'id_hold_s': round(float(np.mean([p['id_hold_s'] for p in found])), 1) if found else None,
        'present_min': round(sum(p['pictures'] for p in scored) * dt / 60, 2),
        'id_switches': int(sum(p['id_switches'] for p in scored)),
        'coverage': round(float(np.mean([p['coverage'] for p in scored])), 3) if scored else None,
        'logged_rows': len(long_tracks),
        'logged_rows_per_person': round(len(long_tracks) / len(scored), 2) if scored else None,
        'false_tracks': len(false_tracks),
    }
    return summary, people


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ds.DEFAULT_ROOT)
    ap.add_argument('--dets-dir', required=True, help='<scene>.npz from run_rapid.py')
    ap.add_argument('--fps', type=float, required=True, help='frame rate of the labelled videos (CEPDOF: 10)')
    ap.add_argument('--step', type=int, required=True, help='the --step the detections were made with')
    ap.add_argument('--trackers', nargs='+',
                    default=['norfair_live', 'norfair', 'norfair_2s', 'sort', 'bytetrack', 'bytetrack_2s'])
    ap.add_argument('--score', type=float, default=0.3, help='detector score floor (RAPiD default 0.3)')
    ap.add_argument('--min-dwell', type=float, default=3.0, help='queue_min_dwell_sec in production')
    ap.add_argument('--visit-gap', type=float, default=2.0,
                    help='a labelled person absent longer than this (s) left and came back: a new visit')
    ap.add_argument('--out-dir', help='default: eval/runs/fisheye/tracking/<dataset>')
    args = ap.parse_args()

    prod = production_settings()
    dt = args.step / args.fps
    pictures_2s = max(1, round(2.0 / dt))
    visit_gap = max(1, round(args.visit_gap / dt))
    files = sorted(Path(args.dets_dir).glob('*.npz'))
    if not files:
        sys.exit(f'No .npz in {args.dets_dir}')
    meta = json.loads(str(np.load(files[0])['meta']))
    dataset = meta['dataset']
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / 'eval' / 'runs' / 'fisheye' / 'tracking' / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{dataset}: {1 / dt:.2f} pictures/s, production max_age={prod["max_age"]}, '
          f'distance={prod["max_distance_between_points"]}, 2 s memory = {pictures_2s} pictures')

    results, per_person_rows = [], []
    for p in files:
        scene = ds.get_scene(dataset, p.stem, args.root)
        by_file = load_dets(p)
        frames = [fr for fr in ds.load_frames(scene) if fr.file_name in by_file]
        for kind in args.trackers:
            tracks = run_tracker(kind, frames, by_file, prod, args.score, pictures_2s)
            summary, people = score_tracks(frames, tracks, dt, args.min_dwell, visit_gap)
            results.append({'scene': scene.name, 'tracker': kind, **summary})
            for r in people:
                per_person_rows.append({'scene': scene.name, 'tracker': kind, **r})
            print(f'  {scene.name:<14} {kind:<13} rows/person {summary["logged_rows_per_person"]}  '
                  f'dwell MAE {summary["dwell_mae_s"]}s  IDs/person {summary["track_ids_per_person"]}', flush=True)

    # totals per tracker, over all scenes
    totals = []
    for kind in args.trackers:
        rows = [r for r in results if r['tracker'] == kind]
        ppl = [r for r in per_person_rows if r['tracker'] == kind and r['real_dwell_s'] >= args.min_dwell]
        found = [r for r in ppl if 'dwell_error_s' in r]
        err = np.array([r['dwell_error_s'] for r in found]) if found else np.zeros(0)
        n_people = sum(r['people_scored'] for r in rows)
        totals.append({
            'tracker': kind, 'people_scored': n_people,
            'logged_rows_per_person': round(sum(r['logged_rows'] for r in rows) / n_people, 2) if n_people else None,
            'dwell_mae_s': round(float(np.abs(err).mean()), 2) if len(err) else None,
            'dwell_median_abs_s': round(float(np.median(np.abs(err))), 2) if len(err) else None,
            'dwell_bias_s': round(float(err.mean()), 2) if len(err) else None,
            'dwell_mae_pct': round(100 * float(np.mean([abs(r['dwell_error_s']) / r['real_dwell_s'] for r in found])), 1) if found else None,
            'track_ids_per_person': round(float(np.mean([r['n_track_ids'] for r in found])), 2) if found else None,
            'people_with_one_id': round(float(np.mean([r['n_track_ids'] == 1 for r in found])), 3) if found else None,
            'id_hold_s': round(float(np.mean([r['id_hold_s'] for r in found])), 1) if found else None,
            'switches_per_min': round(sum(r['id_switches'] for r in rows) / max(sum(r['present_min'] for r in rows), 1e-9), 2),
            'id_switches': sum(r['id_switches'] for r in rows),
            'false_tracks': sum(r['false_tracks'] for r in rows),
            'coverage': round(float(np.mean([r['coverage'] for r in ppl])), 3) if ppl else None,
        })

    with open(out_dir / 'tracking_scenes.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    fields = ['scene', 'tracker', 'person_id', 'visit', 'real_dwell_s', 'pictures', 'coverage', 'n_track_ids',
              'id_switches', 'id_hold_s', 'main_track_s', 'dwell_error_s']
    with open(out_dir / 'tracking_people.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_person_rows)
    with open(out_dir / 'tracking_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'dataset': dataset, 'pictures_per_s': 1 / dt, 'production': prod, 'score_floor': args.score,
                   'min_dwell_s': args.min_dwell, 'pictures_2s': pictures_2s, 'detector_meta': meta,
                   'totals': totals, 'scenes': results}, f, indent=2)

    print(f"\n{'tracker':<13} {'visits':>6} {'rows/visit':>10} {'dwell MAE':>9} {'median':>7} {'MAE %':>6} "
          f"{'IDs/visit':>9} {'one ID':>7} {'ID kept':>8} {'sw/min':>7} {'false':>6}")
    for t in totals:
        print(f"{t['tracker']:<13} {t['people_scored']:>6} {t['logged_rows_per_person']:>10} {t['dwell_mae_s']:>8}s "
              f"{t['dwell_median_abs_s']:>6}s {t['dwell_mae_pct']:>5}% {t['track_ids_per_person']:>9} "
              f"{100 * t['people_with_one_id']:>6.0f}% {t['id_hold_s']:>7}s {t['switches_per_min']:>7} {t['false_tracks']:>6}")
    print(f'-> {out_dir}')


if __name__ == '__main__':
    main()
