"""Replay cached checkout detections through Norfair + the live post-processing.

Needs only the .dets.npz cache (no footage), so it can run on any machine with
numpy, shapely, pyyaml and norfair==2.3.0 (the production version).

    # baseline: production settings, every recorded frame
    python eval/track_replay.py --cache eval/data/cache/CLIP.dets.npz --run-name 000_baseline

    # replay at the live loop's real frame times (from the frame log)
    python eval/track_replay.py --cache ... --run-name 000_baseline_live \
        --frame-log Head-Detector/LOGs/frame_log_Bosch_Camera_exit_20261029.csv \
        --sidecar eval/data/raw/CLIP.json

    # one change at a time, via an override file
    python eval/track_replay.py --cache ... --run-name 001_dist4 --override eval/overrides/dist4.yml

Settings come from Head-Detector/config.yml and config2.yml exactly as the live system
reads them; an override YAML may replace any key listed in DEFAULT_KEYS below.

Outputs in eval/runs/<run-name>/<clip>/:
  tracks.csv  one row per (frame, track): box estimate, matched, state
  events.csv  one row per track death: dwell, lane, outcome (what would reach the DB)
  run.json    effective settings, versions, cache metadata, sampling mode
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import Point, Polygon

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAD_DIR = REPO_ROOT / 'Head-Detector'
sys.path.insert(0, str(HEAD_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from norfair import Detection, Tracker  # noqa: E402
from utils.queue_utils import get_config, iou  # noqa: E402  (same distance function as live)
from pipeline_logic import CheckoutPostProcessor  # noqa: E402

PROD_NORFAIR_VERSION = '2.3.0'

# Keys an override file may set. None = derived from another key, as main.py does.
DEFAULT_KEYS = {
    # config.yml
    'score': 0.3, 'min_delay': 0.0, 'rois': [],
    # config2.yml
    'max_distance_between_points': 2, 'max_age': 10,
    'tip_offset': 1.0, 'queue_min_dwell_sec': 3.0,
    # Norfair arguments that main.py derives or hard-codes; separate keys so each can be tuned alone
    'distance_threshold': None,       # main.py: max_distance_between_points
    'hit_counter_max': None,          # main.py: max_age
    'past_detections_length': None,   # main.py: max_age
    'initialization_delay': 0,        # main.py: hard-coded 0
}


def load_params(override_path):
    stored_config = get_config(str(HEAD_DIR / 'config.yml'))
    config2 = get_config(str(HEAD_DIR / 'config2.yml'))
    if config2.get('tracker', 'norfair').lower() != 'norfair':
        print('[REPLAY] NOTE: config2.yml selects a non-Norfair tracker; this replay always uses Norfair.')

    p = {}
    for key, default in DEFAULT_KEYS.items():
        src = stored_config if key in ('score', 'min_delay', 'rois') else config2
        p[key] = src.get(key, default)
    p['tip_offset'] = min(p['tip_offset'], 1.0)

    overrides = {}
    if override_path:
        with open(override_path, encoding='utf-8') as f:
            overrides = yaml.safe_load(f) or {}
        unknown = set(overrides) - set(DEFAULT_KEYS)
        if unknown:
            sys.exit(f'Unknown override keys: {sorted(unknown)}. Allowed: {sorted(DEFAULT_KEYS)}')
        p.update(overrides)

    if p['distance_threshold'] is None:
        p['distance_threshold'] = p['max_distance_between_points']
    if p['hit_counter_max'] is None:
        p['hit_counter_max'] = p['max_age']
    if p['past_detections_length'] is None:
        p['past_detections_length'] = p['max_age']
    return p, overrides


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    return z['pts'], z['offsets'], z['dets'], json.loads(str(z['meta']))


def sample_frames(pts, mode_fps, frame_log, sidecar):
    """Return a list of (frame_index, clock_seconds) to process."""
    if frame_log:
        if not sidecar:
            sys.exit('--frame-log needs --sidecar (the clip .json from record_clip.ps1) for the start time.')
        with open(sidecar, encoding='utf-8') as f:
            start_epoch = float(json.load(f)['start_epoch'])
        wall = []
        with open(frame_log, encoding='utf-8') as f:
            for r in csv.DictReader(line for line in f if not line.startswith('#')):
                wall.append(float(r['wall_ts']))
        rel = np.asarray(wall) - start_epoch
        rel = rel[(rel >= pts[0]) & (rel <= pts[-1])]
        if len(rel) == 0:
            sys.exit('Frame log has no rows inside this clip\'s time window. Wrong log file or clock offset?')
        # The live loop always reads the newest frame available, i.e. the last frame at or before that time.
        idx = np.clip(np.searchsorted(pts, rel, side='right') - 1, 0, len(pts) - 1)
        return list(zip(idx.tolist(), rel.tolist())), f'frame_log ({len(rel)} rows)'

    if mode_fps == 'all':
        return [(i, float(t)) for i, t in enumerate(pts)], 'all'

    fps = float(mode_fps)
    out, next_t = [], pts[0]
    for i, t in enumerate(pts):
        if t + 1e-6 >= next_t:
            out.append((i, float(t)))
            next_t += 1.0 / fps
            if t > next_t:  # gap in the video: resync instead of bursting
                next_t = t + 1.0 / fps
    return out, f'fps={fps:g}'


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--run-name', required=True, help='e.g. 000_baseline; one change per run')
    ap.add_argument('--override', help='YAML with keys to change (see DEFAULT_KEYS)')
    ap.add_argument('--fps', default='all', help="'all' or a number, e.g. 8")
    ap.add_argument('--frame-log', help='replay at the live loop frame times (frame_log_*.csv)')
    ap.add_argument('--sidecar', help='clip .json written by record_clip.ps1')
    ap.add_argument('--out-root', default=str(REPO_ROOT / 'eval' / 'runs'))
    args = ap.parse_args()

    try:
        nf_version = metadata.version('norfair')
    except metadata.PackageNotFoundError:
        nf_version = None
    if nf_version != PROD_NORFAIR_VERSION:
        print(f'[REPLAY] WARNING: norfair {nf_version} installed, production uses {PROD_NORFAIR_VERSION}. '
              'Results may not match production.')

    p, overrides = load_params(args.override)
    roi_polygons = [(r.get('name', f'lane{i + 1}'), Polygon(np.array(r['points'])))
                    for i, r in enumerate(p['rois']) if len(r.get('points', [])) >= 3]
    roi_names = [n for n, _ in roi_polygons]

    pts, offsets, dets, cache_meta = load_cache(args.cache)
    frames, sampling = sample_frames(pts, args.fps, args.frame_log, args.sidecar)

    tracker = Tracker(
        distance_function=iou,
        distance_threshold=p['distance_threshold'],
        hit_counter_max=p['hit_counter_max'],
        past_detections_length=p['past_detections_length'],
        initialization_delay=p['initialization_delay'],
    )
    post = CheckoutPostProcessor(roi_names, max_age=p['max_age'],
                                 min_delay=p['min_delay'], queue_min_dwell_sec=p['queue_min_dwell_sec'])

    clip = Path(args.cache).name.replace('.dets.npz', '')
    out_dir = Path(args.out_root) / args.run_name / clip
    out_dir.mkdir(parents=True, exist_ok=True)

    t_run = time.time()
    track_fields = ['step', 'frame_idx', 't', 'tid', 'x1', 'y1', 'x2', 'y2', 'matched', 'det_roi_idx',
                    'n_hits', 'hit_sec', 'track_dur', 'state', 'n_dets']
    last_t = frames[-1][1] if frames else 0.0
    with open(out_dir / 'tracks.csv', 'w', newline='', encoding='utf-8') as ft:
        tw = csv.DictWriter(ft, fieldnames=track_fields)
        tw.writeheader()
        for step, (fi, now) in enumerate(frames):
            frame_dets = dets[offsets[fi]:offsets[fi + 1]]
            detections = []
            # Same filtering as main.py: head class, score floor, tip point inside a lane polygon.
            for classid, score, x1, y1, x2, y2, headdir in frame_dets:
                if int(classid) != 0 or score < p['score']:
                    continue
                tip = Point(x1 + (x2 - x1) * p['tip_offset'], y2 - (y2 - y1) / 25)
                roi_idx = next((k for k, (_, zone) in enumerate(roi_polygons) if zone.contains(tip)), None)
                if roi_idx is None:
                    continue
                detections.append(Detection(
                    points=np.array([x1, y1, x2, y2], dtype=np.float64).reshape(2, 2),
                    data={'class_id': 0, 'confidence': float(score), 'roi_idx': int(roi_idx),
                          'head_direction': int(headdir)},
                ))
            tracked = tracker.update(detections=detections)
            for row in post.step(now, tracked):
                row.update(step=step, frame_idx=fi, t=round(now, 4), n_dets=len(detections))
                row['track_dur'] = round(row['track_dur'], 4)
                row['hit_sec'] = round(row['hit_sec'], 4)
                tw.writerow(row)
    post.finish(last_t)

    event_fields = ['tid', 'lane_idx', 'lane', 'entry_t', 'death_t', 'dwell', 'n_hits', 'hit_sec',
                    'active_heads_in_lane', 'outcome', 'written_entrance', 'written_service']
    with open(out_dir / 'events.csv', 'w', newline='', encoding='utf-8') as fe:
        ew = csv.DictWriter(fe, fieldnames=event_fields)
        ew.writeheader()
        for e in post.events:
            e = dict(e, entry_t=round(e['entry_t'], 4), death_t=round(e['death_t'], 4), dwell=round(e['dwell'], 4),
                     hit_sec=round(e['hit_sec'], 4))
            ew.writerow(e)

    outcomes = {}
    for e in post.events:
        outcomes[e['outcome']] = outcomes.get(e['outcome'], 0) + 1
    run = {
        'run_name': args.run_name,
        'clip': clip,
        'cache': str(Path(args.cache).name),
        'cache_meta': cache_meta,
        'sampling': sampling,
        'frames_processed': len(frames),
        'params_effective': p,
        'overrides': overrides,
        'norfair': nf_version,
        'numpy': np.__version__,
        'git_commit': git_commit(),
        'replay_sec': round(time.time() - t_run, 2),
        'event_outcomes': outcomes,
    }
    with open(out_dir / 'run.json', 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=2, default=str)

    print(f'[REPLAY] {clip} | {sampling} | {len(frames)} frames | outcomes={outcomes} | -> {out_dir}')


if __name__ == '__main__':
    main()
