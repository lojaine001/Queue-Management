"""Smoke test for track_replay.py on a synthetic cache with known people (no footage needed).

    python eval/tests/test_replay_synthetic.py

Scenario at 8 fps over 30 s, lanes defined in the test (independent of config.yml):
  A  lane2, t=1..21 s                     -> inserted, dwell ~20 s
  D  lane2, t=3..8 s, while A is there    -> suppressed (A has the longer dwell)
  B  lane3, t=5..12 s, gap 1.5 s, t=13.5..18 s -> two separate inserted tracks (fragmentation)
  C  lane2, t=25..25.3 s (3 frames)       -> skipped_unconfirmed
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

EVAL_DIR = Path(__file__).resolve().parent.parent
FPS = 8.0
DURATION = 30.0

ROIS = [
    {'name': 'lane1', 'points': [[2, 2], [187, 5], [86, 261], [1, 188]]},
    {'name': 'lane2', 'points': [[120, 440], [230, 17], [309, 17], [400, 440]]},
    {'name': 'lane3', 'points': [[345, 4], [480, 446], [567, 101], [458, 7]]},
]
PEOPLE = {
    'A': ((240, 200, 280, 240), [(1.0, 21.0)]),
    'D': ((250, 300, 290, 340), [(3.0, 8.0)]),
    'B': ((450, 110, 490, 150), [(5.0, 12.0), (13.5, 18.0)]),
    'C': ((250, 380, 290, 420), [(25.0, 25.3)]),
}


def make_cache(path):
    rng = np.random.default_rng(0)
    n = int(DURATION * FPS)
    pts = np.arange(n) / FPS
    rows, counts = [], []
    for t in pts:
        c = 0
        for box, spans in PEOPLE.values():
            if any(s <= t < e for s, e in spans):
                jitter = rng.integers(-1, 2, size=4)
                rows.append([0, 0.8, *(np.array(box) + jitter), -1])
                c += 1
        counts.append(c)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    np.savez_compressed(path, pts=pts, offsets=offsets, dets=np.asarray(rows, dtype=np.float32).reshape(-1, 7),
                        infer_ms=np.zeros(n, dtype=np.float32), meta=json.dumps({'video': 'synthetic'}))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / 'synthetic.dets.npz'
        make_cache(cache)
        override = tmp / 'override.yml'
        # Production values from config.yml / config2.yml, pinned here so the test doesn't depend on them.
        override.write_text(yaml.safe_dump({
            'rois': ROIS, 'score': 0.15, 'min_delay': 0.45, 'max_distance_between_points': 6,
            'max_age': 4, 'tip_offset': 0.5, 'queue_min_dwell_sec': 3.0,
        }), encoding='utf-8')

        subprocess.run([sys.executable, str(EVAL_DIR / 'track_replay.py'), '--cache', str(cache),
                        '--run-name', 'test', '--override', str(override), '--out-root', str(tmp / 'runs')],
                       check=True)

        with open(tmp / 'runs' / 'test' / 'synthetic' / 'events.csv', encoding='utf-8') as f:
            events = list(csv.DictReader(f))

    for e in events:
        print(f"  tid={e['tid']:>3} {e['lane']:<7} entry={float(e['entry_t']):6.2f} "
              f"dwell={float(e['dwell']):6.2f} hits={e['n_hits']:>4} hit_sec={float(e['hit_sec']):6.2f} {e['outcome']}")

    by_outcome = {}
    for e in events:
        by_outcome.setdefault(e['outcome'], []).append(e)
    inserted = sorted(by_outcome.get('inserted', []), key=lambda e: float(e['entry_t']))

    assert len(inserted) == 3, f'expected A + B twice inserted, got {len(inserted)}'
    a = [e for e in inserted if e['lane'] == 'lane2']
    assert len(a) == 1 and 20.0 <= float(a[0]['dwell']) <= 21.0, a
    b = [e for e in inserted if e['lane'] == 'lane3']
    assert len(b) == 2, 'B should be split into two tracks by the 1.5 s gap'
    assert len(by_outcome.get('suppressed_sibling', [])) == 1, 'D should be suppressed by A'
    assert len(by_outcome.get('skipped_unconfirmed', [])) == 1, 'C should be skipped (too few hits)'
    print('OK: synthetic replay behaves as expected')


if __name__ == '__main__':
    main()
