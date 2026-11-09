"""Compare hand-marked arrivals/departures (mark_events.py) with what the tracking produced.

    # live system: rows exported on the production machine with `db_check.py export`
    python eval/compare.py --labels eval/data/labels/<clip>.labels.csv \
        --sidecar <clip>.json --db-export <clip>.db.csv

    # offline replay (track_replay.py output), same labels
    python eval/compare.py --labels eval/data/labels/<clip>.labels.csv --replay-events eval/runs/<run>/<clip>/events.csv

For every marked person (lane, arrive, leave) it finds the tracking rows in the same lane that overlap
them in time and reports:
  dwell error   main row (the one overlapping the person most) minus the real dwell, in seconds
  stitched      first row start to last row end, minus the real dwell (what joining fragments would give)
  rows/person   how many rows the person was split into
  missed        marked people with no row;  false rows: rows that overlap nobody

Live rows only have a lane when they reached service_events (dwell >= 3 s); shorter rows are counted
separately. People who stayed less than --min-dwell are passers-by for the pipeline and are reported
apart. People noted 'staff' are left out unless --include-staff.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_labels(path, include_staff):
    people, open_count = [], 0
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if not r['leave_s']:
                open_count += 1
                continue
            if not include_staff and 'staff' in r.get('note', '').lower():
                continue
            people.append({'person': int(r['person']), 'lane': int(r['lane']), 'start': float(r['arrive_s']),
                           'end': float(r['leave_s']), 'note': r.get('note', '')})
    meta_path = Path(path).with_name(Path(path).name.replace('.labels.csv', '.labels.meta.json'))
    window = None
    if meta_path.exists():
        with open(meta_path, encoding='utf-8') as f:
            m = json.load(f)
        window = (float(m['window_start_s']), float(m['window_end_s']))
    return people, open_count, window


def load_db_rows(path, start_epoch):
    """Rows in video seconds (before the clock offset is applied)."""
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            s = float(r['entry_epoch']) - start_epoch
            rows.append({'id': int(r['track_id']), 'start': s, 'end': s + float(r['dwell_s']),
                         'lane': int(r['lane_idx']) + 1 if r['lane_idx'] != '' else None})
    return rows


def load_replay_rows(path, all_tracks):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['outcome'] == 'censored_clip_end':
                continue
            if not all_tracks and r['outcome'] != 'inserted':
                continue
            lane_idx = int(r['lane_idx'])
            rows.append({'id': int(r['tid']), 'start': float(r['entry_t']), 'end': float(r['death_t']),
                         'lane': lane_idx + 1 if lane_idx >= 0 else None})
    return rows


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def temporal_iou(a0, a1, b0, b1):
    ov = overlap(a0, a1, b0, b1)
    union = max(a1, b1) - min(a0, b0)
    return ov / union if union > 0 else 0.0


def total_overlap(people, rows, offset):
    return sum(overlap(p['start'], p['end'], r['start'] + offset, r['end'] + offset)
               for p in people for r in rows if r['lane'] == p['lane'])


def estimate_offset(people, rows, span=5.0, step=0.25):
    """Clock offset between the clip's start time and the DB times that best lines the rows up with the labels."""
    candidates = np.arange(-span, span + step / 2, step)
    scores = [total_overlap(people, rows, o) for o in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(people, rows, window, min_dwell, lanes):
    w0, w1 = window
    # Rows that belong to the labelled window: started inside it (people already present at the
    # window start were not labelled, so their rows are left out). Only the marked lanes are scored.
    cand = [r for r in rows if w0 - 2.0 <= r['start'] <= w1]
    lane_rows = [r for r in cand if r['lane'] in lanes]
    short_rows = [r for r in cand if r['lane'] is None]

    # Each row goes to the marked person (same lane) whose stay it matches best (temporal IoU), so a
    # fragment lying inside two overlapping stays goes to the one it fits, not just the longer one.
    # With several people in one lane this can still be ambiguous; check <clip>.people.csv.
    assigned = {p['person']: [] for p in people}
    false_rows = []
    for r in lane_rows:
        best, best_ov = None, 0.0
        for p in people:
            if p['lane'] != r['lane']:
                continue
            ov = temporal_iou(p['start'], p['end'], r['start'], r['end'])
            if ov > best_ov:
                best, best_ov = p, ov
        if best is None:
            false_rows.append(r)
        else:
            assigned[best['person']].append((r, best_ov))

    per_person = []
    for p in people:
        real = p['end'] - p['start']
        got = assigned[p['person']]
        rec = {'person': p['person'], 'lane': p['lane'], 'arrive_s': p['start'], 'leave_s': p['end'],
               'real_dwell': round(real, 2), 'n_rows': len(got), 'note': p['note'],
               'passer_by': int(real < min_dwell)}
        if got:
            main = max(got, key=lambda x: x[1])[0]
            stitched = max(r['end'] for r, _ in got) - min(r['start'] for r, _ in got)
            rec.update(main_row=main['id'], main_dwell=round(main['end'] - main['start'], 2),
                       main_error=round(main['end'] - main['start'] - real, 2),
                       stitched_dwell=round(stitched, 2), stitched_error=round(stitched - real, 2),
                       start_error=round(main['start'] - p['start'], 2))
        per_person.append(rec)

    scored = [r for r in per_person if not r['passer_by']]
    found = [r for r in scored if r['n_rows'] > 0]

    def stats(vals):
        if not vals:
            return {}
        a = np.abs(vals)
        return {'mae': round(float(a.mean()), 2), 'median_abs': round(float(np.median(a)), 2),
                'p90_abs': round(float(np.percentile(a, 90)), 2), 'bias': round(float(np.mean(vals)), 2)}

    summary = {
        'people_marked': len(people),
        'people_scored': len(scored),
        'passers_by_excluded': len(per_person) - len(scored),
        'missed': len(scored) - len(found),
        'dwell_error_main_row_s': stats([r['main_error'] for r in found]),
        'dwell_error_stitched_s': stats([r['stitched_error'] for r in found]),
        'rows_per_person_mean': round(float(np.mean([r['n_rows'] for r in found])), 2) if found else None,
        'people_split_into_2plus_rows': sum(1 for r in found if r['n_rows'] >= 2),
        'false_rows_with_lane': len(false_rows),
        'lane_rows_per_marked_person': round(len(lane_rows) / len(people), 2) if people else None,
        # Short rows have no lane in the DB, so they can't be split per lane: this counts all lanes.
        'short_rows_without_lane_all_lanes': len(short_rows),
    }
    return summary, per_person


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--db-export', help='<clip>.db.csv from db_check.py export (needs --sidecar)')
    src.add_argument('--replay-events', help='events.csv from track_replay.py')
    ap.add_argument('--sidecar', help='clip .json from record_clip.ps1 (for --db-export)')
    ap.add_argument('--offset', type=float, help='clock offset in seconds; default: estimated for --db-export, 0 for replay')
    ap.add_argument('--all-tracks', action='store_true', help='replay: include tracks the DB would not get')
    ap.add_argument('--min-dwell', type=float, default=3.0, help='real dwell below this = passer-by (not scored)')
    ap.add_argument('--include-staff', action='store_true')
    ap.add_argument('--window', nargs=2, type=float, help='override the labelling window (seconds)')
    ap.add_argument('--lanes', nargs='+', type=int,
                    help='lanes that were marked (1-4, as in mark_events.py); default: lanes found in the labels')
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'eval' / 'runs' / 'compare'))
    args = ap.parse_args()

    people, open_count, window = load_labels(args.labels, args.include_staff)
    if args.window:
        window = tuple(args.window)
    if window is None:
        window = (min(p['start'] for p in people), max(p['start'] for p in people))
    if open_count:
        print(f'NOTE: {open_count} marked people have no leave time and are skipped.')

    if args.db_export:
        if not args.sidecar:
            ap.error('--db-export needs --sidecar')
        with open(args.sidecar, encoding='utf-8-sig') as f:
            start_epoch = float(json.load(f)['start_epoch'])
        rows = load_db_rows(args.db_export, start_epoch)
        source = 'live_db'
        offset = args.offset if args.offset is not None else estimate_offset(people, rows)
    else:
        rows = load_replay_rows(args.replay_events, args.all_tracks)
        source = 'replay' + ('_all' if args.all_tracks else '')
        offset = args.offset if args.offset is not None else 0.0
    for r in rows:
        r['start'] += offset
        r['end'] += offset

    lanes = sorted(set(args.lanes) if args.lanes else {p['lane'] for p in people})
    people = [p for p in people if p['lane'] in lanes]
    summary, per_person = evaluate(people, rows, window, args.min_dwell, set(lanes))
    clip = Path(args.labels).name.replace('.labels.csv', '')
    summary = {'clip': clip, 'source': source, 'source_file': str(args.db_export or args.replay_events),
               'window_s': list(window), 'lanes_scored': lanes, 'clock_offset_s': offset,
               'min_dwell_s': args.min_dwell, **summary}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{clip}__{source}'
    with open(out_dir / f'{stem}.summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    fields = ['person', 'lane', 'arrive_s', 'leave_s', 'real_dwell', 'n_rows', 'main_row', 'main_dwell',
              'main_error', 'stitched_dwell', 'stitched_error', 'start_error', 'passer_by', 'note']
    with open(out_dir / f'{stem}.people.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_person:
            w.writerow(r)

    d, s = summary['dwell_error_main_row_s'], summary['dwell_error_stitched_s']
    print(f"{clip} vs {source}  (lanes {lanes}, window {window[0]:g}-{window[1]:g}s, clock offset {offset:+.2f}s)")
    print(f"  people scored            : {summary['people_scored']}  (+{summary['passers_by_excluded']} passers-by not scored)")
    print(f"  missed (no row at all)   : {summary['missed']}")
    if d:
        print(f"  dwell error, main row    : MAE {d['mae']}s  median {d['median_abs']}s  P90 {d['p90_abs']}s  bias {d['bias']:+}s")
        print(f"  dwell error, stitched    : MAE {s['mae']}s  median {s['median_abs']}s  P90 {s['p90_abs']}s  bias {s['bias']:+}s")
    print(f"  rows per person          : {summary['rows_per_person_mean']}  ({summary['people_split_into_2plus_rows']} people split into 2+ rows)")
    print(f"  false rows (with lane)   : {summary['false_rows_with_lane']}")
    print(f"  rows / marked person     : {summary['lane_rows_per_marked_person']}  (rows >= 3 s in these lanes)")
    print(f"  short rows (<3s, no lane): {summary['short_rows_without_lane_all_lanes']}  (all lanes together)")
    print(f"  -> {out_dir / (stem + '.summary.json')}")


if __name__ == '__main__':
    main()
