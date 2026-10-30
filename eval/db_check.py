"""Read-only checks on the IQMS database for the tracking evaluation.

Uses the same connection settings as the pipeline (.env: DB_HOST, DB_PORT, DB_NAME,
DB_USER, DB_PASSWORD, STORE_TZ). The session is opened read-only, so it cannot change data.

    python eval/db_check.py hours                  # busiest hours, last 14 days
    python eval/db_check.py clip eval/data/raw/checkout_20261029_141413_sizetest.json
    python eval/db_check.py fragments              # rows starting right after the previous one ended
    python eval/db_check.py export eval/data/raw/<clip>.json   # live rows for a clip -> <clip>.db.csv

`export` writes only numbers (track id, times, dwell, lane), so the file can be copied next to the
videos and used by eval/compare.py on another machine.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
DAYS = ['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def connect():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "iqms"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "0000"),
    )
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET timezone = %s", (os.getenv("STORE_TZ", "Europe/Paris"),))
    return cur


def cmd_hours(cur, args):
    cur.execute("""
        SELECT extract(isodow FROM timestamp)::int, extract(hour FROM timestamp)::int,
               round(avg(queue_count)::numeric, 2), max(queue_count), count(*)
        FROM queue_state_snapshots
        WHERE camera_id = %s AND timestamp > now() - make_interval(days => %s)
        GROUP BY 1, 2
        ORDER BY 3 DESC
        LIMIT %s
    """, (args.camera, args.days, args.top))
    print(f"Busiest hours for {args.camera}, last {args.days} days (queue_state_snapshots):")
    print(f"{'day':<4} {'hour':>5} {'avg people':>11} {'max':>5} {'samples':>8}")
    for dow, hour, avg_q, max_q, n in cur.fetchall():
        print(f"{DAYS[dow]:<4} {hour:>2}:00 {avg_q:>11} {max_q:>5} {n:>8}")


def cmd_clip(cur, args):
    with open(args.sidecar, encoding='utf-8') as f:
        side = json.load(f)
    t0, t1 = float(side['start_epoch']), float(side['end_epoch'])
    cam = side.get('cam_id') or args.camera
    cur.execute("""
        SELECT round(avg(queue_count)::numeric, 2), max(queue_count), count(*)
        FROM queue_state_snapshots
        WHERE camera_id = %s AND timestamp BETWEEN to_timestamp(%s) AND to_timestamp(%s)
    """, (cam, t0, t1))
    avg_q, max_q, n_snap = cur.fetchone()
    cur.execute("""
        SELECT count(*), round(avg(total_dwell_sec)::numeric, 1)
        FROM service_events
        WHERE camera_id = %s AND timestamp BETWEEN to_timestamp(%s) AND to_timestamp(%s)
    """, (cam, t0, t1))
    n_serv, avg_dwell = cur.fetchone()
    cur.execute("""
        SELECT count(*)
        FROM entrance_events
        WHERE camera_id = %s AND timestamp BETWEEN to_timestamp(%s) AND to_timestamp(%s)
    """, (cam, t0, t1))
    n_entr = cur.fetchone()[0]
    print(f"Clip {side['file']} ({side['start_local']}, {(t1 - t0) / 60:.1f} min, camera {cam}):")
    print(f"  queue size during clip : avg {avg_q} people, max {max_q} ({n_snap} snapshots)")
    print(f"  rows written by live   : {n_entr} entrance_events, {n_serv} service_events (avg dwell {avg_dwell}s)")


def cmd_fragments(cur, args):
    cur.execute("""
        WITH s AS (
          SELECT lane_id, timestamp AS death_t,
                 timestamp - make_interval(secs => total_dwell_sec) AS entry_t, total_dwell_sec
          FROM service_events
          WHERE camera_id = %s AND timestamp > now() - make_interval(days => %s)
        ), g AS (
          SELECT lane_id, total_dwell_sec,
                 EXTRACT(EPOCH FROM entry_t - LAG(death_t) OVER (PARTITION BY lane_id ORDER BY entry_t)) AS gap_sec
          FROM s
        )
        SELECT lane_id, count(*),
               count(*) FILTER (WHERE gap_sec BETWEEN -1 AND 1.5),
               round(avg(total_dwell_sec)::numeric, 1),
               round(percentile_cont(0.5) WITHIN GROUP (ORDER BY total_dwell_sec)::numeric, 1)
        FROM g GROUP BY lane_id ORDER BY lane_id
    """, (args.camera, args.days))
    print(f"service_events for {args.camera}, last {args.days} days")
    print("('right after' = row starts within 1.5 s of the previous row's end in the same lane: possible split track)")
    print(f"{'lane':>4} {'rows':>7} {'right after':>12} {'share':>6} {'avg dwell':>10} {'median':>7}")
    for lane, n, quick, avg_d, med_d in cur.fetchall():
        share = f"{100 * quick / n:.0f}%" if n else '-'
        print(f"{str(lane):>4} {n:>7} {quick:>12} {share:>6} {avg_d:>10} {med_d:>7}")


def cmd_export(cur, args):
    with open(args.sidecar, encoding='utf-8-sig') as f:
        side = json.load(f)
    t0, t1 = float(side['start_epoch']), float(side['end_epoch'])
    cam = side.get('cam_id') or args.camera
    # Every checkout track the live loop wrote, with its lane when it also reached service_events
    # (only tracks >= queue_min_dwell_sec do). The margins catch tracks that started before the clip
    # or ended after it.
    cur.execute("""
        SELECT e.track_id,
               extract(epoch FROM e.timestamp)::float8      AS entry_epoch,
               e.dwell_seconds,
               e.active_head_tracks_in_lane,
               s.lane_id,
               extract(epoch FROM s.timestamp)::float8      AS service_epoch
        FROM entrance_events e
        LEFT JOIN LATERAL (
            SELECT lane_id, timestamp FROM service_events s
            WHERE s.camera_id = e.camera_id AND s.track_id = e.track_id
              AND s.timestamp BETWEEN e.timestamp + make_interval(secs => e.dwell_seconds) - interval '10 seconds'
                                  AND e.timestamp + make_interval(secs => e.dwell_seconds) + interval '10 seconds'
            ORDER BY s.timestamp LIMIT 1
        ) s ON TRUE
        WHERE e.camera_id = %s
          AND e.timestamp BETWEEN to_timestamp(%s) AND to_timestamp(%s)
        ORDER BY e.timestamp
    """, (cam, t0 - 900, t1 + 60))
    rows = cur.fetchall()
    out = Path(args.out) if args.out else Path(args.sidecar).with_suffix('.db.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['track_id', 'entry_epoch', 'dwell_s', 'active_heads_in_lane', 'lane_idx', 'service_epoch'])
        for track_id, entry, dwell, heads, lane, serv in rows:
            w.writerow([track_id, f'{entry:.3f}', f'{dwell:.3f}', heads,
                        '' if lane is None else lane, '' if serv is None else f'{serv:.3f}'])
    n_lane = sum(1 for r in rows if r[4] is not None)
    print(f'Wrote {len(rows)} rows ({n_lane} with a lane) for {side["file"]} -> {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', default='Bosch_Camera_exit')
    ap.add_argument('--days', type=int, default=14)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_hours = sub.add_parser('hours', help='busiest hours by average queue size')
    p_hours.add_argument('--top', type=int, default=12)
    p_clip = sub.add_parser('clip', help='how busy a recorded clip was, and what live wrote meanwhile')
    p_clip.add_argument('sidecar', help='clip .json written by record_clip.ps1')
    sub.add_parser('fragments', help='rows that start right after the previous one (split-track signal)')
    p_export = sub.add_parser('export', help='write the live rows for a clip to <clip>.db.csv (numbers only)')
    p_export.add_argument('sidecar', help='clip .json written by record_clip.ps1')
    p_export.add_argument('--out', help='output CSV (default: next to the .json)')
    args = ap.parse_args()

    try:
        cur = connect()
    except Exception as e:
        sys.exit(f'Could not connect to PostgreSQL: {e}')
    {'hours': cmd_hours, 'clip': cmd_clip, 'fragments': cmd_fragments,
     'export': cmd_export}[args.cmd](cur, args)


if __name__ == '__main__':
    main()
