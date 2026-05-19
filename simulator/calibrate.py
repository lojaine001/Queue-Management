from __future__ import annotations

"""
calibrate.py
────────────
Reads REAL data from the database (entrance_events, service_events,
queue_state_snapshots) and produces a calibrated scenario config that
the SimulatorEngine can use directly instead of hardcoded guesses.

Usage:
    python -m simulator.calibrate                  # prints report
    python -m simulator.calibrate --output json    # prints JSON config
    python -m simulator.calibrate --output py      # prints Python dict
    python -m simulator.calibrate --days 30        # look back N days
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

load_dotenv(find_dotenv(usecwd=True))

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME",     "iqms"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "0000"),
}

REAL_FILTER = "AND camera_id NOT LIKE 'SIM_%%'"


def connect():
    return psycopg2.connect(**DB_CONFIG)


def table_exists(conn, table_name: str) -> bool:
    q = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        )
    """
    with conn.cursor() as cur:
        cur.execute(q, (table_name,))
        return bool(cur.fetchone()[0])


def get_columns(conn, table_name: str) -> set[str]:
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(q, (table_name,))
        return {row[0] for row in cur.fetchall()}


def load_entrance_events(conn, days: int) -> pd.DataFrame:
    if not table_exists(conn, "entrance_events"):
        return pd.DataFrame(columns=["timestamp", "gender", "age_estimate", "has_caddy", "has_bag", "is_group"])

    since = datetime.now() - timedelta(days=days)
    cols = get_columns(conn, "entrance_events")
    has_caddy_expr = "has_caddy" if "has_caddy" in cols else "FALSE AS has_caddy"
    has_bag_expr = "has_bag" if "has_bag" in cols else "FALSE AS has_bag"
    is_group_expr = "is_group" if "is_group" in cols else "FALSE AS is_group"
    q = f"""
        SELECT
            timestamp,
            gender,
            age_estimate,
            {has_caddy_expr},
            {has_bag_expr},
            {is_group_expr}
        FROM entrance_events
        WHERE timestamp >= %s {REAL_FILTER}
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn, params=(since,))
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df


def load_service_events(conn, days: int) -> pd.DataFrame:
    if not table_exists(conn, "service_events"):
        return pd.DataFrame(columns=["timestamp", "total_dwell_sec", "lane_id"])

    since = datetime.now() - timedelta(days=days)
    cols = get_columns(conn, "service_events")
    lane_id_expr = "lane_id" if "lane_id" in cols else "NULL AS lane_id"
    q = f"""
        SELECT
            timestamp,
            total_dwell_sec,
            {lane_id_expr}
        FROM service_events
        WHERE timestamp >= %s {REAL_FILTER}
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn, params=(since,))
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df


def load_snapshots(conn, days: int) -> pd.DataFrame:
    if not table_exists(conn, "queue_state_snapshots"):
        return pd.DataFrame(columns=["timestamp", "queue_count", "avg_dwell_sec", "max_dwell_sec", "active_lanes"])

    since = datetime.now() - timedelta(days=days)
    cols = get_columns(conn, "queue_state_snapshots")
    max_dwell_expr = "max_dwell_sec" if "max_dwell_sec" in cols else "0 AS max_dwell_sec"
    q = f"""
        SELECT
            timestamp,
            queue_count,
            avg_dwell_sec,
            {max_dwell_expr},
            active_lanes
        FROM queue_state_snapshots
        WHERE timestamp >= %s {REAL_FILTER}
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn, params=(since,))
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df


def calibrate(days: int = 30) -> dict:
    conn = connect()
    print(f"[Calibrate] Connected to DB. Analysing last {days} days of REAL data...", file=sys.stderr)

    ee = load_entrance_events(conn, days)
    se = load_service_events(conn, days)
    ss = load_snapshots(conn, days)
    conn.close()

    result = {}

    # ── Arrival rates ─────────────────────────────────────────────────────────
    if len(ee) > 1:
        ee["hour"] = ee["timestamp"].dt.hour
        ee["dow"]  = ee["timestamp"].dt.dayofweek  # 0=Mon … 6=Sun

        # Average gap between arrivals per hour (seconds)
        ee_sorted = ee.sort_values("timestamp")
        ee_sorted["gap"] = ee_sorted["timestamp"].diff().dt.total_seconds()
        hourly_gaps = (
            ee_sorted.groupby("hour")["gap"]
            .median()
            .round(1)
            .dropna()
            .to_dict()
        )
        # Fill missing hours with a large gap (store closed)
        hourly_arrival_gaps = {h: hourly_gaps.get(h, 300.0) for h in range(24)}
        result["hourly_arrival_gaps"] = hourly_arrival_gaps
        result["avg_arrival_gap_seconds"] = float(round(ee_sorted["gap"].median(), 1))

        # Gender distribution
        gender_counts = ee["gender"].value_counts(normalize=True)
        result["prob_female"] = float(round(gender_counts.get("female", 0.5), 3))

        # Age distribution
        ages = ee["age_estimate"].dropna()
        result["age_mean"] = float(round(ages.mean(), 1))
        result["age_std"]  = float(round(ages.std(), 1))

        # Accessory probabilities
        result["prob_caddy"] = float(round(ee["has_caddy"].mean(), 3)) if "has_caddy" in ee.columns else 0.2
        result["prob_bag"]   = float(round(ee["has_bag"].mean(),   3)) if "has_bag"   in ee.columns else 0.3
        result["prob_group"] = float(round(ee["is_group"].mean(),  3)) if "is_group"  in ee.columns else 0.1

        print(f"[Calibrate]   Entrance events : {len(ee)}", file=sys.stderr)
    else:
        print("[Calibrate]   ⚠ Not enough entrance events — using defaults", file=sys.stderr)
        result.update({
            "avg_arrival_gap_seconds": 60.0,
            "prob_female": 0.5, "age_mean": 45.0, "age_std": 15.0,
            "prob_caddy": 0.2, "prob_bag": 0.3, "prob_group": 0.1,
        })

    # ── Service / dwell time ──────────────────────────────────────────────────
    valid_se = se[se["total_dwell_sec"] > 0].dropna(subset=["total_dwell_sec"])
    if len(valid_se) > 0:
        result["base_service_seconds"] = float(round(valid_se["total_dwell_sec"].mean(), 1))
        result["service_std_seconds"]  = float(round(valid_se["total_dwell_sec"].std(),  1))
        result["service_min_seconds"]  = float(round(valid_se["total_dwell_sec"].quantile(0.05), 1))
        result["service_max_seconds"]  = float(round(valid_se["total_dwell_sec"].quantile(0.95), 1))
        print(f"[Calibrate]   Service events  : {len(valid_se)}", file=sys.stderr)
    else:
        print("[Calibrate]   ⚠ No service events found — using defaults", file=sys.stderr)
        result.update({
            "base_service_seconds": 90.0, "service_std_seconds": 30.0,
            "service_min_seconds": 30.0,  "service_max_seconds": 180.0,
        })

    # ── Active lanes ──────────────────────────────────────────────────────────
    if len(ss) > 0:
        result["num_lanes"]     = int(ss["active_lanes"].mode()[0])
        result["peak_queue"]    = float(round(ss["queue_count"].quantile(0.95), 1))
        result["avg_queue"]     = float(round(ss["queue_count"].mean(), 1))
        print(f"[Calibrate]   Snapshots       : {len(ss)}", file=sys.stderr)
    else:
        print("[Calibrate]   ⚠ No snapshots found — using defaults", file=sys.stderr)
        result.update({"num_lanes": 2, "peak_queue": 10.0, "avg_queue": 3.0})

    # Metadata
    result["calibrated_from_days"] = days
    result["calibrated_at"]        = datetime.now().isoformat()

    return result


def print_report(cfg: dict):
    print("\n" + "=" * 60)
    print("  CALIBRATION REPORT")
    print("=" * 60)
    print(f"  Based on last {cfg['calibrated_from_days']} days of REAL data")
    print(f"  Generated at: {cfg['calibrated_at']}")
    print()
    print("  ARRIVALS")
    print(f"    Avg gap between arrivals : {cfg.get('avg_arrival_gap_seconds')} sec")
    print(f"    % Female customers       : {cfg.get('prob_female', 0) * 100:.1f}%")
    print(f"    Age (mean ± std)         : {cfg.get('age_mean')} ± {cfg.get('age_std')} yrs")
    print(f"    P(has caddy)             : {cfg.get('prob_caddy', 0) * 100:.1f}%")
    print(f"    P(has bag)               : {cfg.get('prob_bag', 0) * 100:.1f}%")
    print(f"    P(is group)              : {cfg.get('prob_group', 0) * 100:.1f}%")
    print()
    print("  CAISSE / SERVICE")
    print(f"    Mean dwell at caisse     : {cfg.get('base_service_seconds')} sec")
    print(f"    Std dev                  : {cfg.get('service_std_seconds')} sec")
    print(f"    5th–95th percentile      : {cfg.get('service_min_seconds')}–{cfg.get('service_max_seconds')} sec")
    print()
    print("  LANES & QUEUE")
    print(f"    Typical active lanes     : {cfg.get('num_lanes')}")
    print(f"    Avg queue depth          : {cfg.get('avg_queue')}")
    print(f"    95th-pct queue depth     : {cfg.get('peak_queue')}")
    print("=" * 60)

    if "hourly_arrival_gaps" in cfg:
        print("\n  HOURLY ARRIVAL GAPS (seconds between customers)")
        print(f"  {'Hour':<6} {'Gap (sec)':>10}")
        for h, gap in sorted(cfg["hourly_arrival_gaps"].items()):
            bar = "█" * max(1, int(60 / max(gap, 1)))
            print(f"  {h:02d}:00  {gap:>10.1f}  {bar}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate simulator from real DB data")
    parser.add_argument("--days",   type=int, default=30,      help="Days of history to analyse")
    parser.add_argument("--output", type=str, default="report", choices=["report", "json", "py"])
    args = parser.parse_args()

    cfg = calibrate(days=args.days)

    if args.output == "json":
        print(json.dumps(cfg, indent=2, default=str))
    elif args.output == "py":
        print("CALIBRATED = " + json.dumps(cfg, indent=4, default=str).replace("null", "None"))
    else:
        print_report(cfg)
