"""
correlation_check.py

Checks whether customer age and gender (from entrance_events) correlate
with checkout dwell time (from service_events).

Run from the project root:
    python correlation_check.py

The script prints correlation coefficients and saves a CSV so you can
share the results with Arnaud even without a chart.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)

BROWSING_GAP_MIN = 25   # minutes between door entry and checkout
WINDOW_MIN       = 10   # ±10 min around the browsing gap

LOOKBACK_DAYS    = 30


def connect():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Cannot connect to database: {e}")
        sys.exit(1)


def load_joined(conn) -> pd.DataFrame:
    """
    For each service event (checkout), find all entrance_events that happened
    BROWSING_GAP_MIN ± WINDOW_MIN minutes earlier and compute:
      - avg_age      : mean age of customers likely heading to that checkout
      - female_ratio : fraction of female customers in that window
    """
    low  = BROWSING_GAP_MIN + WINDOW_MIN   # e.g. 35 min before checkout
    high = BROWSING_GAP_MIN - WINDOW_MIN   # e.g. 15 min before checkout

    query = f"""
        SELECT
            se.timestamp                                          AS checkout_time,
            se.total_dwell_sec / 60.0                            AS dwell_min,
            se.lane_id,
            AVG(ee.age_estimate)                                 AS avg_age,
            COUNT(*) FILTER (WHERE ee.gender = 'female')::float
                / NULLIF(COUNT(*), 0)                            AS female_ratio,
            COUNT(ee.track_id)                                   AS n_entries_in_window
        FROM service_events se
        LEFT JOIN entrance_events ee
            ON ee.timestamp BETWEEN se.timestamp - INTERVAL '{low} minutes'
                                AND se.timestamp - INTERVAL '{high} minutes'
           AND ee.camera_id NOT LIKE 'SIM_%%'
           AND ee.age_estimate IS NOT NULL
        WHERE se.total_dwell_sec > 0
          AND se.camera_id NOT LIKE 'SIM_%%'
          AND se.timestamp >= NOW() - INTERVAL '{LOOKBACK_DAYS} days'
        GROUP BY se.timestamp, se.total_dwell_sec, se.lane_id
        ORDER BY se.timestamp
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        sys.exit(1)

    return df


def run_correlation(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  Demographic correlation with checkout dwell time")
    print(f"{'='*55}")
    print(f"  Total service events loaded : {len(df)}")

    # Only rows where we actually have demographic data
    df_demo = df.dropna(subset=["avg_age", "female_ratio", "dwell_min"])
    print(f"  Rows with demographic match : {len(df_demo)}")

    if len(df_demo) < 10:
        print("\n  [WARNING] Not enough matched rows to draw conclusions.")
        print("  This usually means entrance_events has no age/gender data yet.")
        print("  Check: SELECT COUNT(*) FROM entrance_events WHERE age_estimate IS NOT NULL;")
        return df_demo

    print(f"\n  Correlation with dwell_min (Pearson r):")
    for col, label in [("avg_age", "avg_age     "), ("female_ratio", "female_ratio")]:
        r = df_demo["dwell_min"].corr(df_demo[col])
        direction = "+" if r > 0.05 else ("-" if r < -0.05 else "~0 (no signal)")
        print(f"    {label} : r = {r:+.3f}  ({direction})")

    print(f"\n  Dwell time summary:")
    print(f"    Mean   : {df_demo['dwell_min'].mean():.2f} min")
    print(f"    Median : {df_demo['dwell_min'].median():.2f} min")
    print(f"    Std    : {df_demo['dwell_min'].std():.2f} min")
    print(f"    Min    : {df_demo['dwell_min'].min():.2f} min")
    print(f"    Max    : {df_demo['dwell_min'].max():.2f} min")

    print(f"\n  Age summary (entrance window):")
    print(f"    Mean age   : {df_demo['avg_age'].mean():.1f} years")
    print(f"    Female ratio (mean): {df_demo['female_ratio'].mean():.1%}")

    # Bucket by age group and show mean dwell per group
    df_demo = df_demo.copy()
    df_demo["age_group"] = pd.cut(
        df_demo["avg_age"],
        bins=[0, 30, 50, 100],
        labels=["18-30", "30-50", "50+"],
    )
    print(f"\n  Mean dwell by avg age group:")
    age_summary = df_demo.groupby("age_group", observed=True)["dwell_min"].agg(["mean", "count"])
    for group, row in age_summary.iterrows():
        print(f"    {group} : {row['mean']:.2f} min  (n={int(row['count'])})")

    # Bucket by gender ratio
    df_demo["gender_bucket"] = pd.cut(
        df_demo["female_ratio"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["mostly male", "mixed", "mostly female"],
    )
    print(f"\n  Mean dwell by gender mix:")
    gender_summary = df_demo.groupby("gender_bucket", observed=True)["dwell_min"].agg(["mean", "count"])
    for group, row in gender_summary.iterrows():
        print(f"    {group} : {row['mean']:.2f} min  (n={int(row['count'])})")

    return df_demo


def save_results(df: pd.DataFrame):
    out = "correlation_results.csv"
    df.to_csv(out, index=False)
    print(f"\n  Results saved to: {out}")
    print(f"  Share this file with Arnaud for review.\n")


if __name__ == "__main__":
    print(f"Connecting to {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']} ...")
    conn = connect()
    print("Connected. Loading data...")

    df = load_joined(conn)
    conn.close()

    df_result = run_correlation(df)
    save_results(df_result)