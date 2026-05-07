"""
Prediction Accuracy Evaluation
Compares ensemble_yhat (predicted arrivals / 3-min bucket) against actual
entrance_events counts for the same buckets.

Outputs:
  accuracy_results.csv      — per-slot comparison
  accuracy_by_hour.csv      — MAE/RMSE breakdown by hour of day
  accuracy_by_leadtime.csv  — MAE/RMSE breakdown by prediction lead time
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))

CAMERA_ID  = os.getenv("CAM_ID",          "Bosch_Camera_Entrance")
BUCKET_MIN = int(os.getenv("BUCKET_MINUTES", 3))
EVAL_DAYS  = 14   # how many past days to evaluate

DB_CONFIG = dict(
    host     = os.getenv("DB_HOST",     "localhost"),
    port     = int(os.getenv("DB_PORT", 5432)),
    dbname   = os.getenv("DB_NAME",     "iqms"),
    user     = os.getenv("DB_USER",     "postgres"),
    password = os.getenv("DB_PASSWORD", "0000"),
)


def _conn():
    return psycopg2.connect(**DB_CONFIG)


# ── 1. Load predictions (latest per slot) ─────────────────────────────────────

print(f"[1/4] Loading past predictions (last {EVAL_DAYS} days)...")
with _conn() as conn:
    pred_df = pd.read_sql("""
        SELECT DISTINCT ON (prediction_for)
            predicted_at,
            prediction_for,
            ensemble_yhat  AS predicted_arrivals,
            est_wait_minutes AS predicted_wait_min,
            status
        FROM queue_predictions
        WHERE prediction_for >= NOW() - INTERVAL '14 days'
          AND prediction_for <  NOW()
        ORDER BY prediction_for, predicted_at DESC
    """, conn)

print(f"    {len(pred_df)} prediction slots loaded")

if pred_df.empty:
    print("ERROR: No past predictions found. Run ensemble_predict.py first.")
    sys.exit(1)

pred_df["prediction_for"] = pd.to_datetime(pred_df["prediction_for"], utc=True)
pred_df["predicted_at"]   = pd.to_datetime(pred_df["predicted_at"],   utc=True)
pred_df["lead_minutes"]   = (
    (pred_df["prediction_for"] - pred_df["predicted_at"])
    .dt.total_seconds() / 60
).round(1)


# ── 2. Load actual entrance events, bucketed to 3-min intervals ───────────────

print(f"[2/4] Loading actual entrance events...")
with _conn() as conn:
    actual_df = pd.read_sql(f"""
        SELECT
            time_bucket('{BUCKET_MIN} minutes', timestamp) AS bucket,
            COUNT(*) AS actual_arrivals
        FROM entrance_events
        WHERE (
            (timestamp >= '2026-04-13' AND timestamp < '2026-04-16')
            OR timestamp >= '2026-04-21'
        )
          AND dwell_seconds >= 10
          AND timestamp >= NOW() - INTERVAL '{EVAL_DAYS} days'
          AND timestamp <  NOW()
        GROUP BY bucket
        ORDER BY bucket
    """, conn)

print(f"    {len(actual_df)} actual 3-min buckets loaded")

if actual_df.empty:
    print("ERROR: No entrance event data found in the evaluation period.")
    sys.exit(1)

actual_df["bucket"] = pd.to_datetime(actual_df["bucket"], utc=True)


# ── 3. Merge predictions with actuals ─────────────────────────────────────────

print("[3/4] Joining predictions with actuals...")

merged = pd.merge(
    pred_df,
    actual_df,
    left_on="prediction_for",
    right_on="bucket",
    how="inner",
)

print(f"    {len(merged)} matched slots ({len(pred_df) - len(merged)} predictions had no matching actual data)")

if merged.empty:
    print("ERROR: No matching slots. Check that timestamp timezones align.")
    sys.exit(1)

# Error columns
merged["error"]     = merged["predicted_arrivals"] - merged["actual_arrivals"]
merged["abs_error"] = merged["error"].abs()
merged["sq_error"]  = merged["error"] ** 2

# Time helpers
merged["hour"]       = merged["prediction_for"].dt.hour
merged["day_of_week"]= merged["prediction_for"].dt.day_name()
merged["date"]       = merged["prediction_for"].dt.date


# ── 4. Compute metrics ────────────────────────────────────────────────────────

print("[4/4] Computing metrics...\n")

mae  = merged["abs_error"].mean()
rmse = np.sqrt(merged["sq_error"].mean())
bias = merged["error"].mean()

nonzero = merged[merged["actual_arrivals"] > 0]
mape    = (nonzero["abs_error"] / nonzero["actual_arrivals"]).mean() * 100

n_total     = len(merged)
n_days      = merged["date"].nunique()
date_min    = merged["date"].min()
date_max    = merged["date"].max()


# ── By hour of day ────────────────────────────────────────────────────────────

by_hour = (
    merged.groupby("hour")
    .agg(
        n           = ("abs_error", "count"),
        actual_avg  = ("actual_arrivals", "mean"),
        pred_avg    = ("predicted_arrivals", "mean"),
        MAE         = ("abs_error", "mean"),
        RMSE        = ("sq_error", lambda x: np.sqrt(x.mean())),
        bias        = ("error", "mean"),
    )
    .reset_index()
)
by_hour["actual_avg"] = by_hour["actual_avg"].round(2)
by_hour["pred_avg"]   = by_hour["pred_avg"].round(2)
by_hour["MAE"]        = by_hour["MAE"].round(3)
by_hour["RMSE"]       = by_hour["RMSE"].round(3)
by_hour["bias"]       = by_hour["bias"].round(3)


# ── By lead time bucket ───────────────────────────────────────────────────────

def lead_bin(m):
    if m <= 15:  return "0-15 min"
    if m <= 30:  return "15-30 min"
    if m <= 45:  return "30-45 min"
    return "45-60 min"

merged["lead_bin"] = merged["lead_minutes"].apply(lead_bin)

lead_order = ["0-15 min", "15-30 min", "30-45 min", "45-60 min"]
by_lead = (
    merged.groupby("lead_bin")
    .agg(
        n    = ("abs_error", "count"),
        MAE  = ("abs_error", "mean"),
        RMSE = ("sq_error", lambda x: np.sqrt(x.mean())),
        bias = ("error", "mean"),
    )
    .reindex(lead_order)
    .dropna()
    .reset_index()
)
by_lead["MAE"]  = by_lead["MAE"].round(3)
by_lead["RMSE"] = by_lead["RMSE"].round(3)
by_lead["bias"] = by_lead["bias"].round(3)


# ── By day of week ────────────────────────────────────────────────────────────

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_day = (
    merged.groupby("day_of_week")
    .agg(
        n           = ("abs_error", "count"),
        actual_avg  = ("actual_arrivals", "mean"),
        MAE         = ("abs_error", "mean"),
        RMSE        = ("sq_error", lambda x: np.sqrt(x.mean())),
    )
    .reindex(day_order)
    .dropna()
    .reset_index()
)
by_day["MAE"]  = by_day["MAE"].round(3)
by_day["RMSE"] = by_day["RMSE"].round(3)


# ── Print report ──────────────────────────────────────────────────────────────

W = 62
print("=" * W)
print("  IQMS — PREDICTION ACCURACY REPORT")
print(f"  Period : {date_min} → {date_max}  ({n_days} days)")
print(f"  Bucket : {BUCKET_MIN} min   |   Matched slots : {n_total}")
print("=" * W)
print()
print("  ARRIVAL PREDICTION  (ensemble_yhat vs actual entries/bucket)")
print(f"  {'MAE':<20} {mae:.3f} arrivals / {BUCKET_MIN}-min bucket")
print(f"  {'RMSE':<20} {rmse:.3f} arrivals / {BUCKET_MIN}-min bucket")
print(f"  {'MAPE':<20} {mape:.1f}%  (non-zero buckets only)")
print(f"  {'Bias':<20} {bias:+.3f}  ({'over' if bias > 0 else 'under'}-predicting on average)")
print()
print("  BY HOUR OF DAY")
print(f"  {'Hour':<8} {'n':>5} {'Actual':>8} {'Pred':>8} {'MAE':>8} {'RMSE':>8} {'Bias':>8}")
print("  " + "-" * 55)
for _, r in by_hour.iterrows():
    print(f"  {int(r['hour']):02d}:00  {int(r['n']):>5}  {r['actual_avg']:>8.2f}  {r['pred_avg']:>8.2f}"
          f"  {r['MAE']:>8.3f}  {r['RMSE']:>8.3f}  {r['bias']:>+8.3f}")
print()
print("  BY LEAD TIME")
print(f"  {'Lead':<12} {'n':>5} {'MAE':>8} {'RMSE':>8} {'Bias':>8}")
print("  " + "-" * 45)
for _, r in by_lead.iterrows():
    print(f"  {r['lead_bin']:<12} {int(r['n']):>5}  {r['MAE']:>8.3f}  {r['RMSE']:>8.3f}  {r['bias']:>+8.3f}")
print()
print("  BY DAY OF WEEK")
print(f"  {'Day':<12} {'n':>5} {'Actual avg':>10} {'MAE':>8} {'RMSE':>8}")
print("  " + "-" * 47)
for _, r in by_day.iterrows():
    print(f"  {r['day_of_week']:<12} {int(r['n']):>5}  {r['actual_avg']:>10.2f}  {r['MAE']:>8.3f}  {r['RMSE']:>8.3f}")
print()
print("  NOTE: est_wait_minutes accuracy cannot be directly evaluated")
print("  because ground-truth wait times are not recorded. The arrival")
print("  MAE/RMSE above is the primary reportable accuracy metric.")
print("=" * W)


# ── Save CSVs ─────────────────────────────────────────────────────────────────

out_cols = [
    "prediction_for", "predicted_at", "lead_minutes",
    "predicted_arrivals", "actual_arrivals",
    "error", "abs_error", "predicted_wait_min", "status",
    "hour", "day_of_week", "date",
]
merged[out_cols].to_csv("accuracy_results.csv", index=False)
by_hour.to_csv("accuracy_by_hour.csv",     index=False)
by_lead.to_csv("accuracy_by_leadtime.csv", index=False)
by_day.to_csv("accuracy_by_day.csv",       index=False)

print()
print("Saved:")
print("  accuracy_results.csv")
print("  accuracy_by_hour.csv")
print("  accuracy_by_leadtime.csv")
print("  accuracy_by_day.csv")
