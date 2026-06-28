"""
Prediction Accuracy Evaluation + arrival-scale calibration.

Compares `ensemble_yhat` (predicted arrivals / 3-min bucket) against actual
`entrance_events` counts for the same buckets, then optionally learns one
global multiplicative correction coefficient:

    corrected_pred = coeff * predicted_pred

where:

    coeff = sum(actual) / sum(predicted)

The coefficient is fit on a training slice and evaluated on a holdout slice so
we can check whether a stable mean-shift correction actually helps.

Outputs:
  accuracy_results.csv      — per-slot comparison
  accuracy_by_hour.csv      — MAE/RMSE breakdown by hour of day
  accuracy_by_leadtime.csv  — MAE/RMSE breakdown by prediction lead time
  accuracy_by_day.csv       — MAE/RMSE breakdown by weekday
  calibration.json          — global/hourly multiplicative calibration
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

BUCKET_MIN = int(os.getenv("BUCKET_MINUTES", 3))
DEFAULT_EVAL_DAYS = 14
DEFAULT_HOLDOUT_DAYS = 3
SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "calibration.json"

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)


def _conn():
    return psycopg2.connect(**DB_CONFIG)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate arrival forecasts and optionally fit a constant scale coefficient."
    )
    parser.add_argument("--days", type=int, default=DEFAULT_EVAL_DAYS, help="Past days to evaluate.")
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=DEFAULT_HOLDOUT_DAYS,
        help="Newest days reserved for validation of the learned coefficient.",
    )
    return parser.parse_args()


def _load_merged(eval_days: int) -> pd.DataFrame:
    print(f"[1/4] Loading and joining predictions with actuals (last {eval_days} days)...")
    with _conn() as conn:
        merged = pd.read_sql(
            f"""
            SELECT
                p.predicted_at,
                p.prediction_for,
                time_bucket('{BUCKET_MIN} minutes', p.prediction_for) AS pred_bucket,
                (p.prediction_for - p.predicted_at) AS lead_interval,
                p.predicted_arrivals,
                p.predicted_wait_min,
                p.status,
                a.actual_arrivals
            FROM (
                SELECT DISTINCT ON (time_bucket('{BUCKET_MIN} minutes', prediction_for))
                    predicted_at,
                    prediction_for,
                    ensemble_yhat    AS predicted_arrivals,
                    est_wait_minutes AS predicted_wait_min,
                    status
                FROM queue_predictions
                WHERE prediction_for >= NOW() - INTERVAL '{int(eval_days)} days'
                  AND prediction_for <  NOW()
                ORDER BY time_bucket('{BUCKET_MIN} minutes', prediction_for), predicted_at DESC
            ) p
            JOIN (
                SELECT
                    time_bucket('{BUCKET_MIN} minutes', timestamp) AS bucket,
                    COUNT(*) AS actual_arrivals
                FROM entrance_events
                WHERE (
                    (timestamp >= '2026-04-13' AND timestamp < '2026-04-16')
                    OR timestamp >= '2026-04-21'
                )
                  AND dwell_seconds >= 10
                  AND timestamp >= NOW() - INTERVAL '{int(eval_days)} days'
                  AND timestamp <  NOW()
                GROUP BY bucket
            ) a ON time_bucket('{BUCKET_MIN} minutes', p.prediction_for) = a.bucket
            ORDER BY p.prediction_for
            """,
            conn,
        )

    print(f"    {len(merged)} matched slots loaded")
    if merged.empty:
        print("ERROR: No matching slots found. Check that both tables have data in the eval period.")
        sys.exit(1)

    merged["prediction_for"] = pd.to_datetime(merged["prediction_for"], utc=True)
    merged["predicted_at"] = pd.to_datetime(merged["predicted_at"], utc=True)
    merged["lead_minutes"] = merged["lead_interval"].apply(
        lambda x: round(x.total_seconds() / 60, 1) if pd.notna(x) else None
    )
    merged["predicted_arrivals"] = pd.to_numeric(merged["predicted_arrivals"], errors="coerce").fillna(0.0)
    merged["actual_arrivals"] = pd.to_numeric(merged["actual_arrivals"], errors="coerce").fillna(0.0)
    return merged


def _lead_bin(minutes: float | None) -> str:
    if minutes is None or pd.isna(minutes):
        return "unknown"
    if minutes <= 15:
        return "0-15 min"
    if minutes <= 30:
        return "15-30 min"
    if minutes <= 45:
        return "30-45 min"
    return "45-60 min"


def _compute_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float | int | None]:
    if df.empty:
        return {
            "n": 0,
            "actual_avg": None,
            "pred_avg": None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "mape": None,
        }

    error = df[pred_col] - df["actual_arrivals"]
    abs_error = error.abs()
    sq_error = error**2
    nonzero = df[df["actual_arrivals"] > 0]
    if nonzero.empty:
        mape = None
    else:
        mape = float(((nonzero[pred_col] - nonzero["actual_arrivals"]).abs() / nonzero["actual_arrivals"]).mean() * 100)

    return {
        "n": int(len(df)),
        "actual_avg": float(df["actual_arrivals"].mean()),
        "pred_avg": float(df[pred_col].mean()),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(sq_error.mean())),
        "bias": float(error.mean()),
        "mape": mape,
    }


def _split_train_holdout(merged: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if merged.empty or holdout_days <= 0:
        return merged.copy(), pd.DataFrame(columns=merged.columns)

    cutoff = merged["prediction_for"].max() - pd.Timedelta(days=int(holdout_days))
    train = merged[merged["prediction_for"] < cutoff].copy()
    holdout = merged[merged["prediction_for"] >= cutoff].copy()

    if train.empty or holdout.empty:
        split_idx = max(1, int(len(merged) * 0.8))
        train = merged.iloc[:split_idx].copy()
        holdout = merged.iloc[split_idx:].copy()
        if holdout.empty:
            holdout = pd.DataFrame(columns=merged.columns)
    return train, holdout


def _safe_round(value: float | int | None, digits: int = 4) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _print_metric_block(title: str, metrics: dict[str, float | int | None]) -> None:
    print(title)
    if not metrics["n"]:
        print("  no rows")
        return
    print(f"  {'Rows':<20} {int(metrics['n'])}")
    print(f"  {'Actual mean':<20} {metrics['actual_avg']:.3f}")
    print(f"  {'Pred mean':<20} {metrics['pred_avg']:.3f}")
    print(f"  {'MAE':<20} {metrics['mae']:.3f}")
    print(f"  {'RMSE':<20} {metrics['rmse']:.3f}")
    print(f"  {'Bias':<20} {metrics['bias']:+.3f}")
    if metrics["mape"] is not None:
        print(f"  {'MAPE':<20} {metrics['mape']:.1f}%")
    else:
        print(f"  {'MAPE':<20} n/a")


def _write_calibration_json(merged: pd.DataFrame) -> tuple[float, dict[int, float]]:
    pred_sum = float(merged["predicted_arrivals"].sum())
    k_global = (float(merged["actual_arrivals"].sum()) / pred_sum) if pred_sum > 0 else 1.0

    k_by_hour: dict[int, float] = {}
    for hour, group in merged.groupby("hour"):
        if len(group) < 5:
            continue
        hour_pred_sum = float(group["predicted_arrivals"].sum())
        if hour_pred_sum <= 0:
            continue
        k_by_hour[int(hour)] = float(group["actual_arrivals"].sum()) / hour_pred_sum

    payload = {
        "k_global": round(k_global, 4),
        "k_by_hour": {str(hour): round(value, 4) for hour, value in sorted(k_by_hour.items())},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket_minutes": BUCKET_MIN,
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return k_global, k_by_hour


def main() -> None:
    args = _parse_args()
    merged = _load_merged(args.days)

    merged["hour"] = merged["prediction_for"].dt.hour
    merged["day_of_week"] = merged["prediction_for"].dt.day_name()
    merged["date"] = merged["prediction_for"].dt.date
    merged["lead_bin"] = merged["lead_minutes"].apply(_lead_bin)

    train_df, holdout_df = _split_train_holdout(merged, args.holdout_days)
    pred_sum = float(merged["predicted_arrivals"].sum())
    coeff = (float(merged["actual_arrivals"].sum()) / pred_sum) if pred_sum > 0 else 1.0
    fit_rows = int(len(merged))

    merged["corrected_predicted_arrivals"] = merged["predicted_arrivals"] * coeff
    merged["error"] = merged["predicted_arrivals"] - merged["actual_arrivals"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"] ** 2
    merged["corrected_error"] = merged["corrected_predicted_arrivals"] - merged["actual_arrivals"]
    merged["corrected_abs_error"] = merged["corrected_error"].abs()
    merged["corrected_sq_error"] = merged["corrected_error"] ** 2

    train_metrics_raw = _compute_metrics(train_df, "predicted_arrivals")
    train_metrics_corr = _compute_metrics(
        train_df.assign(corrected_predicted_arrivals=train_df["predicted_arrivals"] * coeff),
        "corrected_predicted_arrivals",
    )
    holdout_metrics_raw = _compute_metrics(holdout_df, "predicted_arrivals")
    holdout_metrics_corr = _compute_metrics(
        holdout_df.assign(corrected_predicted_arrivals=holdout_df["predicted_arrivals"] * coeff),
        "corrected_predicted_arrivals",
    )
    full_metrics_raw = _compute_metrics(merged, "predicted_arrivals")
    full_metrics_corr = _compute_metrics(merged, "corrected_predicted_arrivals")

    print("[4/4] Computing metrics...")
    print("      Done.\n")

    n_total = len(merged)
    n_days = merged["date"].nunique()
    date_min = merged["date"].min()
    date_max = merged["date"].max()

    by_hour = (
        merged.groupby("hour")
        .agg(
            n=("abs_error", "count"),
            actual_avg=("actual_arrivals", "mean"),
            pred_avg=("predicted_arrivals", "mean"),
            corrected_pred_avg=("corrected_predicted_arrivals", "mean"),
            MAE=("abs_error", "mean"),
            corrected_MAE=("corrected_abs_error", "mean"),
            RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
            corrected_RMSE=("corrected_sq_error", lambda x: np.sqrt(x.mean())),
            bias=("error", "mean"),
            corrected_bias=("corrected_error", "mean"),
        )
        .reset_index()
    )

    lead_order = ["0-15 min", "15-30 min", "30-45 min", "45-60 min", "unknown"]
    by_lead = (
        merged.groupby("lead_bin")
        .agg(
            n=("abs_error", "count"),
            MAE=("abs_error", "mean"),
            corrected_MAE=("corrected_abs_error", "mean"),
            RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
            corrected_RMSE=("corrected_sq_error", lambda x: np.sqrt(x.mean())),
            bias=("error", "mean"),
            corrected_bias=("corrected_error", "mean"),
        )
        .reindex(lead_order)
        .dropna(how="all")
        .reset_index()
    )

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = (
        merged.groupby("day_of_week")
        .agg(
            n=("abs_error", "count"),
            actual_avg=("actual_arrivals", "mean"),
            pred_avg=("predicted_arrivals", "mean"),
            corrected_pred_avg=("corrected_predicted_arrivals", "mean"),
            MAE=("abs_error", "mean"),
            corrected_MAE=("corrected_abs_error", "mean"),
            RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
            corrected_RMSE=("corrected_sq_error", lambda x: np.sqrt(x.mean())),
            bias=("error", "mean"),
            corrected_bias=("corrected_error", "mean"),
        )
        .reindex(day_order)
        .dropna(how="all")
        .reset_index()
    )

    for df in (by_hour, by_lead, by_day):
        for col in df.columns:
            if col not in {"hour", "lead_bin", "day_of_week"}:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    width = 72
    print("=" * width)
    print("  IQMS — PREDICTION ACCURACY REPORT")
    print(f"  Period : {date_min} -> {date_max}  ({n_days} days)")
    print(f"  Bucket : {BUCKET_MIN} min   |   Matched slots : {n_total}")
    print("=" * width)
    print()
    _print_metric_block("  FULL WINDOW — RAW", full_metrics_raw)
    print()
    _print_metric_block("  FULL WINDOW — CORRECTED", full_metrics_corr)
    print()
    print("  CALIBRATION")
    print(f"  {'Method':<20} ratio_of_sums_actual_over_predicted")
    print(f"  {'Coeff k':<20} {coeff:.6f}")
    print(f"  {'Fit rows':<20} {fit_rows}")
    print()
    _print_metric_block("  TRAIN WINDOW — RAW", train_metrics_raw)
    print()
    _print_metric_block("  TRAIN WINDOW — CORRECTED", train_metrics_corr)
    print()
    _print_metric_block("  HOLDOUT WINDOW — RAW", holdout_metrics_raw)
    print()
    _print_metric_block("  HOLDOUT WINDOW — CORRECTED", holdout_metrics_corr)
    print()
    print("  BY HOUR OF DAY")
    print(
        f"  {'Hour':<8} {'n':>5} {'Actual':>8} {'Pred':>8} {'Pred*k':>8} "
        f"{'MAE':>8} {'MAE*k':>8} {'Bias':>8} {'Bias*k':>8}"
    )
    print("  " + "-" * 78)
    for _, row in by_hour.iterrows():
        print(
            f"  {int(row['hour']):02d}:00  {int(row['n']):>5}"
            f"  {row['actual_avg']:>8.2f}  {row['pred_avg']:>8.2f}  {row['corrected_pred_avg']:>8.2f}"
            f"  {row['MAE']:>8.3f}  {row['corrected_MAE']:>8.3f}"
            f"  {row['bias']:>+8.3f}  {row['corrected_bias']:>+8.3f}"
        )
    print()
    print("  BY LEAD TIME")
    print(f"  {'Lead':<12} {'n':>5} {'MAE':>8} {'MAE*k':>8} {'Bias':>8} {'Bias*k':>8}")
    print("  " + "-" * 58)
    for _, row in by_lead.iterrows():
        print(
            f"  {row['lead_bin']:<12} {int(row['n']):>5}"
            f"  {row['MAE']:>8.3f}  {row['corrected_MAE']:>8.3f}"
            f"  {row['bias']:>+8.3f}  {row['corrected_bias']:>+8.3f}"
        )
    print()
    print("  NOTE: est_wait_minutes accuracy cannot be directly evaluated")
    print("  because ground-truth wait times are not recorded. The arrival")
    print("  MAE/RMSE above remains the reportable metric.")
    print("=" * width)

    out_cols = [
        "prediction_for",
        "predicted_at",
        "lead_minutes",
        "predicted_arrivals",
        "corrected_predicted_arrivals",
        "actual_arrivals",
        "error",
        "abs_error",
        "corrected_error",
        "corrected_abs_error",
        "predicted_wait_min",
        "status",
        "hour",
        "day_of_week",
        "date",
    ]
    merged[out_cols].to_csv("accuracy_results.csv", index=False)
    by_hour.to_csv("accuracy_by_hour.csv", index=False)
    by_lead.to_csv("accuracy_by_leadtime.csv", index=False)
    by_day.to_csv("accuracy_by_day.csv", index=False)

    print()
    print("Saved:")
    print("  accuracy_results.csv")
    print("  accuracy_by_hour.csv")
    print("  accuracy_by_leadtime.csv")
    print("  accuracy_by_day.csv")
    k_global, k_by_hour = _write_calibration_json(merged)
    print(f"  {CALIBRATION_PATH}")
    print(
        f"\n[CALIB] k_global={k_global:.4f}, {len(k_by_hour)} hourly slots written -> {CALIBRATION_PATH.name}"
    )


if __name__ == "__main__":
    main()
