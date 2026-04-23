"""
quick.py — Lightweight 60-minute Prophet forecast for low-latency use cases.

This module is a streamlined version of pipeline.py designed for dashboard
widgets and API endpoints that need a fast, up-to-date forecast without the
full overhead of dwell analysis, browsing-gap estimation, or multi-lane
scenario computation.

Key differences from pipeline.py
──────────────────────────────────
- Always bootstraps with synthetic history if real data span < MIN_REAL_DAYS_FOR_SIM
  (no opt-in flag required).
- Skips browsing-gap cross-correlation.
- Skips multi-lane "what-if" scenarios.
- Returns a minimal result dict suited for dashboard status cards.
"""
from __future__ import annotations

import pandas as pd
import psycopg2
from prophet import Prophet

from .core import (
    BUCKET_MINUTES,
    DEFAULT_DWELL_MIN,
    DEFAULT_LANES,
    DWELL_MIN_FLOOR,
    DWELL_MAX_CAP,
    FORECAST_STEPS,
    MAX_QUEUE_PER_LANE,
    MAX_WAIT_MIN,
    MIN_REAL_DAYS_FOR_SIM,
    SNAPSHOT_MAX_AGE_MIN,
    add_closed_zeros,
    build_sim_history,
    compute_wait_estimates,
)
from .pipeline import DB_CONFIG, latest_snapshot_for_wait, load_arrivals, load_snapshots


def run_quick_forecast(source: str = "REAL") -> dict:
    """Run a lightweight Prophet forecast for the next 60 minutes.

    Loads 30 days of arrival history, trains Prophet, and returns a compact
    forecast with wait-time estimates.  Automatically mixes in synthetic
    history when real data is sparse (< MIN_REAL_DAYS_FOR_SIM days).

    Parameters
    ----------
    source : str
        Data source filter: "REAL" (camera data only), "SIM" (simulator only),
        or "ALL" (both combined).  Default "REAL".

    Returns
    -------
    dict with keys:
        source           — source filter used
        df_combined      — training DataFrame fed to Prophet
        real_span_days   — span of real data in days
        forecast         — Prophet forecast DataFrame (ds, yhat, yhat_lower, yhat_upper)
        wait_estimates   — list of per-bucket dicts with keys "ds" and "wait_min"
        wait_15m         — estimated wait at +15 min (float or None)
        wait_30m         — estimated wait at +30 min (float or None)
        current_queue    — queue depth used for wait calculation
        avg_dwell_min    — dwell time used (clamped to ≥ 0.5)
        active_lanes     — lane count used for wait calculation
        service_per_bucket — throughput used in the wait model

    Raises
    ------
    RuntimeError
        If the combined training DataFrame is empty after all preparation steps.
    psycopg2.Error
        If the database connection fails.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    df_arrivals = load_arrivals(conn, source=source, days=30)
    df_snapshots = load_snapshots(conn, source=source, days=30)
    conn.close()

    df_real_agg = (
        df_arrivals.groupby("bucket")["entry_count"].sum()
        .reset_index()
        .rename(columns={"bucket": "ds", "entry_count": "y"})
    )
    if not df_real_agg.empty:
        df_real_agg["y"] = pd.to_numeric(df_real_agg["y"], errors="coerce").fillna(0).astype(int)

    real_span_days = (
        (df_real_agg["ds"].max() - df_real_agg["ds"].min()).total_seconds() / 86400
        if len(df_real_agg) > 1 else 0
    )

    if real_span_days >= MIN_REAL_DAYS_FOR_SIM:
        df_combined = df_real_agg.sort_values("ds").reset_index(drop=True)
    else:
        df_sim = build_sim_history(days=7)
        df_combined = (
            pd.concat([df_real_agg, df_sim], ignore_index=True)
            .drop_duplicates(subset="ds")
            .sort_values("ds")
            .reset_index(drop=True)
        )

    df_combined["ds"] = pd.to_datetime(df_combined["ds"], errors="coerce")
    df_combined["y"] = pd.to_numeric(df_combined["y"], errors="coerce").fillna(0)
    df_combined = df_combined.dropna(subset=["ds"])

    if df_combined.empty:
        raise RuntimeError("No valid data to train. Check the source filter or database contents.")

    non_zero = df_combined.loc[df_combined["y"] > 0, "y"]
    if not non_zero.empty:
        cap_value = max(float(non_zero.quantile(0.99)), 30.0)
        df_combined["y"] = df_combined["y"].clip(upper=cap_value)

    df_combined = add_closed_zeros(df_combined, days=30)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.80,
    )
    model.fit(df_combined[["ds", "y"]].copy())

    now = pd.Timestamp.now()
    future_df = pd.DataFrame(
        {
            "ds": [
                now + pd.Timedelta(minutes=BUCKET_MINUTES * (i + 1))
                for i in range(FORECAST_STEPS)
            ]
        }
    )
    forecast = model.predict(future_df)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast["yhat"] = forecast["yhat"].clip(lower=0).round(1)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round(1)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round(1)

    snap_latest = latest_snapshot_for_wait(df_snapshots, source)
    snap_age_min = None
    if snap_latest is not None and "timestamp" in snap_latest:
        snap_ts = pd.Timestamp(snap_latest["timestamp"])
        if snap_ts.tzinfo is not None:
            snap_ts = snap_ts.tz_localize(None)
        snap_age_min = (pd.Timestamp.now() - snap_ts).total_seconds() / 60

    if snap_latest is not None and (snap_age_min is None or snap_age_min < SNAPSHOT_MAX_AGE_MIN):
        current_queue = int(snap_latest["queue_count"])
        avg_dwell_min = float(snap_latest["avg_dwell_sec"]) / 60.0
        active_lanes  = max(1, int(snap_latest["active_lanes"]))
    else:
        current_queue = 0
        avg_dwell_min = DEFAULT_DWELL_MIN
        active_lanes  = DEFAULT_LANES

    est_service_min = min(max(avg_dwell_min, DWELL_MIN_FLOOR), DWELL_MAX_CAP)
    wait_rows, wait_15m, wait_30m, service_per_bucket = compute_wait_estimates(
        forecast,
        current_queue=current_queue,
        avg_dwell_min=est_service_min,
        active_lanes=active_lanes,
        max_queue_per_lane=20,
        max_wait_min=60.0,
    )

    return {
        "source": source,
        "df_combined": df_combined,
        "real_span_days": real_span_days,
        "forecast": forecast,
        "wait_estimates": wait_rows,
        "wait_15m": wait_15m,
        "wait_30m": wait_30m,
        "current_queue": current_queue,
        "avg_dwell_min": max(avg_dwell_min, 0.5),
        "active_lanes": active_lanes,
        "service_per_bucket": service_per_bucket,
    }
