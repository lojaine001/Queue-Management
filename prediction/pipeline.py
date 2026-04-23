"""
pipeline.py — Full ETL prediction pipeline for the IQMS queue management system.

Loads historical data from PostgreSQL (TimescaleDB), trains a Facebook Prophet
time-series model on bucketed arrival counts, forecasts the next 60 minutes of
customer arrivals, and estimates queue wait times for the current lane setup.

Also provides:
  - Multi-lane "what-if" scenarios (1–5 lanes)
  - Browsing-gap estimation via cross-correlation of entrance and service events
  - Stale-snapshot fallback using recent entrance-event lane depth

Database tables used
─────────────────────
  entrance_events        Bucketed entry counts with gender, age, dwell, lane depth
  queue_state_snapshots  Periodic queue-state snapshots (count, dwell, lanes)
  service_events         Checkout/service counter dwell times

Environment variables (DB connection)
──────────────────────────────────────
  DB_HOST      default "localhost"
  DB_PORT      default 5432
  DB_NAME      default "iqms"
  DB_USER      default "postgres"
  DB_PASSWORD  default "0000"
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from .core import (
    BUCKET_MINUTES,
    DEFAULT_DWELL_MIN,
    DEFAULT_LANES,
    DWELL_MIN_FLOOR,
    DWELL_MAX_CAP,
    MAX_QUEUE_PER_LANE,
    MAX_WAIT_MIN,
    MIN_REAL_DAYS_FOR_SIM,
    SHOP_CLOSE,
    SHOP_OPEN,
    SHOP_TIME_MAX_LAG_MIN,
    SNAPSHOT_MAX_AGE_MIN,
    add_closed_zeros,
    build_sim_history,
    compute_wait_estimates,
    future_open_timestamps,
    get_where_clause,
)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME",     "iqms"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "0000"),
}


def to_local(series: pd.Series) -> pd.Series:
    """Convert UTC-aware DB timestamps to local naive timestamps.

    TimescaleDB stores all timestamps as UTC.  This function applies the
    local UTC offset so that Prophet sees wall-clock times matching the
    shop's daily/weekly seasonality pattern.

    Parameters
    ----------
    series : pd.Series  Series of UTC timestamps (tz-aware or tz-naive strings).

    Returns
    -------
    pd.Series  Timezone-naive timestamps in local wall-clock time.
    """
    utc_offset = datetime.now(timezone.utc).astimezone().utcoffset()
    return pd.to_datetime(series).dt.tz_localize(None) + utc_offset


def load_arrivals(
    conn,
    source: str,
    days: int,
    data_since: datetime | None = None,
) -> pd.DataFrame:
    """Load bucketed customer arrival counts from entrance_events.

    Groups entrance events into BUCKET_MINUTES-wide time buckets and
    aggregates counts with gender, age, dwell, and lane-depth statistics.

    Parameters
    ----------
    conn       : psycopg2 connection  Active database connection.
    source     : str                  "REAL", "SIM", or "ALL".
    days       : int                  Lookback window in days (ignored if data_since set).
    data_since : datetime | None      Explicit start timestamp (UTC-aware); overrides `days`.

    Returns
    -------
    pd.DataFrame
        Columns: bucket, entry_count, male_count, female_count, avg_age,
                 avg_dwell_sec, max_lane_depth, avg_lane_depth, camera_id, source.
        Empty DataFrame if no rows match.
    """
    since = data_since.astimezone(timezone.utc) if data_since else (
        datetime.now(timezone.utc) - timedelta(days=days)
    )
    where_clause = get_where_clause(source)
    if where_clause:
        where_clause += f" AND timestamp >= '{since.isoformat()}'"
    else:
        where_clause = f"WHERE timestamp >= '{since.isoformat()}'"

    query = f"""
        SELECT
            time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS bucket,
            COUNT(*) AS entry_count,
            COUNT(*) FILTER (WHERE gender = 'male') AS male_count,
            COUNT(*) FILTER (WHERE gender = 'female') AS female_count,
            ROUND(AVG(age_estimate)::numeric, 1) AS avg_age,
            ROUND(AVG(dwell_seconds) FILTER (WHERE dwell_seconds > 0)::numeric, 1) AS avg_dwell_sec,
            MAX(active_head_tracks_in_lane) AS max_lane_depth,
            ROUND(AVG(active_head_tracks_in_lane)::numeric, 1) AS avg_lane_depth,
            camera_id
        FROM entrance_events
        {where_clause}
        GROUP BY bucket, camera_id
        ORDER BY bucket
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["bucket"] = to_local(df["bucket"])
    df["source"] = df["camera_id"].apply(lambda c: "SIM" if str(c).startswith("SIM_") else "REAL")
    return df


def load_today_actuals(source: str) -> pd.DataFrame:
    """Load today's bucketed arrivals directly from the DB (no app cache).

    Opens its own connection so it can be called independently of the
    main pipeline.  Returns an empty DataFrame on any connection error.

    Parameters
    ----------
    source : str  "REAL", "SIM", or "ALL".

    Returns
    -------
    pd.DataFrame
        Columns: bucket, entry_count, male_count, female_count, avg_age,
                 avg_dwell_sec, max_lane_depth, avg_lane_depth, camera_id, source.
    """
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    where_clause = get_where_clause(source)
    if where_clause:
        where_clause += f" AND timestamp >= '{today_utc.isoformat()}'"
    else:
        where_clause = f"WHERE timestamp >= '{today_utc.isoformat()}'"

    query = f"""
        SELECT
            time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS bucket,
            COUNT(*) AS entry_count,
            COUNT(*) FILTER (WHERE gender = 'male') AS male_count,
            COUNT(*) FILTER (WHERE gender = 'female') AS female_count,
            ROUND(AVG(age_estimate)::numeric, 1) AS avg_age,
            ROUND(AVG(dwell_seconds) FILTER (WHERE dwell_seconds > 0)::numeric, 1) AS avg_dwell_sec,
            MAX(active_head_tracks_in_lane) AS max_lane_depth,
            ROUND(AVG(active_head_tracks_in_lane)::numeric, 1) AS avg_lane_depth,
            camera_id
        FROM entrance_events
        {where_clause}
        GROUP BY bucket, camera_id
        ORDER BY bucket
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn)
        conn.close()
    except Exception:
        return pd.DataFrame(columns=["bucket", "entry_count"])

    if df.empty:
        return df
    df["bucket"] = to_local(df["bucket"])
    df["entry_count"] = pd.to_numeric(df["entry_count"], errors="coerce").fillna(0)
    df["source"] = df["camera_id"].apply(lambda c: "SIM" if str(c).startswith("SIM_") else "REAL")
    return df


def load_snapshots(
    conn,
    source: str,
    days: int,
    data_since: datetime | None = None,
) -> pd.DataFrame:
    """Load periodic queue-state snapshots from queue_state_snapshots.

    Each row represents a point-in-time snapshot of queue depth, average
    dwell time, and the number of active service lanes.

    Parameters
    ----------
    conn       : psycopg2 connection  Active database connection.
    source     : str                  "REAL", "SIM", or "ALL".
    days       : int                  Lookback window in days.
    data_since : datetime | None      Explicit start timestamp; overrides `days`.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (local), queue_count, avg_dwell_sec, active_lanes,
                 camera_id, source.
    """
    since = data_since.astimezone(timezone.utc) if data_since else (
        datetime.now(timezone.utc) - timedelta(days=days)
    )
    where_clause = get_where_clause(source)
    if where_clause:
        where_clause += f" AND timestamp >= '{since.isoformat()}'"
    else:
        where_clause = f"WHERE timestamp >= '{since.isoformat()}'"

    query = f"""
        SELECT timestamp, queue_count, avg_dwell_sec, active_lanes, camera_id
        FROM queue_state_snapshots
        {where_clause}
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["timestamp"] = to_local(df["timestamp"])
    df["source"] = df["camera_id"].apply(lambda c: "SIM" if str(c).startswith("SIM_") else "REAL")
    return df


def load_dwell_events(conn, days: int = 1) -> pd.DataFrame:
    """Load raw individual entrance-queue dwell times (REAL cameras only).

    Returns one row per track with a local timestamp and dwell_min.
    Limited to the last `days` days to keep the query fast.

    Parameters
    ----------
    conn : psycopg2 connection  Active database connection.
    days : int                  Lookback window in days (default 1).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (local), dwell_min.
        Empty DataFrame if no rows match.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = f"""
        SELECT timestamp, dwell_seconds / 60.0 AS dwell_min
        FROM entrance_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND dwell_seconds > 0
          AND timestamp >= '{since.isoformat()}'
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["timestamp"] = to_local(df["timestamp"])
    return df


def load_service_events(conn, days: int = 30) -> pd.DataFrame:
    """Load individual service/checkout dwell times from service_events (REAL only).

    Each row represents one service completion.  service_min is the total time
    spent at the checkout counter — used alongside entrance-queue dwell to
    estimate the full customer journey time.

    Parameters
    ----------
    conn : psycopg2 connection  Active database connection.
    days : int                  Lookback window in days (default 30).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (local), service_min, camera_id.
        Returns an empty DataFrame with correct columns on any DB error.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = f"""
        SELECT timestamp, total_dwell_sec / 60.0 AS service_min,
               camera_id
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND total_dwell_sec > 0
          AND timestamp >= '{since.isoformat()}'
        ORDER BY timestamp
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "service_min", "camera_id"])
    if df.empty:
        return df
    df["timestamp"] = to_local(df["timestamp"])
    return df


def bucketed_counts(df: pd.DataFrame,
                    time_col: str,
                    label: str) -> pd.DataFrame:
    """Resample a timestamped DataFrame to BUCKET_MINUTES event counts.

    Used internally by estimate_browsing_gap() to align entrance-exit and
    service-arrival timelines on the same bucket grid before cross-correlating.

    Parameters
    ----------
    df       : pd.DataFrame  DataFrame containing a datetime column.
    time_col : str           Name of the datetime column to bucket.
    label    : str           Name to assign to the resulting Series.

    Returns
    -------
    pd.Series
        Event counts indexed by bucket timestamp, named `label`.
        Empty Series if df is empty or time_col is missing.
    """
    if df.empty or time_col not in df.columns:
        return pd.Series(dtype=float, name=label)
    ts = pd.to_datetime(df[time_col])
    counts = (
        ts.dt.floor(f"{BUCKET_MINUTES}min")
        .value_counts()
        .sort_index()
        .rename(label)
    )
    return counts


def estimate_browsing_gap(
    df_dwell: pd.DataFrame,
    df_service: pd.DataFrame,
    max_lag_min: int = 120,
) -> dict:
    """Estimate the unmeasured browsing gap via cross-correlation.

    Cross-correlates today's bucketed entrance-exit timestamps against today's
    bucketed service-arrival timestamps.  The lag at peak Pearson correlation
    represents the estimated time between a customer leaving the entrance queue
    and arriving at the service counter (i.e. the browsing/shopping time).

    Only uses today's data so the estimate reflects current store conditions.
    Returns None for peak_lag_min if the peak correlation is below 0.25
    (not enough signal).

    Parameters
    ----------
    df_dwell   : pd.DataFrame  Entrance dwell events (columns: timestamp, dwell_min).
    df_service : pd.DataFrame  Service events (columns: timestamp, service_min).
    max_lag_min : int          Maximum lag to test in minutes (default 120).

    Returns
    -------
    dict with keys:
        peak_lag_min     — estimated browsing gap in minutes (None if low signal)
        correlation      — Pearson r at peak lag (None if low signal)
        lags             — list of tested lags in minutes
        correlations     — list of Pearson r values (same order as lags)
        avg_entrance_min — mean entrance-queue dwell across loaded events
        avg_service_min  — mean service-counter dwell across loaded events
        est_total_min    — peak_lag + avg_entrance + avg_service (None if no signal)
    """
    import numpy as np

    avg_entrance = float(df_dwell["dwell_min"].mean()) if not df_dwell.empty else 0.0
    avg_service  = float(df_service["service_min"].mean()) if not df_service.empty else 0.0

    base = {
        "peak_lag_min": None, "correlation": None,
        "lags": [], "correlations": [],
        "avg_entrance_min": round(avg_entrance, 1),
        "avg_service_min":  round(avg_service,  1),
        "est_total_min":    None,
    }

    if df_dwell.empty or df_service.empty:
        return base

    today = pd.Timestamp.now().normalize()
    dwell_today   = df_dwell[df_dwell["timestamp"] >= today]
    service_today = df_service[df_service["timestamp"] >= today]
    if dwell_today.empty or service_today.empty:
        return base

    entry_counts   = bucketed_counts(dwell_today,   "timestamp", "entries")
    service_counts = bucketed_counts(service_today, "timestamp", "services")

    all_idx = entry_counts.index.union(service_counts.index)
    e = entry_counts.reindex(all_idx, fill_value=0).values.astype(float)
    s = service_counts.reindex(all_idx, fill_value=0).values.astype(float)

    max_steps = max_lag_min // BUCKET_MINUTES
    lags_min, corrs = [], []

    for step in range(0, max_steps + 1):
        lag_min = step * BUCKET_MINUTES
        s_shifted = s[step:] if step > 0 else s
        e_aligned = e[:-step]  if step > 0 else e
        if len(e_aligned) < 4 or e_aligned.std() == 0 or s_shifted.std() == 0:
            lags_min.append(lag_min)
            corrs.append(0.0)
            continue
        r = float(np.corrcoef(e_aligned, s_shifted)[0, 1])
        lags_min.append(lag_min)
        corrs.append(round(r, 4))

    if not corrs:
        return base

    peak_idx  = int(np.argmax(corrs))
    peak_lag  = lags_min[peak_idx]
    peak_corr = corrs[peak_idx]

    if peak_corr < 0.25:
        peak_lag = None

    est_total = (
        round(avg_entrance + peak_lag + avg_service, 1)
        if peak_lag is not None else None
    )

    return {
        "peak_lag_min":      peak_lag,
        "correlation":       peak_corr if peak_lag is not None else None,
        "lags":              lags_min,
        "correlations":      corrs,
        "avg_entrance_min":  round(avg_entrance, 1),
        "avg_service_min":   round(avg_service,  1),
        "est_total_min":     est_total,
    }


def latest_snapshot_for_wait(df_snapshots: pd.DataFrame, source: str):
    """Return the most recent queue-state snapshot row for wait estimation.

    When source is "ALL", prefers REAL camera snapshots over simulator ones
    so that wait estimates reflect actual store conditions.

    Parameters
    ----------
    df_snapshots : pd.DataFrame  Snapshot DataFrame from load_snapshots().
    source       : str           "REAL", "SIM", or "ALL".

    Returns
    -------
    pd.Series | None
        The last snapshot row, or None if df_snapshots is empty.
    """
    if df_snapshots.empty:
        return None
    if source == "ALL" and "camera_id" in df_snapshots.columns:
        real_snaps = df_snapshots[~df_snapshots["camera_id"].str.startswith("SIM_", na=False)]
        if not real_snaps.empty:
            return real_snaps.iloc[-1]
    return df_snapshots.iloc[-1]


def run_prediction_pipeline(
    source: str = "REAL",
    days: int = 30,
    use_bootstrap: bool = False,
    data_since: datetime | None = None,
) -> dict:
    """Run the full Prophet-based prediction pipeline.

    Steps
    -----
    1. Load arrivals, snapshots, dwell events, and service events from DB.
    2. Aggregate arrivals into BUCKET_MINUTES buckets.
    3. Optionally mix in synthetic history if real data span < MIN_REAL_DAYS_FOR_SIM.
    4. Cap outliers at 99th percentile and fill closed-hour gaps with zeros.
    5. Train a Prophet model with daily + weekly seasonality.
    6. Predict in-sample (today's actuals) and future (next 60 open minutes).
    7. Derive current queue state from the latest snapshot (or fall back to
       recent entrance-event lane depth if snapshot is stale/missing).
    8. Compute wait estimates for the current lane count and for 1–5 lane scenarios.
    9. Estimate the browsing gap via cross-correlation.

    Parameters
    ----------
    source        : str              "REAL", "SIM", or "ALL" (default "REAL").
    days          : int              Lookback window for historical data (default 30).
    use_bootstrap : bool             Mix synthetic history if real data is sparse (default False).
    data_since    : datetime | None  Explicit start timestamp; overrides `days`.

    Returns
    -------
    dict with keys:
        df_arrivals      — raw bucketed arrivals DataFrame
        df_snapshots     — queue-state snapshots DataFrame
        df_dwell         — entrance dwell events DataFrame
        df_service       — service counter dwell events DataFrame
        browsing_est     — browsing-gap estimation dict (from estimate_browsing_gap)
        df_combined      — final training DataFrame fed to Prophet
        df_insample      — Prophet in-sample predictions for today
        forecast         — Prophet forecast DataFrame for the next 60 minutes
        comp_df          — 7-day component prediction for seasonality charts
        wait_estimates   — list of per-bucket wait dicts for current lane count
        wait_15m         — estimated wait at +15 min (float or None)
        wait_30m         — estimated wait at +30 min (float or None)
        current_queue    — queue depth used for wait calculation
        active_lanes     — lane count used for wait calculation
        avg_dwell_min    — dwell time used (from snapshot or default)
        est_service_min  — clamped service time used in wait model
        lane_scenarios   — dict[int, dict] with wait estimates for 1–5 lanes
        real_span_days   — span of real data in days
        used_sim_history — True if synthetic history was mixed in
        source           — source filter used
        days             — lookback days used
    """
    from prophet import Prophet  # noqa: PLC0415

    conn = psycopg2.connect(**DB_CONFIG)
    df_arrivals  = load_arrivals(conn, source, days, data_since)
    df_snapshots = load_snapshots(conn, source, days, data_since)
    df_dwell     = load_dwell_events(conn, days=max(days, 7))
    df_service   = load_service_events(conn, days=max(days, 7))
    conn.close()

    df_real_agg = (
        df_arrivals.groupby("bucket")["entry_count"].sum()
        .reset_index()
        .rename(columns={"bucket": "ds", "entry_count": "y"})
    )
    if not df_real_agg.empty:
        df_real_agg["y"] = pd.to_numeric(df_real_agg["y"], errors="coerce").fillna(0).astype(int)

    real_span = (
        (df_real_agg["ds"].max() - df_real_agg["ds"].min()).total_seconds() / 86400
        if len(df_real_agg) > 1 else 0
    )

    used_sim = False
    if use_bootstrap and real_span < MIN_REAL_DAYS_FOR_SIM:
        df_sim_hist = build_sim_history(days=7)
        df_combined = (
            pd.concat([df_real_agg, df_sim_hist])
            .drop_duplicates(subset="ds")
            .sort_values("ds")
            .reset_index(drop=True)
        )
        used_sim = True
    else:
        df_combined = df_real_agg.sort_values("ds").reset_index(drop=True)

    non_zero = df_combined.loc[df_combined["y"] > 0, "y"]
    cap_value: float = 9999.0
    if not non_zero.empty:
        cap_value = max(float(non_zero.quantile(0.99)), 30.0)
        df_combined["y"] = df_combined["y"].clip(upper=cap_value)
    if "entry_count" in df_arrivals.columns:
        df_arrivals = df_arrivals.copy()
        df_arrivals["entry_count"] = df_arrivals["entry_count"].clip(upper=cap_value)

    train_days = max(days, 7)
    df_combined = add_closed_zeros(df_combined, train_days)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.80,
    )
    model.fit(df_combined)

    now_floor = pd.Timestamp.now().floor(f"{BUCKET_MINUTES}min")
    today_open = now_floor.replace(hour=SHOP_OPEN, minute=0, second=0, microsecond=0)
    today_ts = pd.date_range(start=today_open, end=now_floor, freq=f"{BUCKET_MINUTES}min")
    df_insample = model.predict(pd.DataFrame({"ds": today_ts}))

    future_ts = future_open_timestamps(interval_min=BUCKET_MINUTES)
    future_df = pd.DataFrame({"ds": future_ts})
    forecast = model.predict(future_df)
    forecast["yhat"] = forecast["yhat"].clip(lower=0).round(1)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round(1)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round(1)

    comp_df = model.predict(
        pd.DataFrame(
            {
                "ds": pd.date_range(
                    start=df_combined["ds"].min(),
                    periods=7 * 24 * (60 // BUCKET_MINUTES),
                    freq=f"{BUCKET_MINUTES}min",
                )
            }
        )
    )

    snap_latest = latest_snapshot_for_wait(df_snapshots, source)
    snap_age_min = None
    if snap_latest is not None and "timestamp" in snap_latest:
        snap_ts = pd.Timestamp(snap_latest["timestamp"])
        if snap_ts.tzinfo is not None:
            snap_ts = snap_ts.tz_localize(None)
        snap_age_min = (pd.Timestamp.now() - snap_ts).total_seconds() / 60

    if snap_latest is not None and (snap_age_min is None or snap_age_min < SNAPSHOT_MAX_AGE_MIN):
        current_queue = int(snap_latest["queue_count"])
        avg_dwell_min = float(snap_latest["avg_dwell_sec"]) / 60
        active_lanes  = int(snap_latest["active_lanes"])
    else:
        # Snapshot stale/missing — derive current queue depth from recent entrance events
        # (max_lane_depth = max active heads in a single lane at exit time, last 3 buckets)
        if "max_lane_depth" in df_arrivals.columns:
            real_recent = df_arrivals[df_arrivals["source"] != "SIM"].tail(3)
            current_queue = int(real_recent["max_lane_depth"].max()) if not real_recent.empty else 0
        else:
            current_queue = 0
        avg_dwell_min = DEFAULT_DWELL_MIN
        active_lanes  = DEFAULT_LANES
    avg_dwell_min = max(avg_dwell_min, 0.5)

    est_service_min  = min(max(avg_dwell_min, DWELL_MIN_FLOOR), DWELL_MAX_CAP)
    browsing_est     = estimate_browsing_gap(df_dwell, df_service,
                                             max_lag_min=SHOP_TIME_MAX_LAG_MIN)
    wait_estimates, wait_15m, wait_30m, _service_per_bucket = compute_wait_estimates(
        forecast[["ds", "yhat"]].copy(),
        current_queue=current_queue,
        avg_dwell_min=est_service_min,
        active_lanes=active_lanes,
        max_queue_per_lane=MAX_QUEUE_PER_LANE,
        max_wait_min=MAX_WAIT_MIN,
    )

    # ── Multi-lane scenarios (1–5 lanes) ──────────────────────────────────────
    lane_scenarios: dict[int, dict] = {}
    for n in range(1, 6):
        w_rows, w15, w30, _ = compute_wait_estimates(
            forecast[["ds", "yhat"]].copy(),
            current_queue=current_queue,
            avg_dwell_min=est_service_min,
            active_lanes=n,
            max_queue_per_lane=MAX_QUEUE_PER_LANE,
            max_wait_min=MAX_WAIT_MIN,
        )
        lane_scenarios[n] = {
            "wait_estimates": w_rows,
            "wait_15m": w15,
            "wait_30m": w30,
        }

    return {
        "df_arrivals":     df_arrivals,
        "df_snapshots":    df_snapshots,
        "df_dwell":        df_dwell,
        "df_service":      df_service,
        "browsing_est":    browsing_est,
        "df_combined":     df_combined,
        "df_insample":     df_insample,
        "forecast":        forecast,
        "comp_df":         comp_df,
        "wait_estimates":  wait_estimates,
        "wait_15m":        wait_15m,
        "wait_30m":        wait_30m,
        "current_queue":   current_queue,
        "active_lanes":    active_lanes,
        "avg_dwell_min":   avg_dwell_min,
        "est_service_min": est_service_min,
        "lane_scenarios":  lane_scenarios,
        "real_span_days":  real_span,
        "used_sim_history": used_sim,
        "source":          source,
        "days":            days,
    }
