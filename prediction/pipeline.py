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
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from .core import (
    BUCKET_MINUTES,
    DEFAULT_DWELL_MIN,
    FORECAST_HORIZON_MINUTES,
    DEFAULT_LANES,
    DWELL_MIN_FLOOR,
    DWELL_MAX_CAP,
    ENTRANCE_CAMERA_ID,
    EXIT_CAMERA_ID,
    MAX_QUEUE_PER_LANE,
    MAX_WAIT_MIN,
    MIN_DEMAND_BUCKETS,
    MIN_REAL_DAYS_FOR_SIM,
    SERVICE_MIN_DWELL_SEC,
    SHOP_CLOSE,
    SHOP_CLOSE_HOUR,
    SHOP_CLOSE_MINUTE,
    SHOP_CLOSE_TOT,
    SHOP_OPEN,
    SHOP_OPEN_MINUTE,
    SHOP_OPEN_TOT,
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
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    _local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"

    where_clause = get_where_clause(source)
    if where_clause:
        where_clause += f" AND timestamp >= '{since.isoformat()}'"
    else:
        where_clause = f"WHERE timestamp >= '{since.isoformat()}'"

    # Restrict REAL data to the entrance camera only — the exit camera also writes
    # to entrance_events (head counter) and would double-count arrivals.
    where_clause += f" AND (camera_id LIKE 'SIM_%%' OR camera_id = '{ENTRANCE_CAMERA_ID}')"
    # Strict shop-hours filter in local time so out-of-hours rows never reach Prophet.
    where_clause += (
        f" AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) >= {SHOP_OPEN_TOT}"
        f" AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) < {SHOP_CLOSE_TOT}"
    )

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
    local_midnight = datetime.now().astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    today_utc = local_midnight.astimezone(timezone.utc)
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    _local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"

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
          AND (camera_id LIKE 'SIM_%%' OR camera_id = '{ENTRANCE_CAMERA_ID}')
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) < {SHOP_CLOSE_TOT}
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
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    _local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"

    where_clause = get_where_clause(source)
    if where_clause:
        where_clause += f" AND timestamp >= '{since.isoformat()}'"
    else:
        where_clause = f"WHERE timestamp >= '{since.isoformat()}'"

    query = f"""
        SELECT timestamp, queue_count, avg_dwell_sec, active_lanes, camera_id
        FROM queue_state_snapshots
        {where_clause}
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) < {SHOP_CLOSE_TOT}
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
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    _local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"
    query = f"""
        SELECT timestamp, dwell_seconds / 60.0 AS dwell_min
        FROM entrance_events
        WHERE camera_id = '{ENTRANCE_CAMERA_ID}'
          AND dwell_seconds > 0
          AND timestamp >= '{since.isoformat()}'
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) < {SHOP_CLOSE_TOT}
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["timestamp"] = to_local(df["timestamp"])
    return df


def save_prediction(wait_15m: float, wait_30m: float, prophet_yhat: float) -> None:
    """Persist the current wait prediction to queue_predictions for history tracking."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        now_utc = datetime.now(timezone.utc)
        pred_for = now_utc + timedelta(minutes=15)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queue_predictions (
                id               BIGSERIAL,
                predicted_at     TIMESTAMPTZ NOT NULL,
                prediction_for   TIMESTAMPTZ NOT NULL,
                prophet_yhat     NUMERIC(8,2),
                est_wait_minutes NUMERIC(8,2),
                wait_15m         NUMERIC(8,2),
                wait_30m         NUMERIC(8,2),
                status           VARCHAR(10)
            )
        """)
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'queue_predictions' AND column_name = 'pipeline_source'
                ) THEN
                    ALTER TABLE queue_predictions ADD COLUMN pipeline_source VARCHAR(20);
                END IF;
            END $$;
        """)
        cur.execute("""
            INSERT INTO queue_predictions
                (predicted_at, prediction_for, prophet_yhat, est_wait_minutes, wait_15m, wait_30m, status, pipeline_source)
            VALUES (%s, %s, %s, %s, %s, %s, 'OK', 'pipeline_v2')
            ON CONFLICT (prediction_for) DO UPDATE SET
                predicted_at     = EXCLUDED.predicted_at,
                prophet_yhat     = EXCLUDED.prophet_yhat,
                est_wait_minutes = EXCLUDED.est_wait_minutes,
                wait_15m         = EXCLUDED.wait_15m,
                wait_30m         = EXCLUDED.wait_30m,
                status           = 'OK',
                pipeline_source  = 'pipeline_v2'
        """, (now_utc, pred_for, round(prophet_yhat, 2),
              round(wait_15m, 2), round(wait_15m, 2), round(wait_30m, 2)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass  # never crash the pipeline because of a save failure


def load_prediction_history(conn, days: int = 1) -> pd.DataFrame:
    """Load today's wait_15m predictions from queue_predictions.

    Returns columns: prediction_for (local) — the time being predicted,
    and wait_15m. Plotted at prediction_for so the curve aligns with the
    observed wait at the same timestamp on the chart.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = f"""
        SELECT predicted_at, prediction_for, wait_15m
        FROM queue_predictions
        WHERE predicted_at >= '{since.isoformat()}'
          AND wait_15m IS NOT NULL
          AND pipeline_source = 'pipeline_v2'
        ORDER BY prediction_for
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame(columns=["predicted_at", "prediction_for", "wait_15m"])
    if df.empty:
        return df
    df["predicted_at"]   = to_local(df["predicted_at"])
    df["prediction_for"] = to_local(df["prediction_for"])
    df["wait_15m"] = pd.to_numeric(df["wait_15m"], errors="coerce")
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
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    _local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"
    query = f"""
        SELECT timestamp, total_dwell_sec / 60.0 AS service_min,
               total_dwell_sec, lane_id, camera_id
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND total_dwell_sec > 0
          AND timestamp >= '{since.isoformat()}'
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) < {SHOP_CLOSE_TOT}
        ORDER BY timestamp
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "service_min", "total_dwell_sec", "lane_id", "camera_id"])
    if df.empty:
        return df
    df["timestamp"] = to_local(df["timestamp"])
    return df


def load_lane_history_agg(
    conn,
    days: int,
    shop_open: int,
    shop_close: int,
    checkout_service_min: float,
    service_min_dwell_sec: int,
    slot_minutes: int = 15,
) -> pd.DataFrame:
    """Three-source merge for median active lanes per slot_minutes time-of-day slot.

    Phase 1 — service_events with lane_id (best): COUNT(DISTINCT lane_id) per (day, slot).
    Phase 2 — queue_state_snapshots.active_lanes: median per slot across all days.
    Phase 3 — entrance_events arrival rate: median count per slot, returned raw so
               the chart can scale it to demand-implied lanes using checkout_fraction.

    Returns one row per slot (all open hours) with columns:
      slot_label, median_lanes, lanes_source, n_days_observed,
      snapshot_lanes, n_days_snapshot,
      median_entrance_per_30min, n_days_entrance
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_str = since.isoformat()
    # Convert UTC timestamps to local time in SQL so slot labels match the local-time
    # slot labels produced by _forecast_lanes_vector (which uses local pd.Timestamps).
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    _local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"
    slot_expr = (
        f"LPAD(EXTRACT(HOUR FROM {_local_ts})::int::text, 2, '0') || ':' || "
        f"LPAD((FLOOR(EXTRACT(MINUTE FROM {_local_ts}) / {slot_minutes}) * {slot_minutes})::int::text, 2, '0')"
    )
    hour_filter = (
        f"(EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) >= {SHOP_OPEN_TOT} "
        f"AND (EXTRACT(HOUR FROM {_local_ts}) * 60 + EXTRACT(MINUTE FROM {_local_ts})) < {SHOP_CLOSE_TOT}"
    )
    capacity_per_slot = float(slot_minutes) / max(checkout_service_min, 0.5)

    empty = pd.DataFrame(columns=[
        "slot_label", "median_lanes", "lanes_source", "n_days_observed",
        "snapshot_lanes", "n_days_snapshot",
        "median_entrance_per_slot", "n_days_entrance",
    ])

    # ── Phase 1: service_events ───────────────────────────────────────────────
    q1 = f"""
        SELECT {slot_expr} AS slot_label,
               DATE(timestamp) AS day,
               COUNT(DISTINCT lane_id) FILTER (WHERE lane_id IS NOT NULL) AS distinct_lanes,
               COUNT(*) FILTER (WHERE total_dwell_sec > {service_min_dwell_sec}) AS qualifying
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND {hour_filter}
          AND timestamp >= '{since_str}'
        GROUP BY 1, 2
    """
    try:
        df1 = pd.read_sql(q1, conn)
    except Exception:
        df1 = pd.DataFrame()

    if not df1.empty:
        def _lane_est(row):
            if row["distinct_lanes"] > 0:
                return int(row["distinct_lanes"]), "lane_id"
            if row["qualifying"] > 0:
                return max(1, round(row["qualifying"] / capacity_per_slot)), "throughput"
            return 0, "none"
        df1[["lane_est", "source"]] = pd.DataFrame(
            df1.apply(_lane_est, axis=1).tolist(), index=df1.index
        )
        df1 = df1[df1["lane_est"] > 0]
        # Dominant source per slot
        src_per_slot = (
            df1.groupby("slot_label")["source"]
            .agg(lambda s: s.value_counts().idxmax())
            .reset_index()
            .rename(columns={"source": "lanes_source"})
        )
        agg1 = (
            df1.groupby("slot_label")["lane_est"]
            .agg(median_lanes="median", n_days_observed="count")
            .reset_index()
            .merge(src_per_slot, on="slot_label")
        )
    else:
        agg1 = pd.DataFrame(columns=["slot_label", "median_lanes", "n_days_observed", "lanes_source"])

    # ── Phase 2: queue_state_snapshots ────────────────────────────────────────
    q2 = f"""
        SELECT {slot_expr.replace('timestamp', 'timestamp')} AS slot_label,
               DATE(timestamp) AS day,
               AVG(active_lanes) AS avg_lanes
        FROM queue_state_snapshots
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND {hour_filter}
          AND timestamp >= '{since_str}'
        GROUP BY 1, 2
    """
    try:
        df2 = pd.read_sql(q2, conn)
    except Exception:
        df2 = pd.DataFrame()

    if not df2.empty:
        agg2 = (
            df2.groupby("slot_label")["avg_lanes"]
            .agg(snapshot_lanes="median", n_days_snapshot="count")
            .reset_index()
        )
    else:
        agg2 = pd.DataFrame(columns=["slot_label", "snapshot_lanes", "n_days_snapshot"])

    # ── Phase 3: entrance_events arrival count ────────────────────────────────
    q3 = f"""
        SELECT {slot_expr} AS slot_label,
               DATE(timestamp) AS day,
               COUNT(*) AS arrivals
        FROM entrance_events
        WHERE camera_id = '{ENTRANCE_CAMERA_ID}'
          AND {hour_filter}
          AND timestamp >= '{since_str}'
        GROUP BY 1, 2
    """
    try:
        df3 = pd.read_sql(q3, conn)
    except Exception:
        df3 = pd.DataFrame()

    if not df3.empty:
        agg3 = (
            df3.groupby("slot_label")["arrivals"]
            .agg(median_entrance_per_slot="median", n_days_entrance="count")
            .reset_index()
        )
    else:
        agg3 = pd.DataFrame(columns=["slot_label", "median_entrance_per_slot", "n_days_entrance"])

    # ── Build full slot index and merge all three phases ──────────────────────
    all_slots = []
    h = SHOP_OPEN
    while h < SHOP_CLOSE:
        for m in range(0, 60, slot_minutes):
            if SHOP_OPEN_TOT <= h * 60 + m < SHOP_CLOSE_TOT:
                all_slots.append(f"{h:02d}:{m:02d}")
        h += 1
    result = pd.DataFrame({"slot_label": all_slots})

    for agg, defaults in [
        (agg1, {"median_lanes": 0.0, "n_days_observed": 0, "lanes_source": "none"}),
        (agg2, {"snapshot_lanes": 0.0, "n_days_snapshot": 0}),
        (agg3, {"median_entrance_per_slot": 0.0, "n_days_entrance": 0}),
    ]:
        if not agg.empty:
            result = result.merge(agg, on="slot_label", how="left")
        else:
            for col, val in defaults.items():
                result[col] = val
        result = result.fillna(defaults)

    return result


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


def build_service_time_profile(df_service: pd.DataFrame) -> dict:
    """Build a (day_of_week, hour) → median service_min lookup from service_events."""
    if df_service.empty or "service_min" not in df_service.columns:
        return {}
    df = df_service.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce")
    df = df.dropna(subset=["timestamp", "service_min"])
    df = df[df["service_min"] >= DWELL_MIN_FLOOR]
    if df.empty:
        return {}
    df["dow"]  = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    return (
        df.groupby(["dow", "hour"])["service_min"]
        .median()
        .to_dict()
    )


def service_time_for_timestamp(
    profile: dict,
    ts: pd.Timestamp,
    fallback: float,
) -> float:
    """Look up predicted service time for a timestamp; fall back gracefully."""
    val = profile.get((ts.dayofweek, ts.hour))
    if val is None:
        same_hour = [v for (d, h), v in profile.items() if h == ts.hour]
        val = float(np.median(same_hour)) if same_hour else fallback
    return float(min(max(val, DWELL_MIN_FLOOR), DWELL_MAX_CAP))


def _preferred_service_minutes(
    df_service: pd.DataFrame,
    snapshot_dwell_min: float | None,
) -> tuple[float, str, float | None, float | None]:
    """Choose the best per-customer service estimate for the wait model."""
    service_mean_min = None
    service_median_min = None

    if not df_service.empty and "service_min" in df_service.columns:
        service_vals = pd.to_numeric(df_service["service_min"], errors="coerce")
        service_vals = service_vals[(service_vals >= DWELL_MIN_FLOOR) & service_vals.notna()]
        if not service_vals.empty:
            service_mean_min = float(service_vals.mean())
            service_median_min = float(service_vals.median())

    if service_median_min is not None:
        return service_median_min, "service_events_median", service_median_min, service_mean_min
    checkout_service = float(os.getenv("CHECKOUT_SERVICE_MIN", DEFAULT_DWELL_MIN))
    return checkout_service, "config_default", service_median_min, service_mean_min


def evaluate_service_correction_factor(
    source: str = "REAL",
    days: int = 20,
) -> dict:
    """Evaluate a historical correction factor for service time and queue wait.

    Uses the last `days` of service_events and queue_state_snapshots to
    compare observed checkout throughput against the current wait model
    service-time assumption.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df_service = load_service_events(conn, days=days)
        df_snapshots = load_snapshots(conn, source, days=days)
    finally:
        conn.close()

    if df_service.empty:
        return {
            "source": source,
            "days": days,
            "service_time_factor_median": None,
            "service_time_factor_mean": None,
            "throughput_factor_median": None,
            "throughput_factor_mean": None,
            "throughput_factor_total": None,
            "service_time_factor_total": None,
            "model_service_min": None,
            "observed_service_mean": None,
            "observed_service_median": None,
            "throughput_summary": pd.DataFrame(),
        }

    service_vals = pd.to_numeric(df_service["service_min"], errors="coerce")
    service_vals = service_vals[(service_vals >= DWELL_MIN_FLOOR) & service_vals.notna()]
    if service_vals.empty:
        return {
            "source": source,
            "days": days,
            "service_time_factor_median": None,
            "service_time_factor_mean": None,
            "throughput_factor_median": None,
            "throughput_factor_mean": None,
            "throughput_factor_total": None,
            "service_time_factor_total": None,
            "model_service_min": None,
            "observed_service_mean": None,
            "observed_service_median": None,
            "throughput_summary": pd.DataFrame(),
        }

    observed_service_median = float(service_vals.median())
    observed_service_mean = float(service_vals.mean())
    preferred_min, _, _, _ = _preferred_service_minutes(df_service, None)
    model_service_min = float(min(max(preferred_min, DWELL_MIN_FLOOR), DWELL_MAX_CAP))

    service_time_factor_median = (
        observed_service_median / model_service_min
        if model_service_min and model_service_min > 0 else None
    )
    service_time_factor_mean = (
        observed_service_mean / model_service_min
        if model_service_min and model_service_min > 0 else None
    )

    throughput_df = pd.DataFrame()
    throughput_factor_median = None
    throughput_factor_mean = None
    throughput_factor_total = None
    service_time_factor_total = None

    if not df_snapshots.empty:
        snaps = df_snapshots.copy()
        snaps["timestamp"] = pd.to_datetime(snaps["timestamp"], errors="coerce")
        snaps["active_lanes"] = pd.to_numeric(snaps["active_lanes"], errors="coerce")
        snaps = snaps.dropna(subset=["timestamp", "active_lanes"])
        snaps = snaps[snaps["active_lanes"] > 0].copy()
        if not snaps.empty:
            snaps["bucket"] = snaps["timestamp"].dt.floor(f"{BUCKET_MINUTES}min")
            lanes = snaps.groupby("bucket")["active_lanes"].mean().reset_index()
            service_buckets = df_service.copy()
            service_buckets["timestamp"] = pd.to_datetime(service_buckets["timestamp"], errors="coerce")
            service_buckets = service_buckets.dropna(subset=["timestamp"])
            if not service_buckets.empty:
                service_buckets["bucket"] = service_buckets["timestamp"].dt.floor(f"{BUCKET_MINUTES}min")
                served = (
                    service_buckets.groupby("bucket")
                    .size()
                    .rename("served")
                    .reset_index()
                )
                merged = pd.merge(lanes, served, on="bucket", how="inner")
                merged["pred_capacity"] = (
                    merged["active_lanes"] * BUCKET_MINUTES / model_service_min
                )
                merged["throughput_factor"] = np.where(
                    merged["pred_capacity"] > 0,
                    merged["served"] / merged["pred_capacity"],
                    np.nan,
                )
                merged["service_time_factor"] = np.where(
                    merged["throughput_factor"] > 0,
                    1.0 / merged["throughput_factor"],
                    np.nan,
                )
                throughput_df = merged
                throughput_factor_median = float(throughput_df["throughput_factor"].median(skipna=True))
                throughput_factor_mean = float(throughput_df["throughput_factor"].mean(skipna=True))
                total_capacity = float(throughput_df["pred_capacity"].sum())
                total_served = float(throughput_df["served"].sum())
                throughput_factor_total = (
                    total_served / total_capacity if total_capacity > 0 else None
                )
                service_time_factor_total = (
                    1.0 / throughput_factor_total if throughput_factor_total and throughput_factor_total > 0 else None
                )

    return {
        "source": source,
        "days": days,
        "model_service_min": model_service_min,
        "observed_service_median": observed_service_median,
        "observed_service_mean": observed_service_mean,
        "service_time_factor_median": service_time_factor_median,
        "service_time_factor_mean": service_time_factor_mean,
        "throughput_factor_median": throughput_factor_median,
        "throughput_factor_mean": throughput_factor_mean,
        "throughput_factor_total": throughput_factor_total,
        "service_time_factor_total": service_time_factor_total,
        "throughput_summary": throughput_df,
    }


def _build_lane_schedule(
    df_lane_history: pd.DataFrame,
    fallback_lanes: int,
    max_lanes: int = 4,
    historical_checkout_fraction: "float | None" = None,
    checkout_fraction: "float | None" = None,
    checkout_service_min: float = 3.0,
) -> dict:
    """slot_label → integer lanes from historical observations.

    Priority (most → least reliable):
    Phase 1 — service_events WITH distinct lane_id (directly observed, most reliable).
    Phase 2 — queue_state_snapshots active_lanes (directly observed ground truth,
               time-of-day aware — same data Training-Period Wait Comparison uses).
    Phase 3 — demand-implied from entrance arrivals × checkout fraction
               (time-of-day aware but depends on checkout_fraction accuracy).
    Phase 4 — service_events throughput inference (noisy: sensitive to service_min).
    """
    if df_lane_history.empty:
        return {}
    effective_cf = historical_checkout_fraction if historical_checkout_fraction is not None else checkout_fraction
    capacity_per_30 = 30.0 / max(checkout_service_min, 0.5)
    schedule = {}
    for _, row in df_lane_history.iterrows():
        slot = row["slot_label"]
        observed = float(row.get("median_lanes", 0) or 0)
        lanes_source = str(row.get("lanes_source", "none"))

        # Phase 1 — distinct lane_id from service_events (directly observed)
        if observed > 0 and lanes_source == "lane_id":
            schedule[slot] = max(1, min(max_lanes, int(round(observed))))
            continue

        # Phase 2 — snapshot active_lanes (directly observed, time-of-day aware)
        snap = float(row.get("snapshot_lanes", 0) or 0)
        if snap > 0:
            schedule[slot] = max(1, min(max_lanes, int(round(snap))))
            continue

        # Phase 3 — demand-implied from entrance arrivals × checkout fraction
        if effective_cf is not None:
            entrance_per_30 = float(row.get("median_entrance_per_30min", 0) or 0)
            if entrance_per_30 > 0:
                checkout_per_30 = entrance_per_30 * effective_cf
                implied = checkout_per_30 / capacity_per_30
                schedule[slot] = max(1, min(max_lanes, int(np.ceil(implied))))
                continue

        # Phase 4 — throughput inference from service_events (last resort)
        if observed > 0:
            schedule[slot] = max(1, min(max_lanes, int(round(observed))))
    return schedule


def _forecast_lanes_vector(
    forecast_df: pd.DataFrame,
    lane_schedule: dict,
    fallback_lanes: int,
) -> list:
    """Map each forecast bucket timestamp to its lane count from the schedule."""
    result = []
    for ts in pd.to_datetime(forecast_df["ds"]):
        minute_slot = (ts.minute // 30) * 30
        slot_label = f"{ts.hour:02d}:{minute_slot:02d}"
        result.append(lane_schedule.get(slot_label, fallback_lanes))
    return result


def _observed_checkout_rate(df_service: pd.DataFrame, bucket_minutes: int) -> "float | None":
    """Compute median completed-checkout events per bucket from service_events.

    Filters to rows with total_dwell_sec > SERVICE_MIN_DWELL_SEC to exclude
    brief head-detector noise, then groups by bucket and returns the median
    count.

    Returns None when:
    - fewer than MIN_DEMAND_BUCKETS non-empty buckets exist (sparse data), or
    - qualifying events are < 20% of total events (dwell filter not meaningful
      for this dataset — e.g. camera records brief pass-by detections only).
    """
    if df_service.empty or "total_dwell_sec" not in df_service.columns:
        return None
    if "timestamp" not in df_service.columns and "ds" not in df_service.columns:
        return None

    ts_col = "timestamp" if "timestamp" in df_service.columns else "ds"
    dwell = pd.to_numeric(df_service["total_dwell_sec"], errors="coerce").fillna(0)
    qualifying_mask = dwell > SERVICE_MIN_DWELL_SEC

    # Quality gate: qualifying events must represent ≥ 20% of total data.
    # When < 20% pass the filter the dwell distribution doesn't separate real
    # checkouts from noise (e.g. camera records brief exit detections only).
    if qualifying_mask.sum() < len(df_service) * 0.20:
        return None

    svc = df_service.loc[qualifying_mask].copy()
    svc[ts_col] = pd.to_datetime(svc[ts_col])
    svc["_bucket"] = svc[ts_col].dt.floor(f"{bucket_minutes}min")
    counts = svc.groupby("_bucket").size()
    non_empty = counts[counts > 0]
    if len(non_empty) < MIN_DEMAND_BUCKETS:
        return None
    return float(non_empty.median())


def _build_checkout_rate_profile(df_service: pd.DataFrame, bucket_minutes: int) -> dict:
    """Return {(dayofweek, hour): median_checkouts_per_bucket} from service_events.

    Uses service_events timestamps (= checkout time, not entrance time), so the
    shopping lag is already embedded — no correction needed. Grouping by dow+hour
    captures time-of-day demand patterns directly from real checkout activity.

    Applies the same SERVICE_MIN_DWELL_SEC filter as _observed_checkout_rate to
    exclude brief pass-by detections that would inflate the per-bucket rate.
    """
    if df_service.empty or "timestamp" not in df_service.columns:
        return {}
    svc = df_service.copy()
    svc["timestamp"] = pd.to_datetime(svc["timestamp"], errors="coerce")
    svc = svc.dropna(subset=["timestamp"])
    svc["bucket"] = svc["timestamp"].dt.floor(f"{bucket_minutes}min")
    svc["dow"]    = svc["bucket"].dt.dayofweek
    svc["hour"]   = svc["bucket"].dt.hour
    counts = svc.groupby(["dow", "hour", "bucket"]).size().reset_index(name="n")
    return counts.groupby(["dow", "hour"])["n"].median().to_dict()


def _lookup_checkout_rate(profile: dict, ts: "pd.Timestamp", fallback: float) -> float:
    """Look up expected checkouts/bucket for a given timestamp from the profile."""
    key = (ts.dayofweek, ts.hour)
    if key in profile:
        return float(profile[key])
    # Hour-only fallback (any weekday)
    hour_vals = [v for (d, h), v in profile.items() if h == ts.hour]
    if hour_vals:
        return float(np.median(hour_vals))
    return fallback


def _infer_active_lanes(
    df_service_recent: pd.DataFrame,
    est_service_min: float,
    window_min: int,
    max_lanes: int = 4,
) -> "int | None":
    """Infer currently active lane count from recent service_events.

    If lane_id is populated, counts distinct lanes with qualifying events.
    Otherwise estimates from throughput rate: events / (window_min / service_min).
    Returns None when no qualifying events exist in the window.
    """
    if df_service_recent.empty:
        return None
    svc = df_service_recent.copy()
    svc = svc[pd.to_numeric(svc.get("total_dwell_sec", pd.Series(dtype=float)), errors="coerce").fillna(0) > SERVICE_MIN_DWELL_SEC]
    if svc.empty:
        return None

    if "lane_id" in svc.columns:
        valid_lanes = svc["lane_id"].dropna()
        if not valid_lanes.empty:
            return int(min(valid_lanes.nunique(), max_lanes))

    # Fall back to count-based inference
    capacity_per_window = window_min / max(est_service_min, 0.5)
    inferred = round(len(svc) / max(capacity_per_window, 1))
    return int(min(max(inferred, 1), max_lanes))


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
    df_arrivals         = load_arrivals(conn, source, days, data_since)
    df_snapshots        = load_snapshots(conn, source, days, data_since)
    df_dwell            = load_dwell_events(conn, days=days)
    df_service          = load_service_events(conn, days=days)
    df_pred_history     = load_prediction_history(conn, days=1)
    _checkout_svc_min = float(os.getenv("CHECKOUT_SERVICE_MIN", DEFAULT_DWELL_MIN))
    _MAX_LANES = 4

    # Use actual measured service time for capacity_per_30 in lane throughput inference.
    # The env-var default (3.0 min) is often wrong — measured time drives a 2-3× different
    # capacity_per_30, which directly multiplies into the inferred lane count.
    _svc_for_lanes, _, _, _ = _preferred_service_minutes(df_service, None)
    _svc_for_lanes = min(max(_svc_for_lanes, DWELL_MIN_FLOOR), DWELL_MAX_CAP)

    df_lane_history = load_lane_history_agg(
        conn, days=days,
        shop_open=SHOP_OPEN, shop_close=SHOP_CLOSE,  # integer bounds for slot iteration
        checkout_service_min=_svc_for_lanes,
        service_min_dwell_sec=SERVICE_MIN_DWELL_SEC,
    )
    conn.close()

    # Cap lane columns at the hard maximum so both the chart and _build_lane_schedule
    # see consistent values — throughput inference can exceed 4 before clamping.
    if not df_lane_history.empty:
        for _col in ("median_lanes", "snapshot_lanes"):
            if _col in df_lane_history.columns:
                df_lane_history[_col] = pd.to_numeric(
                    df_lane_history[_col], errors="coerce"
                ).clip(upper=_MAX_LANES).fillna(0.0)

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
    today_open = now_floor.replace(hour=SHOP_OPEN, minute=SHOP_OPEN_MINUTE, second=0, microsecond=0)
    # When the current time is before shop open, clamp end to today_open so we
    # always have at least one row for Prophet (empty DataFrames raise ValueError).
    insample_end = max(now_floor, today_open)
    today_ts = pd.date_range(start=today_open, end=insample_end, freq=f"{BUCKET_MINUTES}min")
    df_insample = model.predict(pd.DataFrame({"ds": today_ts}))

    future_ts = future_open_timestamps(interval_min=BUCKET_MINUTES)
    future_df = pd.DataFrame({"ds": future_ts})
    if future_df.empty:
        forecast = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
        forecast_wait = pd.DataFrame(columns=["ds", "yhat"])
    else:
        forecast = model.predict(future_df)
        forecast_wait = forecast[["ds", "yhat"]].copy()
        forecast_wait["yhat"] = forecast_wait["yhat"].clip(lower=0)
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
    # Strip closed-hour rows so no non-zero seasonal components leak outside shop hours.
    _comp_tot = comp_df["ds"].dt.hour * 60 + comp_df["ds"].dt.minute
    comp_df = comp_df[
        (_comp_tot >= SHOP_OPEN_TOT) & (_comp_tot < SHOP_CLOSE_TOT)
    ].reset_index(drop=True)
    for _col in ("yhat", "yhat_lower", "yhat_upper"):
        if _col in comp_df.columns:
            comp_df[_col] = comp_df[_col].clip(lower=0)

    history_end = max(df_combined["ds"].max(), pd.Timestamp.now().floor(f"{BUCKET_MINUTES}min"))
    history_ts  = pd.date_range(
        start=df_combined["ds"].min(),
        end=history_end,
        freq=f"{BUCKET_MINUTES}min",
    )
    _hist_tot = history_ts.hour * 60 + history_ts.minute
    history_ts = history_ts[(_hist_tot >= SHOP_OPEN_TOT) & (_hist_tot < SHOP_CLOSE_TOT)]
    history_fit = model.predict(pd.DataFrame({"ds": history_ts}))
    history_fit["yhat"]       = history_fit["yhat"].clip(lower=0)
    history_fit["yhat_lower"] = history_fit["yhat_lower"].clip(lower=0)
    history_fit["yhat_upper"] = history_fit["yhat_upper"].clip(lower=0)

    snap_latest = latest_snapshot_for_wait(df_snapshots, source)
    snap_age_min = None
    if snap_latest is not None and "timestamp" in snap_latest:
        snap_ts = pd.Timestamp(snap_latest["timestamp"])
        if snap_ts.tzinfo is not None:
            snap_ts = snap_ts.tz_localize(None)
        snap_age_min = (pd.Timestamp.now() - snap_ts).total_seconds() / 60

    snapshot_avg_dwell_min = None
    snapshot_valid = snap_latest is not None and (snap_age_min is None or snap_age_min < SNAPSHOT_MAX_AGE_MIN)
    _raw_queue = 0
    if snapshot_valid:
        _raw_queue    = int(snap_latest["queue_count"])
        active_lanes  = int(snap_latest["active_lanes"])
        current_queue = max(0, _raw_queue - active_lanes)
        avg_dwell_min = float(snap_latest["avg_dwell_sec"]) / 60
        snapshot_avg_dwell_min = avg_dwell_min
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

    preferred_service_min, service_time_source, service_median_min, service_mean_min = _preferred_service_minutes(
        df_service,
        snapshot_avg_dwell_min,
    )
    est_service_min  = min(max(preferred_service_min, DWELL_MIN_FLOOR), DWELL_MAX_CAP)
    browsing_est     = estimate_browsing_gap(df_dwell, df_service,
                                             max_lag_min=SHOP_TIME_MAX_LAG_MIN)

    # ── Apply browsing gap: entrance → checkout arrival shift ─────────────────
    # Customers entering the store at time T reach checkout at T + BROWSING_GAP_MIN.
    # The first `browsing_lag_steps` forecast buckets are served by people already
    # in the store (historical entrance counts); later buckets use the Prophet
    # arrival forecast shifted back by the gap.
    BROWSING_GAP_MIN   = int(os.getenv("BROWSING_GAP_MIN", 25))
    browsing_lag_steps = max(1, round(BROWSING_GAP_MIN / BUCKET_MINUTES))

    # Scale the prophet entrance-count forecast to checkout-arrival units.
    #
    # Primary: use observed checkout throughput from service_events (events per
    # bucket with dwell > SERVICE_MIN_DWELL_SEC).  This is lane-count-agnostic —
    # the rate reflects actual store demand regardless of how many lanes were open.
    avg_prophet_yhat = float(forecast_wait["yhat"].mean()) if not forecast_wait.empty else 0.0

    # Stable historical checkout fraction: (checkouts/bucket) / (arrivals/bucket).
    # Computed from the full training dataset so it doesn't fluctuate with the current
    # forecast window size or Prophet yhat magnitude.
    hist_arrival_per_bucket = (
        float(df_arrivals["entry_count"].mean())
        if not df_arrivals.empty and "entry_count" in df_arrivals.columns
        else 0.0
    )
    hist_checkout_rate = _observed_checkout_rate(df_service, BUCKET_MINUTES)
    if hist_checkout_rate is not None and hist_arrival_per_bucket > 0:
        historical_checkout_fraction: "float | None" = min(
            max(hist_checkout_rate / hist_arrival_per_bucket, 0.01), 1.5
        )
    else:
        historical_checkout_fraction = None

    # Capacity-based demand: how many customers the historical lane count can serve per bucket.
    # Always computed — used as a floor when service_events data gives an unrealistically
    # low checkout rate (camera noise, sparse data).
    if not df_snapshots.empty and "active_lanes" in df_snapshots.columns:
        snap_lanes_series = pd.to_numeric(df_snapshots["active_lanes"], errors="coerce").dropna()
        median_hist_lanes = float(snap_lanes_series.median()) if not snap_lanes_series.empty else float(active_lanes)
    else:
        median_hist_lanes = float(active_lanes)
    _capacity_demand = median_hist_lanes * BUCKET_MINUTES / max(est_service_min, 0.5)

    # checkout_fraction: scales Prophet entrance-arrival forecast → checkout-arrival forecast.
    # Priority:
    #   1. historical_checkout_fraction — stable ratio, immune to yhat denominator collapse
    #   2. demand / mean(yhat) — volatile fallback, can hit 4.0 cap when mean(yhat) is small
    if historical_checkout_fraction is not None:
        checkout_fraction = historical_checkout_fraction
        demand_per_bucket = hist_checkout_rate if hist_checkout_rate is not None else 0.0
        demand_source = "service_events"
    elif hist_checkout_rate is not None:
        demand_per_bucket = hist_checkout_rate
        demand_source = "service_events"
        checkout_fraction = (
            min(max(demand_per_bucket / avg_prophet_yhat, 0.01), 1.5)
            if avg_prophet_yhat > 0 else 1.0
        )
    else:
        demand_per_bucket = _capacity_demand
        demand_source = "snapshot_fallback"
        checkout_fraction = (
            min(max(demand_per_bucket / avg_prophet_yhat, 0.01), 1.5)
            if avg_prophet_yhat > 0 else 1.0
        )

    # Floor: if service_events demand is below 30% of lane capacity, the exit camera
    # data is too noisy/sparse to trust — fall back to capacity-based estimate.
    if demand_per_bucket < _capacity_demand * 0.3:
        demand_per_bucket = _capacity_demand
        demand_source = demand_source + "+capacity_floor"

    # Infer currently active lanes from recent service_events (last SNAPSHOT_MAX_AGE_MIN).
    # Used for in-service subtraction and the t=now anchor — keeps active_lanes in the
    # result dict unchanged for dashboard compatibility.
    cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=SNAPSHOT_MAX_AGE_MIN)
    if "timestamp" in df_service.columns and not df_service.empty:
        df_service_recent = df_service[df_service["timestamp"] >= cutoff]
    else:
        df_service_recent = pd.DataFrame()
    inferred_lanes = _infer_active_lanes(df_service_recent, est_service_min, SNAPSHOT_MAX_AGE_MIN)
    # Apply inferred lanes to in-service subtraction only when the snapshot was valid.
    # When stale, current_queue was already derived from entrance events (no subtraction needed).
    snap_lanes_for_formula = inferred_lanes if inferred_lanes is not None else active_lanes
    if snapshot_valid:
        current_queue = max(0, int(snap_latest["queue_count"]) - snap_lanes_for_formula)

    checkout_rate_profile = _build_checkout_rate_profile(df_service, BUCKET_MINUTES)

    service_profile = build_service_time_profile(df_service)

    # Build per-bucket lane schedule first — needed to anchor the demand floor to the
    # actual lanes the simulation will use, not just the snapshot lane count.
    # Strip any lane history rows outside shop hours (defence against stale DB data
    # or module-cache returning old SHOP_OPEN/SHOP_CLOSE values).
    if not df_lane_history.empty and "slot_label" in df_lane_history.columns:
        df_lane_history = df_lane_history[
            df_lane_history["slot_label"].apply(
                lambda s: SHOP_OPEN_TOT <= int(s.split(":")[0]) * 60 + int(s.split(":")[1]) < SHOP_CLOSE_TOT
            )
        ].copy()

    # Build per-bucket lane schedule from historical demand.
    # Priority: Phase 1 (service_events) → Phase 3 (demand-implied) → Phase 2 (snapshots).
    # historical_checkout_fraction is preferred (stable); checkout_fraction is secondary fallback
    # so Phase 3 still runs when historical data is too sparse for _observed_checkout_rate.
    lane_schedule = _build_lane_schedule(
        df_lane_history,
        fallback_lanes=active_lanes,
        historical_checkout_fraction=historical_checkout_fraction,
        checkout_fraction=checkout_fraction,
        checkout_service_min=_checkout_svc_min,
    )
    forecast_wait = forecast_wait.copy()
    per_bucket_lanes = _forecast_lanes_vector(forecast_wait, lane_schedule, fallback_lanes=active_lanes)

    # ── Demand stack for the queue simulation ─────────────────────────────────
    # All sources must be in CHECKOUT arrivals per bucket — NOT entrance arrivals.
    # Entrance counts have a 15-45 min shopping lag before they become checkout demand,
    # so using them directly inflates demand by 5-15×. ensemble_predict.py handles this
    # via a BROWSING_GAP_MIN shift; here we use sources that already have lag baked in.

    # Floor — anchored to average forecast lane throughput × 1.1 so the simulation
    # always shows slight queue growth at the floor level regardless of lane schedule.
    _avg_forecast_lanes = float(np.mean(per_bucket_lanes)) if per_bucket_lanes else float(active_lanes)
    _capacity_floor = _avg_forecast_lanes * BUCKET_MINUTES / max(est_service_min, 0.5) * 1.1

    _demand_source_label = "capacity_floor"
    _demand = _capacity_floor

    # Source 1 — hist_checkout_rate (median completions/bucket from service_events).
    # Lag is already baked in: events are recorded when people LEAVE checkout, not when
    # they enter the store. Most reliable when service data is sufficient.
    if hist_checkout_rate is not None and hist_checkout_rate > 0:
        _demand = float(hist_checkout_rate)
        _demand_source_label = "hist_checkout_rate"

    # Source 2 — Prophet yhat × historical_checkout_fraction (time-of-day aware).
    # Converts entrance forecast to checkout scale using the observed ratio.
    # Used when service_events are too sparse for hist_checkout_rate.
    elif avg_prophet_yhat > 0:
        _cf_for_demand = historical_checkout_fraction if historical_checkout_fraction is not None else checkout_fraction
        _demand = float(avg_prophet_yhat) * float(_cf_for_demand)
        _demand_source_label = "prophet_x_checkout_fraction"

    # Apply capacity floor unconditionally
    _demand = max(_demand, _capacity_floor)
    demand_source = _demand_source_label

    forecast_wait["yhat"] = _demand

    per_bucket_service = (
        [service_time_for_timestamp(service_profile, row.ds, fallback=est_service_min)
         for row in forecast_wait.itertuples()]
        if service_profile else est_service_min
    )

    # Clip forecast to the next FORECAST_HORIZON_MINUTES for the wait model.
    # future_open_timestamps goes until shop close; the wait queue simulation becomes
    # unreliable beyond ~60 min so we cap it here. The full forecast_wait stays in
    # the result dict for the queue-evolution diagnostic chart.
    _horizon_end = pd.Timestamp.now() + pd.Timedelta(minutes=FORECAST_HORIZON_MINUTES)
    _fw_horizon = forecast_wait[forecast_wait["ds"] <= _horizon_end].copy()
    _n = len(_fw_horizon)
    _pb_lanes_horizon  = per_bucket_lanes[:_n]
    _pb_service_horizon = (
        per_bucket_service[:_n] if isinstance(per_bucket_service, list) else per_bucket_service
    )

    # Wait estimates use per_bucket_lanes — the same lane counts shown in the Active Lanes
    # chart — so the wait forecast is always consistent with the estimated lane schedule.
    wait_estimates, wait_15m, wait_30m, _service_per_bucket = compute_wait_estimates(
        _fw_horizon,
        current_queue=current_queue,
        avg_dwell_min=_pb_service_horizon,
        active_lanes=_pb_lanes_horizon,
        max_queue_per_lane=MAX_QUEUE_PER_LANE,
        max_wait_min=MAX_WAIT_MIN,
    )

    # Prepend a t=now anchor. Use the first scheduled lane count so the anchor is
    # consistent with the per_bucket_lanes that drive the rest of the forecast.
    _anchor_lanes = _pb_lanes_horizon[0] if _pb_lanes_horizon else snap_lanes_for_formula
    _snapshot_wait_now = round(
        min(current_queue * est_service_min / max(_anchor_lanes, 1), MAX_WAIT_MIN), 1
    )
    wait_estimates = [
        {"ds": pd.Timestamp.now().floor(f"{BUCKET_MINUTES}min"), "wait_min": _snapshot_wait_now}
    ] + wait_estimates

    # ── Multi-lane scenarios (1–4 lanes) ──────────────────────────────────────
    # For each scenario the starting queue depth depends on n: with more lanes open,
    # more people are in service → fewer people waiting.
    # Use _raw_queue (total head count) and subtract n so each scenario starts correctly.
    _snap_raw = _raw_queue if snapshot_valid else current_queue
    lane_scenarios: dict[int, dict] = {}
    for n in range(1, 5):
        _scenario_queue = max(0, _snap_raw - n)
        w_rows, w15, w30, _ = compute_wait_estimates(
            _fw_horizon.copy(),
            current_queue=_scenario_queue,
            avg_dwell_min=_pb_service_horizon,
            active_lanes=n,
            max_queue_per_lane=MAX_QUEUE_PER_LANE,
            max_wait_min=MAX_WAIT_MIN,
        )
        _lane_snap_wait = round(
            min(_scenario_queue * est_service_min / max(n, 1), MAX_WAIT_MIN), 1
        )
        w_rows = [{"ds": pd.Timestamp.now().floor(f"{BUCKET_MINUTES}min"), "wait_min": _lane_snap_wait}] + w_rows
        lane_scenarios[n] = {
            "wait_estimates": w_rows,
            "wait_15m": w15,
            "wait_30m": w30,
        }

    result = {
        "df_arrivals":     df_arrivals,
        "df_snapshots":    df_snapshots,
        "df_dwell":        df_dwell,
        "df_service":      df_service,
        "browsing_est":    browsing_est,
        "df_combined":     df_combined,
        "df_insample":     df_insample,
        "forecast":        forecast,
        "comp_df":         comp_df,
        "history_fit":     history_fit,
        "wait_estimates":  wait_estimates,
        "wait_15m":        wait_15m,
        "wait_30m":        wait_30m,
        "current_queue":   current_queue,
        "active_lanes":    active_lanes,
        "avg_dwell_min":   avg_dwell_min,
        "snapshot_avg_dwell_min": snapshot_avg_dwell_min,
        "est_service_min": est_service_min,
        "per_bucket_service": per_bucket_service,
        "service_time_source": service_time_source,
        "service_median_min": service_median_min,
        "service_mean_min": service_mean_min,
        "checkout_service_min": _checkout_svc_min,
        "df_lane_history": df_lane_history,
        "lane_scenarios":  lane_scenarios,
        "real_span_days":  real_span,
        "used_sim_history": used_sim,
        "source":          source,
        "days":            days,
        "checkout_fraction": checkout_fraction,
        "demand_source":   _demand_source_label,
        "demand_per_bucket": demand_per_bucket,
        "avg_prophet_yhat":  avg_prophet_yhat,
        "hist_checkout_rate": hist_checkout_rate,
        "snap_lanes_formula": snap_lanes_for_formula,
        "forecast_wait":   forecast_wait,
        "checkout_rate_profile": checkout_rate_profile,
        "inferred_lanes":  inferred_lanes,
        "lane_schedule":   lane_schedule,
        "per_bucket_lanes": per_bucket_lanes,
        "historical_checkout_fraction": historical_checkout_fraction,
        "df_pred_history": df_pred_history,
        "_demand":          _demand,
        "_capacity_floor":  _capacity_floor,
        "_avg_forecast_lanes": _avg_forecast_lanes,
    }

    save_prediction(
        wait_15m=float(result["wait_15m"]) if result["wait_15m"] is not None else 0.0,
        wait_30m=float(result["wait_30m"]) if result["wait_30m"] is not None else 0.0,
        prophet_yhat=avg_prophet_yhat,
    )
    return result
