"""
core.py — Shared constants, utilities, and wait-time model for the prediction module.

All tuneable parameters are loaded from environment variables (or .env file) so
they can be overridden without touching the code.  Defaults are chosen for a
typical retail shop open 08:30–20:30 with 3-minute arrival buckets.

Environment variables
─────────────────────
  SHOP_OPEN                 int   Opening hour   (default 8,  i.e. 08:xx)
  SHOP_OPEN_MINUTE          int   Opening minute (default 30, i.e. xx:30)
  SHOP_CLOSE_HOUR           int   Closing hour   (default 20, i.e. 20:xx)
  SHOP_CLOSE_MINUTE         int   Closing minute (default 30, i.e. xx:30)
  BUCKET_MINUTES            int   Granularity of arrival buckets in minutes (default 3)
  FORECAST_HORIZON_MINUTES  int   How far ahead to forecast (default 60)
  LOOKBACK_MINUTES          int   Lookback window used in sequence models (default 60)
  ROLLING_WINDOW_MINUTES    int   Rolling-average window length in minutes (default 30)
  MIN_REAL_DAYS_FOR_SIM     int   Minimum real-data span before skipping bootstrap (default 14)
  MAX_QUEUE_PER_LANE        int   Hard cap on queue depth per lane (default 20)
  MAX_WAIT_MIN              float Hard cap on estimated wait in minutes (default 60)
  DEFAULT_DWELL_MIN         float Fallback service time when no snapshot available (default 3.0)
  DWELL_MIN_FLOOR           float Minimum dwell used in wait calculation (default 0.5)
  DWELL_MAX_CAP             float Maximum dwell used in wait calculation (default 10.0)
  DEFAULT_LANES             int   Fallback active-lane count (default 2)
  SNAPSHOT_MAX_AGE_MIN      int   Max snapshot age before it is considered stale (default 30)
  SHOP_TIME_MAX_LAG_MIN     int   Max lag tested in browsing-gap cross-correlation (default 120)
  DATA_SINCE_DEFAULT_DAYS   int   Default lookback when data_since is not specified (default 7)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# ── Shop hours ────────────────────────────────────────────────────────────────
SHOP_OPEN        = int(os.getenv("SHOP_OPEN",         8))   # opening hour (integer, for slot iteration)
SHOP_OPEN_MINUTE = int(os.getenv("SHOP_OPEN_MINUTE",  30))  # opening minute
SHOP_CLOSE_HOUR  = int(os.getenv("SHOP_CLOSE_HOUR",   20))  # closing hour
SHOP_CLOSE_MINUTE = int(os.getenv("SHOP_CLOSE_MINUTE", 30))  # closing minute
SHOP_CLOSE = SHOP_CLOSE_HOUR + 1                             # exclusive upper bound for slot-building loops
# Actual open/close as total minutes from midnight — used for all time comparisons
SHOP_OPEN_TOT  = SHOP_OPEN * 60 + SHOP_OPEN_MINUTE          # 8*60+30 = 510
SHOP_CLOSE_TOT = SHOP_CLOSE_HOUR * 60 + SHOP_CLOSE_MINUTE   # 20*60+30 = 1230

# ── Per-day schedule overrides ────────────────────────────────────────────────
# JSON keyed by weekday number (0=Mon … 6=Sun). Only specified keys/fields are
# overridden; all others inherit the defaults above.
# Example .env:  SHOP_SCHEDULE_OVERRIDE={"6": {"close_hour": 13, "close_minute": 0}}
_SCHEDULE_OVERRIDE: dict = json.loads(os.getenv("SHOP_SCHEDULE_OVERRIDE", "{}"))


def _day_hours(weekday: int) -> tuple[int, int, int, int]:
    """Return (open_hour, open_minute, close_hour, close_minute) for *weekday* (0=Mon, 6=Sun)."""
    ov = _SCHEDULE_OVERRIDE.get(str(weekday), {})
    return (
        int(ov.get("open_hour",    SHOP_OPEN)),
        int(ov.get("open_minute",  SHOP_OPEN_MINUTE)),
        int(ov.get("close_hour",   SHOP_CLOSE_HOUR)),
        int(ov.get("close_minute", SHOP_CLOSE_MINUTE)),
    )

# ── Prediction buckets & horizon ──────────────────────────────────────────────
BUCKET_MINUTES            = int(os.getenv("BUCKET_MINUTES",            3))
FORECAST_HORIZON_MINUTES  = int(os.getenv("FORECAST_HORIZON_MINUTES",  60))
LOOKBACK_MINUTES          = int(os.getenv("LOOKBACK_MINUTES",          60))
ROLLING_WINDOW_MINUTES    = int(os.getenv("ROLLING_WINDOW_MINUTES",    30))
MIN_REAL_DAYS_FOR_SIM     = int(os.getenv("MIN_REAL_DAYS_FOR_SIM",     14))

# ── Wait model ────────────────────────────────────────────────────────────────
MAX_QUEUE_PER_LANE   = int(os.getenv("MAX_QUEUE_PER_LANE",   20))
MAX_WAIT_MIN         = float(os.getenv("MAX_WAIT_MIN",       60.0))
DEFAULT_DWELL_MIN    = float(os.getenv("DEFAULT_DWELL_MIN",  3.0))
DWELL_MIN_FLOOR      = float(os.getenv("DWELL_MIN_FLOOR",    0.5))
DWELL_MAX_CAP        = float(os.getenv("DWELL_MAX_CAP",      10.0))
DEFAULT_LANES        = int(os.getenv("DEFAULT_LANES",        2))
SNAPSHOT_MAX_AGE_MIN = int(os.getenv("SNAPSHOT_MAX_AGE_MIN", 30))
SHOP_TIME_MAX_LAG_MIN = int(os.getenv("SHOP_TIME_MAX_LAG_MIN", 120))
# Minimum dwell in service_events to count as a real completed checkout (filters noise)
SERVICE_MIN_DWELL_SEC = int(os.getenv("SERVICE_MIN_DWELL_SEC", 60))
# Minimum number of non-empty buckets required to use observed-rate calibration
MIN_DEMAND_BUCKETS = int(os.getenv("MIN_DEMAND_BUCKETS", 10))
# Camera IDs — entrance camera counts arrivals; exit camera is the checkout head counter
ENTRANCE_CAMERA_ID = os.getenv("ENTRANCE_CAMERA_ID", "Bosch_Camera_Entrance")
EXIT_CAMERA_ID     = os.getenv("EXIT_CAMERA_ID",     "Bosch_Camera_exit")

# ── Training defaults ─────────────────────────────────────────────────────────
DATA_SINCE_DEFAULT_DAYS = int(os.getenv("DATA_SINCE_DEFAULT_DAYS", 7))

# ── Derived constants (depend on BUCKET_MINUTES) ──────────────────────────────
FORECAST_STEPS        = FORECAST_HORIZON_MINUTES // BUCKET_MINUTES
LAG_HOUR_STEPS        = 60 // BUCKET_MINUTES
ROLLING_WINDOW_STEPS  = ROLLING_WINDOW_MINUTES // BUCKET_MINUTES
SEQUENCE_LEN          = LOOKBACK_MINUTES // BUCKET_MINUTES
WAIT_15M_INDEX        = max(0, (15 // BUCKET_MINUTES) - 1)
WAIT_30M_INDEX        = max(0, (30 // BUCKET_MINUTES) - 1)
WAIT_45M_INDEX        = max(0, (45 // BUCKET_MINUTES) - 1)


def get_where_clause(source: str) -> str:
    """Return a SQL WHERE clause that filters rows by data source.

    Parameters
    ----------
    source : str
        "REAL"  — exclude simulator cameras (camera_id NOT LIKE 'SIM_%')
        "SIM"   — include only simulator cameras (camera_id LIKE 'SIM_%')
        "ALL"   — no filter (empty string returned)

    Returns
    -------
    str
        A SQL fragment starting with "WHERE …", or "" for ALL.
    """
    if source == "REAL":
        return "WHERE camera_id NOT LIKE 'SIM_%'"
    if source == "SIM":
        return "WHERE camera_id LIKE 'SIM_%'"
    return ""


def is_open(ts: pd.Timestamp) -> bool:
    """Return True if the timestamp falls within shop opening hours (minute-accurate, per-day schedule)."""
    oh, om, ch, cm = _day_hours(ts.weekday())
    tot = ts.hour * 60 + ts.minute
    return (oh * 60 + om) <= tot < (ch * 60 + cm)


def future_open_timestamps(interval_min: int = BUCKET_MINUTES) -> list[pd.Timestamp]:
    """Generate a list of future timestamps during shop opening hours (minute-accurate).

    Starting from the next bucket after now, steps forward by `interval_min`
    until the shop closes today (or opens tomorrow if currently outside hours).
    """
    now = pd.Timestamp.now().floor(f"{interval_min}min")
    now_tot = now.hour * 60 + now.minute
    today = now.normalize()
    oh, om, ch, cm = _day_hours(today.weekday())
    open_tot_today  = oh * 60 + om
    close_tot_today = ch * 60 + cm

    open_ts  = today.replace(hour=oh, minute=om, second=0, microsecond=0)
    close_ts = today.replace(hour=ch, minute=cm, second=0, microsecond=0)

    if now_tot >= close_tot_today:
        tomorrow = today + pd.Timedelta(days=1)
        oh2, om2, ch2, cm2 = _day_hours(tomorrow.weekday())
        open_ts  = tomorrow.replace(hour=oh2, minute=om2, second=0, microsecond=0)
        close_ts = tomorrow.replace(hour=ch2, minute=cm2, second=0, microsecond=0)
    elif now_tot >= open_tot_today:
        open_ts = now + pd.Timedelta(minutes=interval_min)

    result: list[pd.Timestamp] = []
    t = open_ts
    while t < close_ts:
        result.append(t)
        t += pd.Timedelta(minutes=interval_min)
    return result


def add_closed_zeros(
    df: pd.DataFrame,
    days: int,
    interval_min: int = BUCKET_MINUTES,
) -> pd.DataFrame:
    """Fill Prophet training data with zero-arrival rows during closed hours.

    Prophet requires a continuous time series.  Without closed-hour rows the
    model treats gaps as missing data and may fit spurious trends.  This
    function appends zero-count rows for every bucket outside opening hours
    over the last `days` days, then deduplicates and sorts the result.

    Parameters
    ----------
    df           : pd.DataFrame  Training DataFrame with columns ["ds", "y"].
    days         : int           Number of past days to fill.
    interval_min : int           Bucket granularity in minutes.

    Returns
    -------
    pd.DataFrame
        Original rows + closed-hour zeros, sorted by "ds", duplicates removed.
    """
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = pd.Timestamp.now()
    rows = []
    t = start
    while t <= end:
        if not is_open(t):
            rows.append({"ds": t, "y": 0})
        t += pd.Timedelta(minutes=interval_min)
    if not rows:
        return df
    zeros = pd.DataFrame(rows)
    return (
        pd.concat([df, zeros])
        .drop_duplicates(subset="ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )


def build_sim_history(
    days: int = 7,
    interval_min: int = BUCKET_MINUTES,
) -> pd.DataFrame:
    """Generate synthetic arrival history using a Poisson process.

    Used to bootstrap Prophet training when fewer than MIN_REAL_DAYS_FOR_SIM
    days of real data are available.  Arrival rates vary by time-of-day and
    weekend vs. weekday to produce a plausible baseline pattern.

    Parameters
    ----------
    days         : int  Number of synthetic days to generate (default 7).
    interval_min : int  Bucket granularity in minutes.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["ds", "y"] covering `days` synthetic days.
    """
    rows = []
    start = datetime.now() - timedelta(days=days)
    for day in range(days):
        current_day = start + timedelta(days=day)
        is_weekend = current_day.weekday() >= 5
        _doh, _dom, _dch, _dcm = _day_hours(current_day.weekday())
        for hour in range(_doh, _dch + 1):
            for minute in range(0, 60, interval_min):
                t = current_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if not is_open(pd.Timestamp(t)):
                    continue
                if 8 <= hour < 10:
                    base = 3 if is_weekend else 2
                elif 10 <= hour < 13:
                    base = 8 if is_weekend else 5
                elif 13 <= hour < 15:
                    base = 6 if is_weekend else 7
                elif 15 <= hour < 18:
                    base = 5 if is_weekend else 4
                elif 18 <= hour < 20:
                    base = 4 if is_weekend else 8
                else:
                    base = 2
                rows.append({"ds": t, "y": max(0, int(np.random.poisson(base)))})
    return pd.DataFrame(rows)


def compute_wait_estimates(
    forecast_df: pd.DataFrame,
    current_queue: float,
    avg_dwell_min: "float | list[float]",
    active_lanes: "int | list[int]",
    *,
    arrivals_col: str = "yhat",
    bucket_minutes: int = BUCKET_MINUTES,
    max_queue_per_lane: int | None = None,
    max_wait_min: float | None = None,
) -> tuple[list[dict[str, object]], float | None, float | None, float]:
    """Simulate queue evolution to estimate customer wait times.

    For each future bucket the model:
      1. Adds the forecasted arrivals to the running queue.
      2. Subtracts the service capacity (lanes × bucket_min / dwell_per_step[i]).
      3. Clamps the queue to [0, max_queue_per_lane × lanes].
      4. Converts remaining queue depth to a wait-time estimate.

    Parameters
    ----------
    forecast_df        : pd.DataFrame            Must contain columns "ds" and `arrivals_col`.
    current_queue      : float                   Queue depth right now (people waiting).
    avg_dwell_min      : float | list[float]     Scalar OR per-bucket predicted service time (min).
    active_lanes       : int | list[int]         Scalar OR per-bucket lane count. When a list is
                                                  provided each bucket uses its own lane count,
                                                  enabling time-of-day lane schedules.
    arrivals_col       : str                     Column name for forecast values (default "yhat").
    bucket_minutes     : int                     Bucket size in minutes (default BUCKET_MINUTES).
    max_queue_per_lane : int | None              Per-lane queue cap; None = uncapped.
    max_wait_min       : float | None            Maximum wait cap in minutes; None = uncapped.

    Returns
    -------
    wait_estimates     : list[dict]   One dict per bucket with keys "ds" and "wait_min".
    wait_15m           : float | None Estimated wait at the +15-minute mark.
    wait_30m           : float | None Estimated wait at the +30-minute mark.
    mean_service_per_bucket : float   Mean throughput across all buckets.
    """
    n_steps = len(forecast_df)
    if n_steps == 0:
        return [], None, None, 0.0

    if isinstance(avg_dwell_min, (int, float)):
        dwell_per_step = [max(float(avg_dwell_min), 0.5)] * n_steps
    else:
        dwell_per_step = [max(float(d), 0.5) for d in avg_dwell_min]
        _dwell_fill = dwell_per_step[-1] if dwell_per_step else 1.0
        if len(dwell_per_step) < n_steps:
            dwell_per_step += [_dwell_fill] * (n_steps - len(dwell_per_step))

    if isinstance(active_lanes, (int, float)):
        lanes_per_step = [max(1, int(active_lanes))] * n_steps
    else:
        lanes_per_step = [max(1, int(l)) for l in active_lanes]
        _lanes_fill = lanes_per_step[-1] if lanes_per_step else 1
        if len(lanes_per_step) < n_steps:
            lanes_per_step += [_lanes_fill] * (n_steps - len(lanes_per_step))

    running_queue = max(0.0, float(current_queue))
    if max_queue_per_lane is not None:
        running_queue = min(running_queue, float(lanes_per_step[0] * max_queue_per_lane))

    wait_estimates: list[dict[str, object]] = []
    wait_15m = None
    wait_30m = None
    total_service_per_bucket = 0.0

    for step_i, row in enumerate(forecast_df.itertuples(index=False)):
        lanes_now = lanes_per_step[step_i]
        service_per_bucket = lanes_now * (bucket_minutes / dwell_per_step[step_i])
        total_service_per_bucket += service_per_bucket

        arrivals      = max(0.0, float(getattr(row, arrivals_col)))
        running_queue = max(0.0, running_queue + arrivals - service_per_bucket)
        if max_queue_per_lane is not None:
            running_queue = min(running_queue, float(lanes_now * max_queue_per_lane))

        wait_min = (
            (running_queue / service_per_bucket) * bucket_minutes
            if service_per_bucket > 0 else 0.0
        )
        if max_wait_min is not None:
            wait_min = min(wait_min, max_wait_min)
        wait_min = round(float(wait_min), 2)

        wait_estimates.append({"ds": getattr(row, "ds"), "wait_min": wait_min})
        if step_i == WAIT_15M_INDEX:
            wait_15m = wait_min
        if step_i == WAIT_30M_INDEX:
            wait_30m = wait_min

    mean_service_per_bucket = total_service_per_bucket / max(n_steps, 1)
    return wait_estimates, wait_15m, wait_30m, mean_service_per_bucket
