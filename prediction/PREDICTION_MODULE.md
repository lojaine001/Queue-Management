# Prediction Module — Documentation

Queue wait-time forecasting using Facebook Prophet, driven by real camera data
and/or simulator data stored in TimescaleDB (PostgreSQL).

---

## Overview

The prediction module produces short-term (60-minute) forecasts of customer
arrival rates and translates them into estimated queue wait times. It is used
by the Streamlit dashboard, REST APIs, and the command-line interface.

---

## File Structure

```
prediction/
├── core.py         Constants, utility functions, and the wait-time model
├── pipeline.py     Full ETL pipeline (load → train → forecast → wait estimates)
├── hybrid_wait.py  Separate REAL-only hybrid wait projection layer
├── quick.py        Lightweight fast forecast for dashboard use
├── cli.py          Command-line interface (3 subcommands)
├── __init__.py     Package init
├── __main__.py     Entry point for python -m prediction
└── PREDICTION_MODULE.md  This file
```

---

## End-to-End Data Flow

```
PostgreSQL / TimescaleDB
    │
    ├─ entrance_events          → bucketed arrival counts, dwell times, lane depth
    ├─ queue_state_snapshots    → current queue depth, avg dwell, active lanes
    └─ service_events           → checkout counter dwell times
    │
    ▼
pipeline.py / quick.py
    │
    ├─ 1. Load data (30 days by default)
    ├─ 2. Aggregate into BUCKET_MINUTES (3-min) buckets
    ├─ 3. Bootstrap with synthetic history if real span < 14 days
    ├─ 4. Cap outliers at 99th percentile
    ├─ 5. Fill closed-hour gaps with zeros
    ├─ 6. Train Prophet (daily + weekly seasonality)
    ├─ 7. Forecast next 60 open minutes
    ├─ 8. Derive current waiting backlog from latest snapshot
    ├─ 9. Prefer measured checkout service time from service_events
    ├─ 10. Rescale entrance forecast into checkout-arrival demand
    └─ 11. Compute wait estimates bucket-by-bucket
    │
    ▼
Output dict (forecast DataFrame + wait times + lane scenarios)
```

---

## Module Files

### `core.py`

Configuration hub and shared utility functions.

**Constants** (all overridable via environment variables or `.env`):

| Constant | Default | Description |
|----------|---------|-------------|
| `SHOP_OPEN` | 8 | Default opening hour (all days) |
| `SHOP_CLOSE` | 21 | Default closing hour (all days) |
| `SHOP_SCHEDULE_OVERRIDE` | `{}` | JSON dict of per-day overrides; keyed by weekday string `"0"`=Mon…`"6"`=Sun |
| `BUCKET_MINUTES` | 3 | Arrival bucket granularity in minutes |
| `FORECAST_HORIZON_MINUTES` | 60 | How far ahead to forecast |
| `MIN_REAL_DAYS_FOR_SIM` | 14 | Min real-data span before skipping bootstrap |
| `MAX_QUEUE_PER_LANE` | 20 | Hard cap on queue depth per lane |
| `MAX_WAIT_MIN` | 60.0 | Hard cap on estimated wait time (minutes) |
| `DEFAULT_DWELL_MIN` | 3.0 | Fallback service time when no snapshot available |
| `DEFAULT_LANES` | 2 | Fallback active-lane count |
| `SNAPSHOT_MAX_AGE_MIN` | 30 | Max snapshot age before considered stale |

**Key functions**:

- `get_where_clause(source)` — Returns SQL WHERE clause filtering by REAL/SIM/ALL
- `_day_hours(weekday)` — Returns `(open_hour, open_minute, close_hour, close_minute)` for a given weekday (0=Mon…6=Sun), applying any `SHOP_SCHEDULE_OVERRIDE` entry
- `is_open(ts)` — Returns True if timestamp falls within the shop's open window for that day (uses `_day_hours`)
- `future_open_timestamps()` — List of future 3-min bucket timestamps during the current or next operating window (uses `_day_hours` to handle per-day close times)
- `add_closed_zeros(df, days)` — Fills Prophet training data with zeros outside shop hours
- `build_sim_history(days=7)` — Generates synthetic Poisson-distributed arrival history for bootstrapping
- `compute_wait_estimates(forecast_df, current_queue, avg_dwell_min, active_lanes)` — Core wait-time simulation; returns `([], None, None, 0.0)` immediately when `forecast_df` is empty; pads `dwell_per_step` and `lanes_per_step` lists to match forecast length with safe fallback values

---

### `pipeline.py`

Full production pipeline. Used by the Streamlit dashboard and the `pipeline` CLI subcommand.

**Database loaders**:

| Function | Table | Returns |
|----------|-------|---------|
| `load_arrivals(conn, source, days)` | `entrance_events` | Bucketed entry counts with demographics |
| `load_snapshots(conn, source, days)` | `queue_state_snapshots` | Queue depth + dwell + lanes over time |
| `load_dwell_events(conn, days=1)` | `entrance_events` | Raw per-track entrance dwell times (REAL only) |
| `load_service_events(conn, days=30)` | `service_events` | Checkout dwell times (REAL only) |
| `load_today_actuals(source)` | `entrance_events` | Today's live arrival counts (opens own connection) |

**Analysis**:

- `estimate_browsing_gap(df_dwell, df_service)` — Cross-correlates entrance exits vs service arrivals to estimate unmeasured browsing time between queue and checkout
- `latest_snapshot_for_wait(df_snapshots, source)` — Returns most recent snapshot, preferring REAL cameras
- `_preferred_service_minutes(df_service, snapshot_dwell_min)` — Picks the best service-time basis for the wait model
- `_observed_checkout_rate(df_service, bucket_minutes)` — Estimates checkout throughput from `service_events`
- `_infer_active_lanes(df_service_recent, est_service_min, window_min)` — Estimates live lane count from recent service activity when possible

**Main function**:

```python
result = run_prediction_pipeline(
    source="REAL",     # "REAL" | "SIM" | "ALL"
    days=30,           # lookback window
    use_bootstrap=False,  # mix synthetic history if data is sparse
    data_since=None,   # explicit start datetime (overrides days)
)
```

**Return dict keys**:

| Key | Type | Description |
|-----|------|-------------|
| `forecast` | DataFrame | Prophet forecast (ds, yhat, yhat_lower, yhat_upper) |
| `wait_15m` | float\|None | Estimated wait at +15 minutes |
| `wait_30m` | float\|None | Estimated wait at +30 minutes |
| `wait_estimates` | list[dict] | Per-bucket wait times |
| `lane_scenarios` | dict[int, dict] | Wait estimates for 1–5 lanes |
| `current_queue` | int | Waiting backlog at forecast time (queue after subtracting in-service lanes) |
| `active_lanes` | int | Snapshot lane count kept for dashboard compatibility |
| `inferred_lanes` | int\|None | Lane count inferred from recent `service_events` |
| `est_service_min` | float | Service time used by the wait model |
| `service_time_source` | str | `service_events_median`, `snapshot_avg_dwell`, or `default_dwell` |
| `service_median_min` | float\|None | Median service time from `service_events` |
| `checkout_fraction` | float | Scalar used to convert entrance forecast into checkout-arrival demand |
| `demand_source` | str | `service_events` or `snapshot_fallback` |
| `browsing_est` | dict | Browsing-gap cross-correlation results |
| `df_combined` | DataFrame | Final training data fed to Prophet |
| `real_span_days` | float | Days of real data available |
| `used_sim_history` | bool | True if synthetic data was mixed in |

---

## Current Wait-Model Strategy

The production wait model now uses a layered strategy:

1. **Queue backlog now**
   - Start from the newest `queue_state_snapshots` row
   - Convert raw `queue_count` into **waiting backlog** by subtracting lanes currently serving customers
   - If recent `service_events` are available, the pipeline may infer a better live lane count than the snapshot alone

2. **Per-customer service time**
   - Prefer the **median** `service_events.total_dwell_sec`
   - Fall back to snapshot `avg_dwell_sec`
   - Fall back again to `DEFAULT_DWELL_MIN` only when no better signal exists

3. **Demand conversion**
   - Prophet forecasts **entrance arrivals**
   - The pipeline rescales those arrivals into **checkout-arrival demand** using a single `checkout_fraction`
   - Primary source: observed checkout throughput from `service_events`
   - Fallback source: snapshot lane history and estimated service capacity

4. **Forward simulation**
   - For each forecast bucket:
     - add predicted checkout arrivals
     - subtract service capacity (`lanes × bucket_minutes / service_time`)
     - clamp backlog to `MAX_QUEUE_PER_LANE × lanes`
     - convert remaining backlog into wait minutes

This is intentionally transparent rather than “black box”. The Streamlit app surfaces the chosen service-time source, checkout scaling factor, lane estimate, and historical observed-vs-estimated wait comparisons for debugging.

---

## Prediction Tab Diagnostics

`simulator/app.py` now exposes several wait-debug views:

- **Wait model inputs**
  - queue backlog now
  - snapshot or inferred lane count
  - chosen service time and its source
  - checkout scaling factor and demand source
- **Training-Period Wait Comparison**
  - overlays historical observed wait proxy and model-estimated wait over the same period used to train Prophet
- **Today's Forecast values**
  - shows the exact 15-minute aggregated forecast values rendered in the chart, alongside the aggregated wait forecast

These diagnostics are intended to help locate “factor” errors: whether the mismatch comes from service time, lane count, demand scaling, or backlog interpretation.

---

### `hybrid_wait.py`

Additive helper layer for the separate `simulator/app_hybrid_wait.py` dashboard.
It does not replace `pipeline.py`, and it does not change the behavior of the
existing `simulator/app.py` prediction tab.

**Purpose**:

- keep the existing Prophet + wait pipeline intact
- build a second, REAL-only wait projection path that anchors on current
  operational state first
- blend recent measured checkout behavior with historical/predicted future
  inflow

**High-level strategy**:

1. Reuse `pipeline.run_prediction_pipeline(source="REAL", ...)` as a read-only
   base forecast
2. Derive current state from REAL snapshots and service events:
   - waiting backlog now
   - snapshot lanes and inferred live lanes
   - recent checkout throughput
   - recent median service time
3. Build historical profiles from REAL history:
   - checkout completions by time-of-day bucket
   - lane counts by time-of-day bucket
   - service-time profile by time-of-day bucket
   - sparse checkout history is zero-filled across reference buckets so the
     model does not overstate demand by averaging only non-empty buckets
4. Blend:
   - **current signal** from recent REAL operations
   - **historical/predicted signal** from historical profiles plus the reused
     Prophet demand shape
5. Roll the queue forward bucket-by-bucket and return:
   - `wait_15m`
   - `wait_30m`
   - a detailed per-bucket hybrid wait curve
   - a historical comparison frame for factor debugging

**Primary functions**:

- `load_base_real_prediction(...)` — cached-friendly wrapper around the
  existing REAL prediction pipeline
- `build_hybrid_wait_view(base_result, ...)` — overlays the hybrid wait model
  on top of a previously loaded base result
- `run_hybrid_wait_dashboard(...)` — convenience wrapper that loads the base
  result and builds the hybrid view in one call

**Dashboard behavior exposed by `simulator/app_hybrid_wait.py`**:

- direct comparison of base-pipeline vs hybrid `+15 min` and `+30 min` waits
- model-input diagnostics including snapshot freshness
- future wait values table with per-bucket blend inputs
- training-period backtest with factor and absolute-error metrics

**Important scope note**:

The hybrid wait dashboard is intentionally additive. It exists so the team can
compare a more operationally anchored future-wait model without destabilizing
the current dashboard or CLI behavior.

### `quick.py`

Lightweight version of the pipeline for latency-sensitive use cases (dashboard status cards, frequent polling).

**Differences from `pipeline.py`**:
- Always bootstraps if real data < 14 days (no flag needed)
- Skips browsing-gap estimation
- Skips multi-lane scenarios
- Returns a smaller result dict

```python
result = run_quick_forecast(source="REAL")
# result keys: source, forecast, wait_15m, wait_30m,
#              wait_estimates, current_queue, active_lanes,
#              service_per_bucket, df_combined, real_span_days
```

---

### `cli.py`

Three subcommands exposed via `run_prediction.py` or `python -m prediction`.

#### `quick` — Fast 60-minute forecast

```bash
python run_prediction.py quick
python run_prediction.py quick --source SIM
```

Prints a formatted table of predicted entries + wait times for each 3-minute
bucket in the next 60 minutes, plus wait summaries at +15 and +30 minutes.

#### `pipeline` — Full pipeline

```bash
python run_prediction.py pipeline
python run_prediction.py pipeline --source ALL --days 60
python run_prediction.py pipeline --bootstrap   # force synthetic history mix-in
```

Prints training row count, real data span, wait estimates, and a 10-row
forecast preview.

#### `actuals` — Today's live DB data

```bash
python run_prediction.py actuals
python run_prediction.py actuals --source SIM
```

Prints today's bucketed entry counts as stored in TimescaleDB.

---

## Configuration via `.env`

Create a `.env` file in the `Queue-Management/` root to override defaults:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=iqms
DB_USER=postgres
DB_PASSWORD=your_password

# Shop hours (defaults apply to all days)
SHOP_OPEN=8
SHOP_CLOSE=21

# Per-day overrides (JSON, weekday 0=Mon … 6=Sun). Only keys that differ from
# the defaults are needed. Supported keys: open_hour, open_minute,
# close_hour, close_minute.
# Example: Sunday closes at 13:00
SHOP_SCHEDULE_OVERRIDE={"6": {"close_hour": 13, "close_minute": 0}}

# Prediction settings
BUCKET_MINUTES=3
FORECAST_HORIZON_MINUTES=60
MIN_REAL_DAYS_FOR_SIM=14

# Wait model
MAX_QUEUE_PER_LANE=20
MAX_WAIT_MIN=60
DEFAULT_DWELL_MIN=3.0
DEFAULT_LANES=2
SNAPSHOT_MAX_AGE_MIN=30
```

---

## Wait-Time Model

The wait-time model in `core.compute_wait_estimates()` simulates queue evolution
step by step:

```
For each future bucket i:
    running_queue += forecasted_arrivals[i]
    running_queue -= active_lanes × (BUCKET_MINUTES / avg_dwell_min)
    running_queue  = clamp(running_queue, 0, active_lanes × MAX_QUEUE_PER_LANE)
    wait_min[i]    = (running_queue / service_rate) × avg_dwell_min
    wait_min[i]    = clamp(wait_min[i], 0, MAX_WAIT_MIN)
```

**Inputs**:
- `forecast_df` — Prophet `yhat` values for the next 60 minutes
- `current_queue` — people in queue right now (from latest snapshot)
- `avg_dwell_min` — average service time per person; may be a list (one value per bucket) or a scalar
- `active_lanes` — open service lanes; may be a list (one value per bucket) or a scalar

**Edge-case guards**:
- Returns `([], None, None, 0.0)` immediately when `forecast_df` is empty (`n_steps == 0`)
- `dwell_per_step` is padded to `n_steps` with a fill value of `1.0` if the source list is shorter or empty
- `lanes_per_step` is padded to `n_steps` with a fill value of `1` if the source list is shorter or empty

---

## Bootstrapping

When fewer than `MIN_REAL_DAYS_FOR_SIM` (14) days of real data are available,
`build_sim_history()` generates 7 days of synthetic arrivals using a Poisson
process with time-of-day and weekday/weekend variation:

| Time window | Weekday rate | Weekend rate |
|-------------|-------------|--------------|
| 08:00–10:00 | 2/bucket | 3/bucket |
| 10:00–13:00 | 5/bucket | 8/bucket |
| 13:00–15:00 | 7/bucket | 6/bucket |
| 15:00–18:00 | 4/bucket | 5/bucket |
| 18:00–20:00 | 8/bucket | 4/bucket |
| 20:00–21:00 | 2/bucket | 2/bucket |

This ensures Prophet has enough data to learn daily and weekly seasonality
even during system cold-start.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `prophet` | Time-series forecasting |
| `pandas` | DataFrame manipulation |
| `psycopg2` | PostgreSQL connection |
| `numpy` | Numerical operations |
| `python-dotenv` | `.env` configuration loading |
