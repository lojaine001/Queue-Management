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
    ├─ 8. Derive current queue state from latest snapshot
    └─ 9. Compute wait estimates bucket-by-bucket
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
| `SHOP_OPEN` | 8 | Shop opening hour |
| `SHOP_CLOSE` | 21 | Shop closing hour |
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
- `is_open(ts)` — Returns True if timestamp is within shop hours
- `future_open_timestamps()` — List of future bucket timestamps during opening hours
- `add_closed_zeros(df, days)` — Fills Prophet training data with zeros outside shop hours
- `build_sim_history(days=7)` — Generates synthetic Poisson-distributed arrival history for bootstrapping
- `compute_wait_estimates(forecast_df, current_queue, avg_dwell_min, active_lanes)` — Core wait-time simulation

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
| `current_queue` | int | Queue depth at forecast time |
| `active_lanes` | int | Lanes used for wait calculation |
| `browsing_est` | dict | Browsing-gap cross-correlation results |
| `df_combined` | DataFrame | Final training data fed to Prophet |
| `real_span_days` | float | Days of real data available |
| `used_sim_history` | bool | True if synthetic data was mixed in |

---

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

# Shop hours
SHOP_OPEN=8
SHOP_CLOSE=21

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
- `avg_dwell_min` — average service time per person (from snapshot or default)
- `active_lanes` — open service lanes (from snapshot or default)

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
