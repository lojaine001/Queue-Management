"""Dwell model preparation shared by the background worker and history charts."""
import os
from zoneinfo import ZoneInfo
import pandas as pd
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from prediction.core import DWELL_MIN_FLOOR, DWELL_MAX_CAP, is_open

_DWELL_FEAT = ["hour", "minute_of_hour", "day_of_week", "is_weekend"]
DWELL_MIN_TRAIN_BUCKETS = int(os.getenv("DWELL_MIN_TRAIN_BUCKETS", 24))
DWELL_MIN_TRAIN_EVENTS = int(os.getenv("DWELL_MIN_TRAIN_EVENTS", 60))
DWELL_LSTM_SEQ_LEN = int(os.getenv("DWELL_LSTM_SEQ_LEN", 8))
DWELL_LSTM_MIN_BUCKETS = int(os.getenv("DWELL_LSTM_MIN_BUCKETS", 48))
DWELL_LSTM_MIN_EVENTS = int(os.getenv("DWELL_LSTM_MIN_EVENTS", 120))


def _to_local_series(series):
    local = pd.to_datetime(series)
    if getattr(local.dt, "tz", None) is not None:
        local = local.dt.tz_convert(ZoneInfo(os.getenv("STORE_TZ", "Europe/Paris"))).dt.tz_localize(None)
    return local


def _build_dashboard_dwell_model(service_history: pd.DataFrame, mode: str):
    train = service_history.dropna(subset=["dwell_min"]).copy()
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No dwell history buckets."}

    train["dwell_min"] = pd.to_numeric(train["dwell_min"], errors="coerce")
    train["n_events"] = pd.to_numeric(train.get("n_events"), errors="coerce").fillna(1)
    train = train.dropna(subset=["dwell_min"])
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No valid dwell values."}

    train["ds"] = _to_local_series(train["ds"])
    train = train[train["ds"].apply(lambda t: is_open(pd.Timestamp(t)))]
    train["dwell_min"] = train["dwell_min"].clip(DWELL_MIN_FLOOR, DWELL_MAX_CAP)
    train["hour"] = train["ds"].apply(lambda t: pd.Timestamp(t).hour)
    train["minute_of_hour"] = train["ds"].apply(lambda t: pd.Timestamp(t).minute)
    train["day_of_week"] = train["ds"].apply(lambda t: pd.Timestamp(t).dayofweek)
    train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)

    bucket_count = int(len(train))
    event_count = int(train["n_events"].sum())
    if xgb is None:
        return None, {
            "mode": mode,
            "status": "fallback",
            "reason": "xgboost is not installed; using default dwell time fallback.",
            "bucket_count": bucket_count,
            "event_count": event_count,
        }
    if mode == "safe":
        if bucket_count < DWELL_MIN_TRAIN_BUCKETS or event_count < DWELL_MIN_TRAIN_EVENTS:
            return None, {
                "mode": mode,
                "status": "fallback",
                "reason": (
                    f"Need at least {DWELL_MIN_TRAIN_BUCKETS} dwell buckets and "
                    f"{DWELL_MIN_TRAIN_EVENTS} service events."
                ),
                "bucket_count": bucket_count,
                "event_count": event_count,
            }

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    fit_kwargs = {}
    if mode == "safe":
        fit_kwargs["sample_weight"] = train["n_events"].clip(lower=1).to_numpy(dtype=float)
    model.fit(train[_DWELL_FEAT], train["dwell_min"], **fit_kwargs)
    return model, {
        "mode": mode,
        "status": "trained",
        "reason": "Weighted bucket model." if mode == "safe" else "Legacy bucket model.",
        "bucket_count": bucket_count,
        "event_count": event_count,
    }


def _build_dashboard_dwell_lstm_model(service_history: pd.DataFrame, mode: str):
    train = service_history.dropna(subset=["dwell_min"]).copy()
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No dwell history buckets."}

    train["dwell_min"] = pd.to_numeric(train["dwell_min"], errors="coerce")
    train["n_events"] = pd.to_numeric(train.get("n_events"), errors="coerce").fillna(1)
    train = train.dropna(subset=["dwell_min"])
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No valid dwell values."}

    train["ds"] = _to_local_series(train["ds"])
    train = train[train["ds"].apply(lambda t: is_open(pd.Timestamp(t)))]
    train = train.sort_values("ds").reset_index(drop=True)
    train["dwell_min"] = train["dwell_min"].clip(DWELL_MIN_FLOOR, DWELL_MAX_CAP)

    bucket_count = int(len(train))
    event_count = int(train["n_events"].sum())
    min_buckets = max(DWELL_MIN_TRAIN_BUCKETS, DWELL_LSTM_MIN_BUCKETS) if mode == "safe" else max(12, DWELL_LSTM_SEQ_LEN + 4)
    min_events = max(DWELL_MIN_TRAIN_EVENTS, DWELL_LSTM_MIN_EVENTS) if mode == "safe" else 1
    if bucket_count < min_buckets or event_count < min_events:
        return None, {
            "mode": mode,
            "status": "fallback",
            "reason": f"Need at least {min_buckets} dwell buckets and {min_events} service events.",
            "bucket_count": bucket_count,
            "event_count": event_count,
        }

    values = train["dwell_min"].to_numpy(dtype=float)
    seq_len = max(4, min(DWELL_LSTM_SEQ_LEN, len(values) - 1))
    if len(values) <= seq_len:
        return None, {
            "mode": mode,
            "status": "fallback",
            "reason": f"Need more than {seq_len} buckets for LSTM sequences.",
            "bucket_count": bucket_count,
            "event_count": event_count,
        }

    # Keep only plain input data in Streamlit. Its end-of-run Keras cleanup
    # would otherwise reset state while another browser session uses a model.
    return {"history_values": values.tolist(), "seq_len": seq_len}, {
        "mode": mode,
        "status": "ready",
        "reason": "Sequential open-hour dwell model.",
        "bucket_count": bucket_count,
        "event_count": event_count,
    }
