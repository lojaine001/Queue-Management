from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    from . import pipeline as pred_pipeline
    from .core import (
        BUCKET_MINUTES,
        DEFAULT_DWELL_MIN,
        DEFAULT_LANES,
        DWELL_MAX_CAP,
        DWELL_MIN_FLOOR,
        MAX_QUEUE_PER_LANE,
        MAX_WAIT_MIN,
        SHOP_CLOSE_TOT,
        SHOP_OPEN_TOT,
        SNAPSHOT_MAX_AGE_MIN,
        SERVICE_MIN_DWELL_SEC,
        WAIT_15M_INDEX,
        WAIT_30M_INDEX,
        is_open,
    )
except ImportError:
    import pipeline as pred_pipeline
    from core import (
        BUCKET_MINUTES,
        DEFAULT_DWELL_MIN,
        DEFAULT_LANES,
        DWELL_MAX_CAP,
        DWELL_MIN_FLOOR,
        MAX_QUEUE_PER_LANE,
        MAX_WAIT_MIN,
        SHOP_CLOSE_TOT,
        SHOP_OPEN_TOT,
        SNAPSHOT_MAX_AGE_MIN,
        SERVICE_MIN_DWELL_SEC,
        WAIT_15M_INDEX,
        WAIT_30M_INDEX,
        is_open,
    )

AUTO_SHIFT_STEP = 0.15


@dataclass(frozen=True)
class HybridWeights:
    current: float
    history: float


def _local_now_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now())


def _localize_ts(ts: pd.Timestamp | datetime | None) -> pd.Timestamp | None:
    if ts is None or pd.isna(ts):
        return None
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        local_tz = datetime.now().astimezone().tzinfo
        stamp = pd.Timestamp(stamp.tz_convert(local_tz).to_pydatetime().replace(tzinfo=None))
    return stamp


def _clamp_service_minutes(value: float | None) -> float:
    if value is None or pd.isna(value):
        return DEFAULT_DWELL_MIN
    return float(min(max(float(value), DWELL_MIN_FLOOR), DWELL_MAX_CAP))


def _series_median(values: pd.Series, default: float | None = None) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return default
    return float(numeric.median())


def _is_hybrid_open(ts: pd.Timestamp | datetime | None) -> bool:
    stamp = _localize_ts(ts)
    if stamp is None or pd.isna(stamp):
        return False
    tot = stamp.hour * 60 + stamp.minute
    return SHOP_OPEN_TOT <= tot < SHOP_CLOSE_TOT


def _qualifying_service_events(df_service: pd.DataFrame) -> pd.DataFrame:
    if df_service.empty:
        return df_service.copy()
    svc = df_service.copy()
    svc["timestamp"] = pd.to_datetime(svc["timestamp"], errors="coerce")
    if "total_dwell_sec" in svc.columns:
        dwell = pd.to_numeric(svc["total_dwell_sec"], errors="coerce").fillna(0)
        svc = svc[dwell > SERVICE_MIN_DWELL_SEC].copy()
    svc = svc.dropna(subset=["timestamp"])
    svc = svc[svc["timestamp"].apply(_is_hybrid_open)].copy()
    return svc


def _recent_checkout_rate_per_bucket(
    df_service: pd.DataFrame,
    *,
    window_min: int,
    as_of: pd.Timestamp | None = None,
    _svc_qualified: pd.DataFrame | None = None,
) -> float | None:
    svc = _svc_qualified.copy() if _svc_qualified is not None else _qualifying_service_events(df_service)
    if svc.empty:
        return None
    anchor = _localize_ts(as_of) if as_of is not None else _local_now_ts()
    if not _is_hybrid_open(anchor):
        return 0.0
    cutoff = anchor - pd.Timedelta(minutes=window_min)
    svc = svc[svc["timestamp"] >= cutoff].copy()
    if svc.empty:
        bucket_count = max(int(window_min / BUCKET_MINUTES), 1)
        return 0.0 if bucket_count > 0 else None
    svc["_bucket"] = svc["timestamp"].dt.floor(f"{BUCKET_MINUTES}min")
    counts = svc.groupby("_bucket").size()
    bucket_start = cutoff.floor(f"{BUCKET_MINUTES}min")
    bucket_end = anchor.floor(f"{BUCKET_MINUTES}min")
    full_index = pd.date_range(bucket_start, bucket_end, freq=f"{BUCKET_MINUTES}min")
    if len(full_index) == 0:
        return None
    counts = counts.reindex(full_index, fill_value=0)
    return float(counts.mean())


def _recent_service_minutes(
    df_service: pd.DataFrame,
    *,
    window_min: int,
    as_of: pd.Timestamp | None = None,
    _svc_qualified: pd.DataFrame | None = None,
) -> tuple[float | None, float | None]:
    svc = _svc_qualified.copy() if _svc_qualified is not None else _qualifying_service_events(df_service)
    if svc.empty or "service_min" not in svc.columns:
        return None, None
    anchor = _localize_ts(as_of) if as_of is not None else _local_now_ts()
    if not _is_hybrid_open(anchor):
        return None, None
    cutoff = anchor - pd.Timedelta(minutes=window_min)
    svc = svc[svc["timestamp"] >= cutoff].copy()
    if svc.empty:
        return None, None
    service_vals = pd.to_numeric(svc["service_min"], errors="coerce")
    service_vals = service_vals[(service_vals >= DWELL_MIN_FLOOR) & service_vals.notna()]
    if service_vals.empty:
        return None, None
    return float(service_vals.median()), float(service_vals.mean())


def _historical_checkout_profile(
    df_service: pd.DataFrame,
    *,
    reference_labels: list[str] | None = None,
) -> dict[str, float]:
    svc = _qualifying_service_events(df_service)
    if svc.empty:
        return {}
    svc["_bucket"] = svc["timestamp"].dt.floor(f"{BUCKET_MINUTES}min")
    counts = svc.groupby("_bucket").size().rename("count").reset_index()
    counts["date"] = counts["_bucket"].dt.strftime("%Y-%m-%d")
    counts["label"] = counts["_bucket"].dt.strftime("%H:%M")

    if reference_labels:
        labels = sorted(set(reference_labels))
    else:
        labels = sorted(counts["label"].dropna().unique().tolist())
    dates = sorted(counts["date"].dropna().unique().tolist())
    if not labels or not dates:
        return counts.groupby("label")["count"].median().to_dict()

    full_grid = pd.MultiIndex.from_product([dates, labels], names=["date", "label"]).to_frame(index=False)
    counts = full_grid.merge(counts[["date", "label", "count"]], on=["date", "label"], how="left")
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0.0)
    return counts.groupby("label")["count"].median().to_dict()


def _historical_lanes_profile(df_snapshots: pd.DataFrame, *, max_lanes: int) -> dict[str, float]:
    if df_snapshots.empty or "active_lanes" not in df_snapshots.columns:
        return {}
    snaps = df_snapshots.copy()
    snaps["timestamp"] = pd.to_datetime(snaps["timestamp"], errors="coerce")
    snaps["active_lanes"] = pd.to_numeric(snaps["active_lanes"], errors="coerce")
    snaps = snaps.dropna(subset=["timestamp", "active_lanes"])
    snaps = snaps[
        snaps["timestamp"].apply(
            _is_hybrid_open
        )
    ].copy()
    snaps["active_lanes"] = snaps.apply(
        lambda row: _normalize_lane_value(row["active_lanes"], ts=row["timestamp"], max_lanes=max_lanes),
        axis=1,
    )
    snaps = snaps[snaps["active_lanes"] > 0].copy()
    if snaps.empty:
        return {}
    snaps["_bucket"] = snaps["timestamp"].dt.floor(f"{BUCKET_MINUTES}min")
    snaps["label"] = snaps["_bucket"].dt.strftime("%H:%M")
    return snaps.groupby("label")["active_lanes"].median().to_dict()


def _lookup_profile_value(profile: dict[str, float], ts: pd.Timestamp, fallback: float) -> float:
    if not profile:
        return float(fallback)
    value = profile.get(ts.strftime("%H:%M"))
    if value is not None:
        return float(value)
    return float(pd.Series(list(profile.values())).median()) if profile else float(fallback)


def _lane_bounds(
    df_snapshots: pd.DataFrame,
    df_service: pd.DataFrame,
    *,
    _svc_qualified: pd.DataFrame | None = None,
) -> tuple[int, int]:
    max_candidates = [DEFAULT_LANES, 1]

    if not df_snapshots.empty and "active_lanes" in df_snapshots.columns:
        snaps = df_snapshots.copy()
        snaps["timestamp"] = pd.to_datetime(snaps["timestamp"], errors="coerce")
        snaps["active_lanes"] = pd.to_numeric(snaps["active_lanes"], errors="coerce")
        snaps = snaps.dropna(subset=["timestamp", "active_lanes"])
        snaps = snaps[
            snaps["timestamp"].apply(
                _is_hybrid_open
            )
        ].copy()
        if not snaps.empty:
            max_candidates.append(int(snaps["active_lanes"].max()))

    service_open = _svc_qualified.copy() if _svc_qualified is not None else _qualifying_service_events(df_service)
    if not service_open.empty and "lane_id" in service_open.columns:
        lane_count = int(service_open["lane_id"].dropna().nunique())
        if lane_count > 0:
            max_candidates.append(lane_count)

    max_lanes = max(1, max(max_candidates))
    return 1, max_lanes


def _infer_lane_evidence(
    recent_service: pd.DataFrame,
    *,
    est_service_min: float,
    window_min: int,
    max_lanes: int,
) -> dict:
    if recent_service.empty:
        return {
            "recent_service_events": 0,
            "recent_distinct_lane_ids": 0,
            "raw_inferred_lanes": None,
            "lane_source": "snapshot",
        }

    distinct_lane_ids = 0
    if "lane_id" in recent_service.columns:
        distinct_lane_ids = int(recent_service["lane_id"].dropna().nunique())

    raw_inferred = pred_pipeline._infer_active_lanes(
        recent_service,
        est_service_min,
        window_min,
        max_lanes=max_lanes,
    )

    lane_source = "service_lane_id" if distinct_lane_ids > 0 else "service_throughput" if raw_inferred is not None else "snapshot"
    return {
        "recent_service_events": int(len(recent_service)),
        "recent_distinct_lane_ids": distinct_lane_ids,
        "raw_inferred_lanes": raw_inferred,
        "lane_source": lane_source,
    }


def _normalize_lane_value(
    value: float | int | None,
    *,
    ts: pd.Timestamp | None = None,
    max_lanes: int | None = None,
) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    lane_int = int(round(float(numeric)))
    upper = max(1, int(max_lanes or DEFAULT_LANES))
    lane_int = min(max(lane_int, 0), upper)
    if ts is not None and _is_hybrid_open(pd.Timestamp(ts)):
        lane_int = max(lane_int, 1)
    return lane_int


def _effective_weights(
    *,
    current_weight_pct: int,
    future_timestamps: list[pd.Timestamp],
    auto_shift_history: bool,
) -> list[HybridWeights]:
    start_current = min(max(current_weight_pct / 100.0, 0.0), 1.0)
    if not future_timestamps:
        return []

    if auto_shift_history:
        end_current = max(0.0, start_current - AUTO_SHIFT_STEP)
    else:
        end_current = start_current

    total_steps = max(len(future_timestamps) - 1, 1)
    weights: list[HybridWeights] = []
    for idx, _ts in enumerate(future_timestamps):
        ratio = idx / total_steps
        current = start_current + (end_current - start_current) * ratio
        current = min(max(current, 0.0), 1.0)
        weights.append(HybridWeights(current=current, history=1.0 - current))
    return weights


def _forecast_checkout_series(base_result: dict) -> pd.DataFrame:
    forecast = base_result["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    checkout_fraction = float(base_result.get("checkout_fraction", 1.0) or 1.0)
    forecast["checkout_pred"] = pd.to_numeric(forecast["yhat"], errors="coerce").fillna(0.0) * checkout_fraction
    forecast["checkout_pred_lower"] = pd.to_numeric(forecast["yhat_lower"], errors="coerce").fillna(0.0) * checkout_fraction
    forecast["checkout_pred_upper"] = pd.to_numeric(forecast["yhat_upper"], errors="coerce").fillna(0.0) * checkout_fraction
    return forecast


def _current_state(
    base_result: dict,
    df_service: pd.DataFrame,
    *,
    recent_window_min: int,
    lane_strategy: str = "auto",
    manual_active_lanes: int | None = None,
) -> dict:
    df_snapshots = base_result.get("df_snapshots", pd.DataFrame())
    svc_qualified = _qualifying_service_events(df_service)
    _min_lanes, max_lanes_cap = _lane_bounds(
        df_snapshots,
        df_service,
        _svc_qualified=svc_qualified,
    )
    snapshot_lanes = int(base_result.get("active_lanes", DEFAULT_LANES) or DEFAULT_LANES)
    current_waiting_backlog = int(max(0, base_result.get("current_queue", 0) or 0))
    snapshot_queue_count = current_waiting_backlog + snapshot_lanes
    snapshot_ts = None
    snapshot_avg_dwell_min = base_result.get("snapshot_avg_dwell_min")

    if not df_snapshots.empty:
        latest = df_snapshots.iloc[-1]
        snapshot_ts = pd.to_datetime(latest.get("timestamp"), errors="coerce")
        snapshot_queue_count = int(pd.to_numeric(pd.Series([latest.get("queue_count")]), errors="coerce").fillna(snapshot_queue_count).iloc[0])
        snapshot_lanes = int(pd.to_numeric(pd.Series([latest.get("active_lanes")]), errors="coerce").fillna(snapshot_lanes).iloc[0])
        snapshot_avg_dwell_min = (
            pd.to_numeric(pd.Series([latest.get("avg_dwell_sec")]), errors="coerce").fillna(0).iloc[0] / 60.0
            if "avg_dwell_sec" in latest else snapshot_avg_dwell_min
        )

    est_service_min = _clamp_service_minutes(base_result.get("est_service_min"))
    current_clock_ts = _local_now_ts()
    rate_anchor_ts = _localize_ts(snapshot_ts) if snapshot_ts is not None and not pd.isna(snapshot_ts) else current_clock_ts
    anchor_is_open = _is_hybrid_open(current_clock_ts)
    snapshot_lanes = _normalize_lane_value(snapshot_lanes, ts=current_clock_ts, max_lanes=max_lanes_cap)

    recent_service = svc_qualified.copy()
    if not recent_service.empty:
        recent_cutoff = pd.Timestamp(rate_anchor_ts) - pd.Timedelta(minutes=recent_window_min)
        recent_service = recent_service[
            (recent_service["timestamp"] >= recent_cutoff)
            & (recent_service["timestamp"] <= pd.Timestamp(rate_anchor_ts))
        ].copy()
        recent_service = recent_service[
            recent_service["timestamp"].apply(
                _is_hybrid_open
            )
        ].copy()

    lane_evidence = _infer_lane_evidence(
        recent_service,
        est_service_min=est_service_min,
        window_min=recent_window_min,
        max_lanes=max_lanes_cap,
    )
    inferred_lanes_raw = lane_evidence["raw_inferred_lanes"]
    inferred_lanes = (
        _normalize_lane_value(inferred_lanes_raw, ts=rate_anchor_ts, max_lanes=max_lanes_cap)
        if inferred_lanes_raw is not None
        else None
    )
    snapshot_age_min = (
        (current_clock_ts - rate_anchor_ts).total_seconds() / 60
        if rate_anchor_ts is not None and not pd.isna(rate_anchor_ts)
        else None
    )
    snapshot_is_fresh = snapshot_age_min is not None and snapshot_age_min <= SNAPSHOT_MAX_AGE_MIN

    if lane_strategy == "manual" and manual_active_lanes is not None:
        active_lanes_now = _normalize_lane_value(
            manual_active_lanes,
            ts=rate_anchor_ts,
            max_lanes=max_lanes_cap,
        )
        active_lane_source = "manual"
    elif lane_strategy == "snapshot":
        active_lanes_now = _normalize_lane_value(
            snapshot_lanes,
            ts=rate_anchor_ts,
            max_lanes=max_lanes_cap,
        )
        active_lane_source = "snapshot_forced"
    elif lane_strategy == "inferred" and inferred_lanes is not None:
        active_lanes_now = _normalize_lane_value(
            inferred_lanes,
            ts=rate_anchor_ts,
            max_lanes=max_lanes_cap,
        )
        active_lane_source = lane_evidence["lane_source"]
    elif snapshot_is_fresh and snapshot_lanes >= 1:
        active_lanes_now = snapshot_lanes
        active_lane_source = "snapshot"
    elif inferred_lanes is not None:
        active_lanes_now = _normalize_lane_value(
            inferred_lanes,
            ts=rate_anchor_ts,
            max_lanes=max_lanes_cap,
        )
        active_lane_source = lane_evidence["lane_source"]
    else:
        active_lanes_now = _normalize_lane_value(
            snapshot_lanes,
            ts=rate_anchor_ts,
            max_lanes=max_lanes_cap,
        )
        active_lane_source = "snapshot_fallback"
    current_waiting_backlog = max(0, snapshot_queue_count - active_lanes_now)

    recent_checkout_rate = _recent_checkout_rate_per_bucket(
        df_service,
        window_min=recent_window_min,
        as_of=rate_anchor_ts,
        _svc_qualified=svc_qualified,
    )
    recent_service_median, recent_service_mean = _recent_service_minutes(
        df_service,
        window_min=max(recent_window_min, 60),
        as_of=rate_anchor_ts,
        _svc_qualified=svc_qualified,
    )

    if not anchor_is_open:
        snapshot_queue_count = 0
        snapshot_lanes = 0
        inferred_lanes = 0
        active_lanes_now = 0
        current_waiting_backlog = 0
        recent_checkout_rate = 0.0
        recent_service_median = None
        recent_service_mean = None

    return {
        "snapshot_ts": snapshot_ts,
        "current_clock_ts": current_clock_ts,
        "snapshot_queue_count": snapshot_queue_count,
        "snapshot_lanes": snapshot_lanes,
        "snapshot_avg_dwell_min": snapshot_avg_dwell_min,
        "max_lanes_cap": max_lanes_cap,
        "anchor_is_open": anchor_is_open,
        "snapshot_age_min": snapshot_age_min,
        "inferred_lanes": inferred_lanes,
        "recent_service_events": lane_evidence["recent_service_events"],
        "recent_distinct_lane_ids": lane_evidence["recent_distinct_lane_ids"],
        "raw_inferred_lanes": lane_evidence["raw_inferred_lanes"],
        "active_lane_source": active_lane_source,
        "active_lanes_now": active_lanes_now,
        "current_waiting_backlog": current_waiting_backlog,
        "recent_checkout_rate": recent_checkout_rate,
        "recent_service_median": recent_service_median,
        "recent_service_mean": recent_service_mean,
        "est_service_min": est_service_min,
    }


def _classify_backtest_root_cause(row: pd.Series) -> str:
    factor = pd.to_numeric(pd.Series([row.get("factor")]), errors="coerce").fillna(0.0).iloc[0]
    wait_delta = pd.to_numeric(pd.Series([row.get("wait_delta")]), errors="coerce").fillna(0.0).iloc[0]
    service_ratio = pd.to_numeric(pd.Series([row.get("service_ratio")]), errors="coerce").fillna(1.0).iloc[0]
    lane_ratio = pd.to_numeric(pd.Series([row.get("lane_ratio")]), errors="coerce").fillna(1.0).iloc[0]
    if factor <= 1.1 or wait_delta <= 0.25:
        return "Aligned"
    service_hot = service_ratio >= 1.15
    lane_thin = lane_ratio <= 0.90
    if service_hot and lane_thin:
        return "Service + lanes"
    if service_hot:
        return "Service-led"
    if lane_thin:
        return "Lane-led"
    return "Other inflation"


def _smooth_profile_map(
    profile: dict[str, float],
    *,
    min_factor: float,
    max_factor: float,
    neighbor_span: int = 1,
) -> dict[str, float]:
    if not profile:
        return {}

    labels = sorted(profile.keys())
    values = [float(profile[label]) for label in labels]
    smoothed: dict[str, float] = {}
    for idx, label in enumerate(labels):
        start = max(0, idx - neighbor_span)
        end = min(len(labels), idx + neighbor_span + 1)
        window = values[start:end]
        if not window:
            continue
        smoothed[label] = round(min(max(float(pd.Series(window).median()), min_factor), max_factor), 4)
    return smoothed


def _apply_profile_strength(profile: dict[str, object], strength: float) -> dict[str, object]:
    if not profile:
        return profile

    strength = min(max(float(strength), 0.0), 1.0)

    def _blend_factor(value: float) -> float:
        return round(1.0 + ((float(value) - 1.0) * strength), 4)

    return {
        "service_profile": {k: _blend_factor(v) for k, v in profile.get("service_profile", {}).items()},
        "lane_profile": {k: _blend_factor(v) for k, v in profile.get("lane_profile", {}).items()},
        "inflow_profile": {k: _blend_factor(v) for k, v in profile.get("inflow_profile", {}).items()},
        "global_service_factor": _blend_factor(profile.get("global_service_factor", 1.0)),
        "global_lane_factor": _blend_factor(profile.get("global_lane_factor", 1.0)),
        "global_inflow_factor": _blend_factor(profile.get("global_inflow_factor", 1.0)),
    }


def _derive_calibration_profiles(comp: pd.DataFrame) -> dict[str, object]:
    if comp.empty:
        return {
            "service_profile": {},
            "lane_profile": {},
            "inflow_profile": {},
            "global_service_factor": 1.0,
            "global_lane_factor": 1.0,
            "global_inflow_factor": 1.0,
        }

    prof = comp.copy()
    prof["ds"] = pd.to_datetime(prof["ds"], errors="coerce")
    prof = prof.dropna(subset=["ds"])
    if prof.empty:
        return {
            "service_profile": {},
            "lane_profile": {},
            "inflow_profile": {},
            "global_service_factor": 1.0,
            "global_lane_factor": 1.0,
            "global_inflow_factor": 1.0,
        }

    for col in ["factor", "wait_delta", "service_ratio", "lane_ratio"]:
        prof[col] = pd.to_numeric(prof[col], errors="coerce")
    prof["label"] = prof["ds"].dt.strftime("%H:%M")
    prof["root_cause"] = prof.apply(_classify_backtest_root_cause, axis=1)

    service_profile: dict[str, float] = {}
    lane_profile: dict[str, float] = {}
    inflow_profile: dict[str, float] = {}

    global_service_factor = 1.0
    global_lane_factor = 1.0
    global_inflow_factor = 1.0

    service_rows = prof[prof["root_cause"].isin(["Service-led", "Service + lanes"])].copy()
    if not service_rows.empty:
        med = float(service_rows["service_ratio"].dropna().median())
        if med > 0:
            global_service_factor = min(max(1.0 / med, 0.80), 1.05)

    lane_rows = prof[prof["root_cause"].isin(["Lane-led", "Service + lanes"])].copy()
    if not lane_rows.empty:
        med = float(lane_rows["lane_ratio"].dropna().median())
        if med > 0:
            global_lane_factor = min(max(1.0 / med, 1.0), 1.25)

    inflow_rows = prof[prof["root_cause"] == "Other inflation"].copy()
    if not inflow_rows.empty:
        med = float(inflow_rows["factor"].dropna().median())
        if med > 0:
            global_inflow_factor = min(max(1.0 / med, 0.85), 1.0)

    for label, block in prof.groupby("label"):
        causes = block["root_cause"].value_counts()
        dominant = causes.index[0] if not causes.empty else "Aligned"
        bucket_count = int(len(block))
        if bucket_count < 2:
            continue

        if dominant in {"Service-led", "Service + lanes"}:
            med = pd.to_numeric(block["service_ratio"], errors="coerce").dropna().median()
            if pd.notna(med) and med > 0:
                service_profile[label] = round(min(max(1.0 / float(med), 0.80), 1.05), 4)

        if dominant in {"Lane-led", "Service + lanes"}:
            med = pd.to_numeric(block["lane_ratio"], errors="coerce").dropna().median()
            if pd.notna(med) and med > 0:
                lane_profile[label] = round(min(max(1.0 / float(med), 1.0), 1.25), 4)

        if dominant == "Other inflation":
            med = pd.to_numeric(block["factor"], errors="coerce").dropna().median()
            if pd.notna(med) and med > 0:
                inflow_profile[label] = round(min(max(1.0 / float(med), 0.85), 1.0), 4)

    service_profile = _smooth_profile_map(service_profile, min_factor=0.80, max_factor=1.05, neighbor_span=1)
    lane_profile = _smooth_profile_map(lane_profile, min_factor=1.0, max_factor=1.25, neighbor_span=1)
    inflow_profile = _smooth_profile_map(inflow_profile, min_factor=0.85, max_factor=1.0, neighbor_span=1)

    return {
        "service_profile": service_profile,
        "lane_profile": lane_profile,
        "inflow_profile": inflow_profile,
        "global_service_factor": round(global_service_factor, 4),
        "global_lane_factor": round(global_lane_factor, 4),
        "global_inflow_factor": round(global_inflow_factor, 4),
    }


def _hybrid_wait_curve(
    base_result: dict,
    *,
    current_weight_pct: int,
    auto_shift_history: bool,
    recent_window_min: int,
    lane_strategy: str = "auto",
    manual_active_lanes: int | None = None,
    inflow_multiplier: float = 1.0,
    service_multiplier: float = 1.0,
    lane_multiplier: float = 1.0,
    calibration_profiles: dict[str, object] | None = None,
    _precomputed_state: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    df_service = base_result.get("df_service", pd.DataFrame()).copy()
    df_snapshots = base_result.get("df_snapshots", pd.DataFrame()).copy()
    state = _precomputed_state or _current_state(
        base_result,
        df_service,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
    )

    calibration_profiles = calibration_profiles or {}
    service_cal_profile = calibration_profiles.get("service_profile", {})
    lane_cal_profile = calibration_profiles.get("lane_profile", {})
    inflow_cal_profile = calibration_profiles.get("inflow_profile", {})
    global_service_factor = float(calibration_profiles.get("global_service_factor", 1.0) or 1.0)
    global_lane_factor = float(calibration_profiles.get("global_lane_factor", 1.0) or 1.0)
    global_inflow_factor = float(calibration_profiles.get("global_inflow_factor", 1.0) or 1.0)

    current_service_min = _clamp_service_minutes((state["recent_service_median"] or state["est_service_min"]) * service_multiplier * global_service_factor)
    service_profile = pred_pipeline.build_service_time_profile(df_service)
    lanes_profile = _historical_lanes_profile(df_snapshots, max_lanes=state["max_lanes_cap"])

    forecast_checkout = _forecast_checkout_series(base_result)
    reference_labels = pd.to_datetime(forecast_checkout["ds"], errors="coerce").dt.strftime("%H:%M").dropna().unique().tolist()
    checkout_profile = _historical_checkout_profile(df_service, reference_labels=reference_labels)
    future_ts = [pd.Timestamp(ts) for ts in forecast_checkout["ds"].tolist()]
    weights = _effective_weights(
        current_weight_pct=current_weight_pct,
        future_timestamps=future_ts,
        auto_shift_history=auto_shift_history,
    )

    recent_checkout_rate = float(state["recent_checkout_rate"] or 0.0)
    mean_forecast_rate = (
        float(pd.to_numeric(forecast_checkout["checkout_pred"], errors="coerce").fillna(0.0).mean())
        if not forecast_checkout.empty
        else 0.0
    )
    running_queue = float(state["current_waiting_backlog"])
    base_active_lanes_now = int(state["active_lanes_now"] or DEFAULT_LANES)
    active_lanes_now = _normalize_lane_value(
        base_active_lanes_now * lane_multiplier * global_lane_factor,
        ts=state["current_clock_ts"] if state.get("current_clock_ts") is not None else _local_now_ts(),
        max_lanes=state["max_lanes_cap"],
    )
    wait_rows: list[dict[str, object]] = []

    anchor_wait = round(
        min(running_queue * current_service_min / max(active_lanes_now, 1), MAX_WAIT_MIN),
        2,
    )
    if not state["anchor_is_open"]:
        anchor_wait = 0.0
        running_queue = 0.0
        active_lanes_now = 0
    wait_rows.append(
        {
            "ds": _local_now_ts().floor(f"{BUCKET_MINUTES}min"),
            "wait_min": anchor_wait,
            "queue_after": running_queue,
            "lanes_used": active_lanes_now,
            "service_min_used": current_service_min,
            "service_capacity": round(active_lanes_now * (BUCKET_MINUTES / max(current_service_min, 0.5)), 2) if active_lanes_now > 0 else 0.0,
            "net_queue_change": None,
            "current_weight": weights[0].current if weights else current_weight_pct / 100.0,
            "history_weight": weights[0].history if weights else 1.0 - (current_weight_pct / 100.0),
            "recent_real_checkout_rate": recent_checkout_rate,
            "forecast_checkout_pred": None,
            "historical_checkout_rate": None,
            "historical_pred_checkout_rate": None,
            "blended_checkout_rate": None,
        }
    )

    for idx, row in enumerate(forecast_checkout.itertuples(index=False)):
        weight = weights[idx]
        ts = pd.Timestamp(row.ds)
        if not _is_hybrid_open(ts):
            running_queue = 0.0
            wait_rows.append(
                {
                    "ds": ts,
                    "wait_min": 0.0,
                    "queue_after": 0.0,
                    "lanes_used": 0,
                    "service_min_used": 0.0,
                    "service_capacity": 0.0,
                    "net_queue_change": 0.0,
                    "current_weight": round(weight.current, 4),
                    "history_weight": round(weight.history, 4),
                    "recent_real_checkout_rate": round(recent_checkout_rate, 2),
                    "forecast_checkout_pred": 0.0,
                    "historical_checkout_rate": 0.0,
                    "historical_pred_checkout_rate": 0.0,
                    "blended_checkout_rate": 0.0,
                }
            )
            continue
        forecast_rate = max(0.0, float(row.checkout_pred))
        historical_checkout = max(0.0, _lookup_profile_value(checkout_profile, ts, forecast_rate))
        historical_pred = (forecast_rate + historical_checkout) / 2.0
        inflow_correction = _lookup_profile_value(inflow_cal_profile, ts, global_inflow_factor)
        if mean_forecast_rate > 0:
            current_shaped_rate = recent_checkout_rate * (forecast_rate / mean_forecast_rate)
        else:
            current_shaped_rate = recent_checkout_rate
        blended_inflow = ((weight.current * current_shaped_rate) + (weight.history * historical_pred)) * inflow_multiplier * inflow_correction

        historical_service = pred_pipeline.service_time_for_timestamp(service_profile, ts, fallback=current_service_min)
        service_correction = _lookup_profile_value(service_cal_profile, ts, global_service_factor)
        service_used = _clamp_service_minutes(
            ((weight.current * current_service_min) + (weight.history * historical_service)) * service_correction
        )

        historical_lanes = _lookup_profile_value(lanes_profile, ts, active_lanes_now)
        lane_correction = _lookup_profile_value(lane_cal_profile, ts, global_lane_factor)
        lanes_used = _normalize_lane_value(
            ((weight.current * active_lanes_now) + (weight.history * historical_lanes)) * lane_correction,
            ts=ts,
            max_lanes=state["max_lanes_cap"],
        )

        service_capacity = lanes_used * (BUCKET_MINUTES / max(service_used, 0.5))
        net_queue_change = blended_inflow - service_capacity
        running_queue = max(0.0, running_queue + net_queue_change)
        running_queue = min(running_queue, float(lanes_used * MAX_QUEUE_PER_LANE))
        wait_min = min(running_queue * service_used / lanes_used, MAX_WAIT_MIN)

        wait_rows.append(
            {
                "ds": ts,
                "wait_min": round(wait_min, 2),
                "queue_after": round(running_queue, 2),
                "lanes_used": lanes_used,
                "service_min_used": round(service_used, 2),
                "service_capacity": round(service_capacity, 2),
                "net_queue_change": round(net_queue_change, 2),
                "current_weight": round(weight.current, 4),
                "history_weight": round(weight.history, 4),
                "recent_real_checkout_rate": round(current_shaped_rate, 2),
                "forecast_checkout_pred": round(forecast_rate, 2),
                "historical_checkout_rate": round(historical_checkout, 2),
                "historical_pred_checkout_rate": round(historical_pred, 2),
                "blended_checkout_rate": round(blended_inflow, 2),
            }
        )

    wait_df = pd.DataFrame(wait_rows)
    return wait_df, state


def _training_wait_comparison(
    base_result: dict,
    state: dict,
    *,
    current_weight_pct: int,
    auto_shift_history: bool,
    lane_strategy: str = "auto",
    service_multiplier: float = 1.0,
    lane_multiplier: float = 1.0,
    calibration_profiles: dict[str, object] | None = None,
) -> pd.DataFrame:
    df_snapshots = base_result.get("df_snapshots", pd.DataFrame()).copy()
    df_service = base_result.get("df_service", pd.DataFrame()).copy()
    df_combined = base_result.get("df_combined", pd.DataFrame()).copy()
    if df_snapshots.empty or df_combined.empty:
        return pd.DataFrame(
            columns=[
                "ds",
                "observed_wait_min",
                "estimated_wait_min",
                "queue_count",
                "active_lanes",
                "estimated_lanes",
                "waiting_queue",
                "observed_service_min",
                "estimated_service_min",
                "wait_delta",
                "factor",
                "service_ratio",
                "lane_ratio",
            ]
        )

    df_snapshots["timestamp"] = pd.to_datetime(df_snapshots["timestamp"], errors="coerce")
    hist_start = pd.to_datetime(df_combined["ds"], errors="coerce").min()
    hist_end = pd.to_datetime(df_combined["ds"], errors="coerce").max()
    snaps = df_snapshots[(df_snapshots["timestamp"] >= hist_start) & (df_snapshots["timestamp"] <= hist_end)].copy()
    snaps = snaps[snaps["timestamp"].apply(_is_hybrid_open)].copy()
    if snaps.empty:
        return pd.DataFrame(
            columns=[
                "ds",
                "observed_wait_min",
                "estimated_wait_min",
                "queue_count",
                "active_lanes",
                "estimated_lanes",
                "waiting_queue",
                "observed_service_min",
                "estimated_service_min",
                "wait_delta",
                "factor",
                "service_ratio",
                "lane_ratio",
            ]
        )

    calibration_profiles = calibration_profiles or {}
    service_cal_profile = calibration_profiles.get("service_profile", {})
    lane_cal_profile = calibration_profiles.get("lane_profile", {})
    global_service_factor = float(calibration_profiles.get("global_service_factor", 1.0) or 1.0)
    global_lane_factor = float(calibration_profiles.get("global_lane_factor", 1.0) or 1.0)

    service_profile = pred_pipeline.build_service_time_profile(df_service)
    lanes_profile = _historical_lanes_profile(df_snapshots, max_lanes=state["max_lanes_cap"])
    current_service_min = _clamp_service_minutes((state["recent_service_median"] or state["est_service_min"]) * service_multiplier * global_service_factor)
    current_lanes_now = _normalize_lane_value(
        (state["active_lanes_now"] or DEFAULT_LANES) * lane_multiplier * global_lane_factor,
        ts=state["current_clock_ts"] if state.get("current_clock_ts") is not None else _local_now_ts(),
        max_lanes=state["max_lanes_cap"],
    )

    current_weight = min(max(current_weight_pct / 100.0, 0.0), 1.0)
    history_weight = 1.0 - current_weight
    if auto_shift_history:
        history_weight = min(1.0, history_weight + (AUTO_SHIFT_STEP * (2.0 / 3.0)))
        current_weight = 1.0 - history_weight

    rows: list[dict[str, object]] = []
    for snap in snaps.itertuples(index=False):
        ts = pd.Timestamp(snap.timestamp)
        queue_count = max(0.0, float(getattr(snap, "queue_count", 0) or 0))
        raw_snapshot_lanes = max(1.0, float(getattr(snap, "active_lanes", current_lanes_now) or current_lanes_now))
        if lane_strategy == "snapshot":
            active_lanes = raw_snapshot_lanes
        elif lane_strategy == "inferred":
            active_lanes = float(current_lanes_now)
        elif lane_strategy == "manual":
            active_lanes = float(current_lanes_now)
        else:
            active_lanes = raw_snapshot_lanes
        service_proxy = _clamp_service_minutes((float(getattr(snap, "avg_dwell_sec", 0) or 0) / 60.0) if getattr(snap, "avg_dwell_sec", None) is not None else current_service_min)
        waiting_queue = max(0.0, queue_count - active_lanes)
        observed_wait = min(waiting_queue * service_proxy / active_lanes, MAX_WAIT_MIN)

        hist_service = pred_pipeline.service_time_for_timestamp(service_profile, ts, fallback=current_service_min)
        service_correction = _lookup_profile_value(service_cal_profile, ts, global_service_factor)
        blended_service = _clamp_service_minutes(((current_weight * current_service_min) + (history_weight * hist_service)) * service_correction)
        hist_lanes = _lookup_profile_value(lanes_profile, ts, current_lanes_now)
        lane_correction = _lookup_profile_value(lane_cal_profile, ts, global_lane_factor)
        blended_lanes = _normalize_lane_value(
            ((current_weight * current_lanes_now) + (history_weight * hist_lanes)) * lane_correction,
            ts=ts,
            max_lanes=state["max_lanes_cap"],
        )
        estimated_wait = min(waiting_queue * blended_service / blended_lanes, MAX_WAIT_MIN)

        rows.append(
            {
                "ds": ts,
                "observed_wait_min": round(observed_wait, 2),
                "estimated_wait_min": round(estimated_wait, 2),
                "queue_count": round(queue_count, 2),
                "active_lanes": round(active_lanes, 2),
                "estimated_lanes": round(blended_lanes, 2),
                "waiting_queue": round(waiting_queue, 2),
                "observed_service_min": round(service_proxy, 2),
                "estimated_service_min": round(blended_service, 2),
                "wait_delta": round(estimated_wait - observed_wait, 2),
                "factor": round((estimated_wait / observed_wait), 2) if observed_wait > 0 else None,
                "service_ratio": round((blended_service / service_proxy), 2) if service_proxy > 0 else None,
                "lane_ratio": round((blended_lanes / active_lanes), 2) if active_lanes > 0 else None,
            }
        )

    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp
    comp["_bucket"] = comp["ds"].dt.floor("15min")
    comp = (
        comp.groupby("_bucket", as_index=False)
        .agg(
            observed_wait_min=("observed_wait_min", "mean"),
            estimated_wait_min=("estimated_wait_min", "mean"),
            queue_count=("queue_count", "mean"),
            active_lanes=("active_lanes", "mean"),
            estimated_lanes=("estimated_lanes", "mean"),
            waiting_queue=("waiting_queue", "mean"),
            observed_service_min=("observed_service_min", "mean"),
            estimated_service_min=("estimated_service_min", "mean"),
            wait_delta=("wait_delta", "mean"),
            factor=("factor", "mean"),
            service_ratio=("service_ratio", "mean"),
            lane_ratio=("lane_ratio", "mean"),
        )
        .rename(columns={"_bucket": "ds"})
    )
    return comp


def run_hybrid_wait_dashboard(
    *,
    days: int = 30,
    use_bootstrap: bool = False,
    data_since: datetime | None = None,
    current_weight_pct: int = 70,
    auto_shift_history: bool = True,
    recent_window_min: int = 45,
) -> dict:
    base_result = load_base_real_prediction(
        days=days,
        use_bootstrap=use_bootstrap,
        data_since=data_since,
    )
    return build_hybrid_wait_view(
        base_result,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
    )


def load_base_real_prediction(
    *,
    days: int = 30,
    use_bootstrap: bool = False,
    data_since: datetime | None = None,
) -> dict:
    try:
        return pred_pipeline.run_prediction_pipeline(
            source="REAL",
            days=days,
            use_bootstrap=use_bootstrap,
            data_since=data_since,
        )
    except Exception as exc:
        return _fallback_base_result(days=days, data_since=data_since, root_error=str(exc))


def _sql_literal(value: object) -> str:
    return pred_pipeline.psycopg2.extensions.adapt(value).getquoted().decode()


def _fallback_load_arrivals(conn, days: int, data_since: datetime | None) -> pd.DataFrame:
    since = data_since.astimezone(timezone.utc) if data_since else (datetime.now(timezone.utc) - timedelta(days=days))
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"
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
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp >= {_sql_literal(since)}
          AND (camera_id LIKE 'SIM_%%' OR camera_id = {_sql_literal(pred_pipeline.ENTRANCE_CAMERA_ID)})
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) < {SHOP_CLOSE_TOT}
        GROUP BY bucket, camera_id
        ORDER BY bucket
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["bucket"] = pred_pipeline.to_local(df["bucket"])
    df["source"] = df["camera_id"].apply(lambda c: "SIM" if str(c).startswith("SIM_") else "REAL")
    return df


def _fallback_load_snapshots(conn, days: int, data_since: datetime | None) -> pd.DataFrame:
    since = data_since.astimezone(timezone.utc) if data_since else (datetime.now(timezone.utc) - timedelta(days=days))
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"
    query = f"""
        SELECT timestamp, queue_count, avg_dwell_sec, active_lanes, camera_id
        FROM queue_state_snapshots
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp >= {_sql_literal(since)}
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) < {SHOP_CLOSE_TOT}
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["timestamp"] = pred_pipeline.to_local(df["timestamp"])
    df["source"] = df["camera_id"].apply(lambda c: "SIM" if str(c).startswith("SIM_") else "REAL")
    return df


def _fallback_load_dwell_events(conn, days: int) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"
    query = f"""
        SELECT timestamp, dwell_seconds / 60.0 AS dwell_min
        FROM entrance_events
        WHERE camera_id = {_sql_literal(pred_pipeline.ENTRANCE_CAMERA_ID)}
          AND dwell_seconds > 0
          AND timestamp >= {_sql_literal(since)}
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) < {SHOP_CLOSE_TOT}
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["timestamp"] = pred_pipeline.to_local(df["timestamp"])
    return df


def _fallback_load_service_events(conn, days: int) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    utc_offset_sec = int(datetime.now(timezone.utc).astimezone().utcoffset().total_seconds())
    local_ts = f"(timestamp + INTERVAL '{utc_offset_sec} seconds')"
    query = f"""
        SELECT timestamp, total_dwell_sec / 60.0 AS service_min,
               total_dwell_sec, lane_id, camera_id
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND total_dwell_sec > 0
          AND timestamp >= {_sql_literal(since)}
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) >= {SHOP_OPEN_TOT}
          AND (EXTRACT(HOUR FROM {local_ts}) * 60 + EXTRACT(MINUTE FROM {local_ts})) < {SHOP_CLOSE_TOT}
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["timestamp"] = pred_pipeline.to_local(df["timestamp"])
    return df


def _fallback_base_result(
    *,
    days: int,
    data_since: datetime | None,
    root_error: str,
) -> dict:
    conn = pred_pipeline.psycopg2.connect(**pred_pipeline.DB_CONFIG)
    try:
        df_arrivals = _fallback_load_arrivals(conn, days, data_since)
        df_snapshots = _fallback_load_snapshots(conn, days, data_since)
        df_dwell = _fallback_load_dwell_events(conn, days=days)
        df_service = _fallback_load_service_events(conn, days=days)
    finally:
        conn.close()

    df_combined = (
        df_arrivals.groupby("bucket")["entry_count"].sum().reset_index().rename(columns={"bucket": "ds", "entry_count": "y"})
        if not df_arrivals.empty and {"bucket", "entry_count"}.issubset(df_arrivals.columns)
        else pd.DataFrame(columns=["ds", "y"])
    )
    if not df_combined.empty:
        df_combined["ds"] = pd.to_datetime(df_combined["ds"], errors="coerce")
        df_combined["y"] = pd.to_numeric(df_combined["y"], errors="coerce").fillna(0.0)
        df_combined = df_combined.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)

    real_span_days = (
        (df_combined["ds"].max() - df_combined["ds"].min()).total_seconds() / 86400
        if len(df_combined) > 1 else 0.0
    )

    snap_latest = pred_pipeline.latest_snapshot_for_wait(df_snapshots, "REAL")
    snap_age_min = None
    snapshot_avg_dwell_min = None
    if snap_latest is not None and "timestamp" in snap_latest:
        snap_ts = _localize_ts(pd.Timestamp(snap_latest["timestamp"]))
        snap_age_min = (_local_now_ts() - snap_ts).total_seconds() / 60
    snapshot_valid = snap_latest is not None and (snap_age_min is None or snap_age_min < SNAPSHOT_MAX_AGE_MIN)

    if snapshot_valid:
        snap_ts = _localize_ts(pd.Timestamp(snap_latest.get("timestamp")))
        if _is_hybrid_open(snap_ts):
            raw_queue = int(snap_latest.get("queue_count", 0) or 0)
            active_lanes = int(snap_latest.get("active_lanes", DEFAULT_LANES) or DEFAULT_LANES)
            current_queue = max(0, raw_queue - active_lanes)
            snapshot_avg_dwell_min = float(snap_latest.get("avg_dwell_sec", DEFAULT_DWELL_MIN * 60) or (DEFAULT_DWELL_MIN * 60)) / 60.0
        else:
            active_lanes = 0
            current_queue = 0
            snapshot_avg_dwell_min = DEFAULT_DWELL_MIN
    else:
        if "max_lane_depth" in df_arrivals.columns and not df_arrivals.empty:
            current_queue = int(pd.to_numeric(df_arrivals["max_lane_depth"], errors="coerce").fillna(0).tail(3).max())
        else:
            current_queue = 0
        active_lanes = DEFAULT_LANES
        snapshot_avg_dwell_min = DEFAULT_DWELL_MIN

    preferred_service_min, service_time_source, service_median_min, service_mean_min = pred_pipeline._preferred_service_minutes(
        df_service,
        snapshot_avg_dwell_min,
    )
    est_service_min = _clamp_service_minutes(preferred_service_min)

    observed_rate = pred_pipeline._observed_checkout_rate(df_service, BUCKET_MINUTES)
    avg_arrivals = (
        float(pd.to_numeric(df_arrivals["entry_count"], errors="coerce").fillna(0).mean())
        if not df_arrivals.empty and "entry_count" in df_arrivals.columns
        else 0.0
    )
    if observed_rate is not None:
        demand_source = "service_events"
        checkout_fraction = min(max((observed_rate / avg_arrivals), 0.01), 4.0) if avg_arrivals > 0 else 1.0
    else:
        demand_source = "snapshot_fallback"
        checkout_fraction = 1.0

    anchor_wait = round(min(current_queue * est_service_min / max(active_lanes, 1), MAX_WAIT_MIN), 1)
    now_anchor = _local_now_ts().floor(f"{BUCKET_MINUTES}min")

    return {
        "df_arrivals": df_arrivals,
        "df_snapshots": df_snapshots,
        "df_dwell": df_dwell,
        "df_service": df_service,
        "browsing_est": {"warning": "Fallback base result used; Prophet forecast unavailable."},
        "df_combined": df_combined,
        "df_insample": pd.DataFrame(columns=["ds", "yhat"]),
        "forecast": pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"]),
        "comp_df": pd.DataFrame(columns=["ds", "daily", "weekly"]),
        "history_fit": pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"]),
        "wait_estimates": [{"ds": now_anchor, "wait_min": anchor_wait}],
        "wait_15m": None,
        "wait_30m": None,
        "current_queue": current_queue,
        "active_lanes": active_lanes,
        "avg_dwell_min": snapshot_avg_dwell_min,
        "snapshot_avg_dwell_min": snapshot_avg_dwell_min,
        "est_service_min": est_service_min,
        "lane_scenarios": {},
        "real_span_days": real_span_days,
        "used_sim_history": False,
        "source": "REAL",
        "days": days,
        "service_time_source": service_time_source,
        "service_median_min": service_median_min,
        "service_mean_min": service_mean_min,
        "checkout_fraction": checkout_fraction,
        "demand_source": demand_source,
        "inferred_lanes": None,
        "fallback_warning": root_error,
    }


def build_hybrid_wait_view(
    base_result: dict,
    *,
    current_weight_pct: int = 70,
    auto_shift_history: bool = True,
    recent_window_min: int = 45,
    lane_strategy: str = "auto",
    manual_active_lanes: int | None = None,
    inflow_multiplier: float = 1.0,
    service_multiplier: float = 1.0,
    lane_multiplier: float = 1.0,
    use_profile_calibration: bool = True,
    profile_strength: float = 1.0,
) -> dict:

    baseline_wait_curve, state = _hybrid_wait_curve(
        base_result,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
        inflow_multiplier=inflow_multiplier,
        service_multiplier=service_multiplier,
        lane_multiplier=lane_multiplier,
        calibration_profiles=None,
    )
    baseline_comparison = _training_wait_comparison(
        base_result,
        state,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        lane_strategy=lane_strategy,
        service_multiplier=service_multiplier,
        lane_multiplier=lane_multiplier,
        calibration_profiles=None,
    )
    raw_calibration_profiles = _derive_calibration_profiles(baseline_comparison) if use_profile_calibration else {
        "service_profile": {},
        "lane_profile": {},
        "inflow_profile": {},
        "global_service_factor": 1.0,
        "global_lane_factor": 1.0,
        "global_inflow_factor": 1.0,
    }
    calibration_profiles = _apply_profile_strength(raw_calibration_profiles, profile_strength) if use_profile_calibration else raw_calibration_profiles

    wait_curve, state = _hybrid_wait_curve(
        base_result,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
        inflow_multiplier=inflow_multiplier,
        service_multiplier=service_multiplier,
        lane_multiplier=lane_multiplier,
        calibration_profiles=calibration_profiles,
        _precomputed_state=state,
    )
    comparison = _training_wait_comparison(
        base_result,
        state,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        lane_strategy=lane_strategy,
        service_multiplier=service_multiplier,
        lane_multiplier=lane_multiplier,
        calibration_profiles=calibration_profiles,
    )

    future_only = wait_curve.iloc[1:].reset_index(drop=True) if len(wait_curve) > 1 else wait_curve.copy()
    wait_15m = float(future_only.iloc[WAIT_15M_INDEX]["wait_min"]) if len(future_only) > WAIT_15M_INDEX else None
    wait_30m = float(future_only.iloc[WAIT_30M_INDEX]["wait_min"]) if len(future_only) > WAIT_30M_INDEX else None

    current_weight_15 = min(max(current_weight_pct / 100.0, 0.0), 1.0)
    current_weight_30 = max(0.0, current_weight_15 - AUTO_SHIFT_STEP) if auto_shift_history else current_weight_15

    warnings: list[str] = []
    if base_result.get("forecast", pd.DataFrame()).empty:
        warnings.append("No future open buckets were available for the current forecast horizon. Hybrid wait cards stay empty until there is at least one forecast bucket ahead.")
    if state["recent_checkout_rate"] is None:
        warnings.append("Recent checkout throughput is sparse; inflow falls back more heavily toward historical/predicted demand.")
    if state["recent_service_median"] is None:
        warnings.append("Recent service-event timing is sparse; service minutes fall back to historical/default estimates.")
    if state["inferred_lanes"] is None:
        warnings.append("Could not infer live lanes from recent service events; using snapshot lane count.")

    return {
        "base_result": base_result,
        "forecast": base_result["forecast"],
        "history_fit": base_result.get("history_fit"),
        "df_combined": base_result.get("df_combined", pd.DataFrame()),
        "df_service": base_result.get("df_service", pd.DataFrame()),
        "df_snapshots": base_result.get("df_snapshots", pd.DataFrame()),
        "df_arrivals": base_result.get("df_arrivals", pd.DataFrame()),
        "baseline_hybrid_wait_curve": baseline_wait_curve,
        "baseline_training_wait_comparison": baseline_comparison,
        "hybrid_wait_curve": wait_curve,
        "training_wait_comparison": comparison,
        "wait_15m": wait_15m,
        "wait_30m": wait_30m,
        "anchor_is_open": state["anchor_is_open"],
        "current_clock_ts": state.get("current_clock_ts"),
        "snapshot_ts": state["snapshot_ts"],
        "current_waiting_backlog": state["current_waiting_backlog"],
        "snapshot_queue_count": state["snapshot_queue_count"],
        "snapshot_lanes": state["snapshot_lanes"],
        "max_lanes_cap": state["max_lanes_cap"],
        "snapshot_age_min": state["snapshot_age_min"],
        "inferred_lanes": state["inferred_lanes"],
        "raw_inferred_lanes": state["raw_inferred_lanes"],
        "recent_service_events": state["recent_service_events"],
        "recent_distinct_lane_ids": state["recent_distinct_lane_ids"],
        "active_lane_source": state["active_lane_source"],
        "active_lanes_now": state["active_lanes_now"],
        "service_median_min": state["recent_service_median"],
        "service_mean_min": state["recent_service_mean"],
        "service_fallback_min": state["est_service_min"],
        "recent_checkout_rate": state["recent_checkout_rate"],
        "lane_strategy": lane_strategy,
        "current_weight_pct": current_weight_pct,
        "history_weight_pct": 100 - current_weight_pct,
        "effective_current_weight_15_pct": round(current_weight_15 * 100, 1),
        "effective_history_weight_15_pct": round((1.0 - current_weight_15) * 100, 1),
        "effective_current_weight_30_pct": round(current_weight_30 * 100, 1),
        "effective_history_weight_30_pct": round((1.0 - current_weight_30) * 100, 1),
        "auto_shift_history": auto_shift_history,
        "recent_window_min": recent_window_min,
        "inflow_multiplier": inflow_multiplier,
        "service_multiplier": service_multiplier,
        "lane_multiplier": lane_multiplier,
        "use_profile_calibration": use_profile_calibration,
        "profile_strength": profile_strength,
        "raw_calibration_profiles": raw_calibration_profiles,
        "calibration_profiles": calibration_profiles,
        "warnings": warnings,
    }
