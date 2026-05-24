"""Plotly chart builders for the Streamlit prediction tab."""

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from prediction.core import (  # noqa: E402
    BUCKET_MINUTES,
    DEFAULT_DWELL_MIN,
    DWELL_MAX_CAP,
    DWELL_MIN_FLOOR,
    FORECAST_HORIZON_MINUTES,
    MAX_QUEUE_PER_LANE,
    MAX_WAIT_MIN,
    SERVICE_MIN_DWELL_SEC,
    SHOP_CLOSE,
    SHOP_CLOSE_TOT,
    SHOP_OPEN,
    SHOP_OPEN_TOT,
    WAIT_15M_INDEX,
    WAIT_30M_INDEX,
)

# ── Chart builders ────────────────────────────────────────────────────────────

DARK = "#0f172a"


def _zero_outside_hours(
    df: pd.DataFrame,
    time_col: str,
    value_cols: "list[str]",
    freq_min: int = 15,
) -> pd.DataFrame:
    """Insert explicit 0 rows for every closed-hour bucket across the df's date range.

    Ensures Plotly draws a flat 0 line between 20:00 and 09:00 instead of
    connecting the last open-hour point straight to the next open-hour point.
    """
    if df.empty:
        return df
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    if df.empty:
        return df
    start = df[time_col].min().normalize()
    end   = df[time_col].max().normalize() + pd.Timedelta(days=1)
    full_range = pd.date_range(start=start, end=end, freq=f"{freq_min}min")
    _range_tot = full_range.hour * 60 + full_range.minute
    closed_ts  = full_range[(_range_tot < SHOP_OPEN_TOT) | (_range_tot >= SHOP_CLOSE_TOT)]
    if closed_ts.empty:
        return df
    zero_rows = pd.DataFrame({time_col: closed_ts})
    for col in value_cols:
        if col in df.columns:
            zero_rows[col] = 0.0
    return (
        pd.concat([df, zero_rows], ignore_index=True)
        .sort_values(time_col)
        .drop_duplicates(subset=[time_col], keep="first")
        .reset_index(drop=True)
    )
GRID = "#1e293b"
DISPLAY_FORECAST_BUCKET_MIN = 15


def _normalized_snapshot_waits(
    df_snap: pd.DataFrame,
    est_service_min: "float | None" = None,
) -> pd.DataFrame:
    """Convert snapshot rows into wait estimates using the same model as the KPI cards."""
    if df_snap.empty:
        return pd.DataFrame(columns=["timestamp", "wait_min"])

    required = {"timestamp", "queue_count", "active_lanes"}
    if not required.issubset(df_snap.columns):
        return pd.DataFrame(columns=["timestamp", "wait_min"])

    hist = df_snap[list(required | {"avg_dwell_sec"} if "avg_dwell_sec" in df_snap.columns else required)].copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["queue_count"] = pd.to_numeric(hist["queue_count"], errors="coerce")
    hist["active_lanes"] = pd.to_numeric(hist["active_lanes"], errors="coerce")
    hist = hist.dropna(subset=["timestamp", "queue_count", "active_lanes"])
    hist = hist[hist["active_lanes"] > 0].copy()
    if hist.empty:
        return pd.DataFrame(columns=["timestamp", "wait_min"])

    if est_service_min is not None:
        service_min = float(min(max(est_service_min, DWELL_MIN_FLOOR), DWELL_MAX_CAP))
    elif "avg_dwell_sec" in hist.columns:
        hist["avg_dwell_sec"] = pd.to_numeric(hist["avg_dwell_sec"], errors="coerce")
        service_min_s = hist["avg_dwell_sec"].fillna(DEFAULT_DWELL_MIN * 60) / 60.0
        service_min_s = service_min_s.clip(lower=DWELL_MIN_FLOOR, upper=DWELL_MAX_CAP)
        waiting_queue = (hist["queue_count"] - hist["active_lanes"]).clip(lower=0.0, upper=hist["active_lanes"] * MAX_QUEUE_PER_LANE)
        wait_min = (waiting_queue * service_min_s / hist["active_lanes"]).clip(lower=0.0, upper=MAX_WAIT_MIN)
        return pd.DataFrame({"timestamp": hist["timestamp"], "wait_min": wait_min.round(1)}).sort_values("timestamp")
    else:
        service_min = DEFAULT_DWELL_MIN

    waiting_queue = (hist["queue_count"] - hist["active_lanes"]).clip(lower=0.0, upper=hist["active_lanes"] * MAX_QUEUE_PER_LANE)
    wait_min = (waiting_queue * service_min / hist["active_lanes"]).clip(lower=0.0, upper=MAX_WAIT_MIN)

    return pd.DataFrame(
        {
            "timestamp": hist["timestamp"],
            "wait_min": wait_min.round(1),
        }
    ).sort_values("timestamp")


def _aggregate_time_series(
    df: pd.DataFrame,
    time_col: str,
    agg_map: dict[str, str],
    *,
    freq_min: int = DISPLAY_FORECAST_BUCKET_MIN,
) -> pd.DataFrame:
    """Aggregate a time series into display-friendly time buckets."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col)
    out = (
        out.groupby(pd.Grouper(key=time_col, freq=f"{freq_min}min"))
        .agg(agg_map)
        .dropna(how="all")
        .reset_index()
    )
    return out


def forecast_display_frame(result: dict) -> pd.DataFrame:
    """Return the same aggregated forecast values used by Today's Forecast."""
    fc = result["forecast"].copy()
    return _aggregate_time_series(
        fc,
        "ds",
        {
            "yhat": "sum",
            "yhat_lower": "sum",
            "yhat_upper": "sum",
        },
    )


def wait_display_frame(result: dict) -> pd.DataFrame:
    """Return display-friendly wait values aligned to the forecast chart buckets."""
    waits = pd.DataFrame(result.get("wait_estimates", []))
    if waits.empty:
        return pd.DataFrame(columns=["ds", "wait_min"])
    waits["ds"] = pd.to_datetime(waits["ds"], errors="coerce")
    waits["wait_min"] = pd.to_numeric(waits["wait_min"], errors="coerce")
    waits = waits.dropna(subset=["ds", "wait_min"])
    if waits.empty:
        return pd.DataFrame(columns=["ds", "wait_min"])
    # Use "first" so the t=now anchor (prepended in pipeline) is preserved in the
    # first display bucket, guaranteeing visual continuity with the historical line.
    return _aggregate_time_series(waits, "ds", {"wait_min": "first"})



def training_wait_comparison_frame(result: dict) -> pd.DataFrame:
    """Observed vs estimated wait over the same period used for training.

    The 'estimated_wait_min' line uses the EXACT SAME code path as the
    historical line in fig_wait_estimates (_normalized_snapshot_waits),
    so both charts are guaranteed to agree.
    """
    df_snap = result.get("df_snapshots", pd.DataFrame()).copy()
    df_combined = result.get("df_combined", pd.DataFrame()).copy()
    est_service_min_raw = result.get("est_service_min")
    est_service_min = pd.to_numeric(
        pd.Series([est_service_min_raw]), errors="coerce"
    ).iloc[0]

    if df_snap.empty or df_combined.empty or not {"timestamp", "queue_count", "active_lanes"}.issubset(df_snap.columns):
        return pd.DataFrame(columns=["ds", "observed_wait_min", "estimated_wait_min", "queue_count", "active_lanes"])

    hist_start = pd.to_datetime(df_combined["ds"], errors="coerce").min()
    hist_end = pd.to_datetime(df_combined["ds"], errors="coerce").max()
    if pd.isna(hist_start) or pd.isna(hist_end):
        return pd.DataFrame(columns=["ds", "observed_wait_min", "estimated_wait_min", "queue_count", "active_lanes"])

    # ── Estimated line: counterfactual with selected lanes ────────────────────
    # If the caller overrides active_lanes (e.g. via the lane slider), apply it
    # to a copy of df_snap so _normalized_snapshot_waits uses the selected count.
    override_lanes = result.get("active_lanes")
    df_snap_est = df_snap.copy()
    if override_lanes is not None and "active_lanes" in df_snap_est.columns:
        df_snap_est["active_lanes"] = int(override_lanes)

    svc_param = float(est_service_min) if not pd.isna(est_service_min) else None
    est_frame = _normalized_snapshot_waits(df_snap_est, est_service_min=svc_param)
    if est_frame.empty:
        return pd.DataFrame(columns=["ds", "observed_wait_min", "estimated_wait_min", "queue_count", "active_lanes"])
    est_frame = est_frame.rename(columns={"timestamp": "ds", "wait_min": "estimated_wait_min"})
    est_frame = est_frame[
        (est_frame["ds"] >= hist_start) & (est_frame["ds"] <= hist_end)
    ].copy()

    # ── Observed line: use the same queue/backlog-based wait formula as the model
    obs_cols = {"timestamp", "queue_count", "active_lanes", "avg_dwell_sec"}
    has_dwell = obs_cols.issubset(df_snap.columns)
    if has_dwell:
        obs = df_snap[list(obs_cols)].copy()
        obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
        obs["queue_count"] = pd.to_numeric(obs["queue_count"], errors="coerce")
        obs["active_lanes"] = pd.to_numeric(obs["active_lanes"], errors="coerce")
        obs["avg_dwell_sec"] = pd.to_numeric(obs["avg_dwell_sec"], errors="coerce")
        obs = obs.dropna(subset=["timestamp", "queue_count", "active_lanes"])
        obs = obs[(obs["timestamp"] >= hist_start) & (obs["timestamp"] <= hist_end)]
        obs = obs[obs["active_lanes"] > 0].copy()
        obs["service_min"] = (obs["avg_dwell_sec"] / 60.0).fillna(DEFAULT_DWELL_MIN).clip(lower=DWELL_MIN_FLOOR, upper=DWELL_MAX_CAP)
        obs["waiting_queue"] = (obs["queue_count"] - obs["active_lanes"]).clip(lower=0.0, upper=obs["active_lanes"] * MAX_QUEUE_PER_LANE)
        obs["observed_wait_min"] = (
            obs["waiting_queue"] * obs["service_min"] / obs["active_lanes"]
        ).clip(lower=0.0, upper=MAX_WAIT_MIN)
        obs = obs.rename(columns={"timestamp": "ds"})
        obs_agg = _aggregate_time_series(
            obs, "ds",
            {"observed_wait_min": "mean", "queue_count": "mean", "active_lanes": "mean"},
        )
    else:
        obs_agg = pd.DataFrame(columns=["ds", "observed_wait_min", "queue_count", "active_lanes"])

    est_agg = _aggregate_time_series(est_frame, "ds", {"estimated_wait_min": "mean"})

    if obs_agg.empty:
        est_agg["observed_wait_min"] = float("nan")
        est_agg["queue_count"] = float("nan")
        est_agg["active_lanes"] = float("nan")
        return est_agg[["ds", "observed_wait_min", "estimated_wait_min", "queue_count", "active_lanes"]]

    merged = pd.merge(obs_agg, est_agg, on="ds", how="outer").sort_values("ds").reset_index(drop=True)
    return merged[["ds", "observed_wait_min", "estimated_wait_min", "queue_count", "active_lanes"]]


def fig_training_data(result: dict,
                       live_actuals: "pd.DataFrame | None" = None) -> go.Figure:
    """Full training history — smoothed actuals + Prophet fit overlay."""
    df = result["df_arrivals"].copy()

    if live_actuals is not None and not live_actuals.empty:
        today = pd.Timestamp.now().normalize()
        df = df[df["bucket"] < today]
        df = pd.concat([df, live_actuals.copy()], ignore_index=True)

    fig = go.Figure()

    # ── Actuals: REAL then SIM, aggregated to display bucket size ────────────
    for src, color, fill in [
        ("REAL", "#38bdf8", "rgba(56,189,248,0.10)"),
        ("SIM",  "#a78bfa", "rgba(167,139,250,0.10)"),
    ]:
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        sub = _zero_outside_hours(sub, "bucket", ["entry_count"])
        agg = _aggregate_time_series(sub, "bucket", {"entry_count": "sum"})
        fig.add_trace(go.Scatter(
            x=agg["bucket"], y=agg["entry_count"],
            mode="lines", name=src,
            line=dict(color=color, width=2, shape="spline", smoothing=0.6),
            fill="tozeroy", fillcolor=fill,
        ))

    # ── Prophet in-sample fit over the full training history ─────────────────
    if "history_fit" in result and not result["history_fit"].empty:
        hf = result["history_fit"][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        hf["ds"] = pd.to_datetime(hf["ds"])
        # Insert zeros for closed hours before resampling so the fit line
        # drops to 0 at 20:00 and stays at 0 until 09:00 (no overnight connection).
        hf = _zero_outside_hours(hf, "ds", ["yhat", "yhat_lower", "yhat_upper"])
        hf = (
            hf.set_index("ds")
            .resample(f"{DISPLAY_FORECAST_BUCKET_MIN}min")
            .sum(min_count=1)
            .dropna()
            .reset_index()
        )
        if not hf.empty:
            ds_fwd = hf["ds"].tolist()
            ds_rev = ds_fwd[::-1]
            fig.add_trace(go.Scatter(
                x=ds_fwd + ds_rev,
                y=hf["yhat_upper"].tolist() + hf["yhat_lower"].tolist()[::-1],
                fill="toself", fillcolor="rgba(244,114,182,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="80% CI",
            ))
            fig.add_trace(go.Scatter(
                x=hf["ds"], y=hf["yhat"],
                mode="lines", name="Prophet fit",
                line=dict(color="#f472b6", width=1.5, dash="dot",
                          shape="spline", smoothing=0.5),
            ))

    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316", line_width=1)
    fig.update_layout(
        title=f"Training History — arrivals / {DISPLAY_FORECAST_BUCKET_MIN} min",
        xaxis_title="Time", yaxis_title=f"Arrivals / {DISPLAY_FORECAST_BUCKET_MIN} min",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_prophet_forecast(result: dict,
                          live_actuals: "pd.DataFrame | None" = None) -> go.Figure:
    """Prophet forecast for today: in-sample fit from shop open + forecast to close.

    live_actuals — if provided (a fresh DB query done at render time), it is
    used for the 'Actual' dots instead of the cached training set so the line
    always extends to now, not to when training last ran.
    """
    fc = result["forecast"].copy()

    today_date = pd.Timestamp.now().normalize()

    # Actuals: sum entry_count per bucket across all cameras, then resample to display buckets.
    # Prefer the fresh live query (correct local-midnight cutoff); fall back to training set.
    if live_actuals is not None and not live_actuals.empty:
        actuals = live_actuals.copy()
        actuals["bucket"] = pd.to_datetime(actuals["bucket"], errors="coerce")
        actuals = actuals.dropna(subset=["bucket"])
        # Sum across cameras for each bucket, rename to the standard (ds, y) schema.
        hist = (
            actuals.groupby("bucket")["entry_count"]
            .sum()
            .reset_index()
            .rename(columns={"bucket": "ds", "entry_count": "y"})
        )
    else:
        hist = result["df_combined"][result["df_combined"]["ds"] >= today_date].copy()

    insamp = result["df_insample"].copy()

    hist   = _zero_outside_hours(hist,   "ds", ["y"])
    insamp = _zero_outside_hours(insamp, "ds", ["yhat", "yhat_lower", "yhat_upper"])
    hist_agg = _aggregate_time_series(hist, "ds", {"y": "sum"})
    insamp_agg = _aggregate_time_series(insamp, "ds", {"yhat": "sum"})
    fc_agg = forecast_display_frame(result)

    fig = go.Figure()
    # Single "Actual" trace — today's real arrivals aggregated to display bucket size
    fig.add_trace(go.Scatter(
        x=hist_agg["ds"], y=hist_agg["y"],
        mode="lines", name=f"Actual ({DISPLAY_FORECAST_BUCKET_MIN} min)",
        line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=0.55),
    ))
    # In-sample fit (aggregated)
    fig.add_trace(go.Scatter(
        x=insamp_agg["ds"], y=insamp_agg["yhat"],
        mode="lines", name="Prophet fit",
        line=dict(color="#f472b6", width=1.8, dash="dot", shape="spline", smoothing=0.5),
    ))
    # Aggregated confidence band
    ds_fwd = fc_agg["ds"].tolist()
    ds_rev = fc_agg["ds"].tolist()[::-1]
    yu = fc_agg["yhat_upper"].tolist()
    yl = fc_agg["yhat_lower"].tolist()[::-1]
    fig.add_trace(go.Scatter(
        x=ds_fwd + ds_rev,
        y=yu + yl,
        fill="toself", fillcolor="rgba(167,139,250,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="80% CI",
    ))
    # Aggregated forecast
    fig.add_trace(go.Scatter(
        x=fc_agg["ds"], y=fc_agg["yhat"],
        mode="lines", name=f"Forecast ({DISPLAY_FORECAST_BUCKET_MIN} min)",
        line=dict(color="#a78bfa", width=3, shape="spline", smoothing=0.6),
    ))
    # Now line
    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316")

    fig.update_layout(
        title=f"Prophet — Today's Forecast ({DISPLAY_FORECAST_BUCKET_MIN} min view)",
        xaxis_title="Time", yaxis_title=f"Arrivals / {DISPLAY_FORECAST_BUCKET_MIN} min",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_prophet_components(result: dict) -> go.Figure:
    """Prophet daily + weekly seasonality components vs actual observed data."""
    from prediction.core import SHOP_OPEN_TOT, SHOP_CLOSE_TOT

    comp       = result["comp_df"]
    df_arr     = result.get("df_arrivals", pd.DataFrame())

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=["Busy hours of the day", "Busy days of the week"],
    )

    # ── Daily: shop-hours only, x-axis as HH:MM ──────────────────────────────
    daily_df = comp.copy()
    daily_df["hour_f"] = daily_df["ds"].dt.hour + daily_df["ds"].dt.minute / 60
    _daily_tot = daily_df["ds"].dt.hour * 60 + daily_df["ds"].dt.minute
    daily_df = daily_df[(_daily_tot >= SHOP_OPEN_TOT) & (_daily_tot < SHOP_CLOSE_TOT)]
    daily_agg = daily_df.groupby("hour_f")["daily"].mean().reset_index()
    daily_agg["label"] = daily_agg["hour_f"].apply(
        lambda h: f"{int(h):02d}:{int((h % 1) * 60):02d}"
    )
    fig.add_trace(go.Scatter(
        x=daily_agg["label"], y=daily_agg["daily"],
        mode="lines", fill="tozeroy",
        line=dict(color="#38bdf8", width=2),
        fillcolor="rgba(56,189,248,0.15)", name="Prophet (hourly)",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#64748b", width=1, dash="dot"),
                  annotation_text="avg", annotation_position="right",
                  annotation_font=dict(color="#94a3b8", size=10), row=1, col=1)

    # ── Weekly: actual observed mean daily arrivals per day-of-week ──────────
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    obs_by_dow = pd.DataFrame({"dow": list(range(7)), "obs_mean": [0.0] * 7, "n_days": [0] * 7})
    if not df_arr.empty and "bucket" in df_arr.columns and "entry_count" in df_arr.columns:
        obs = df_arr.copy()
        obs["bucket"] = pd.to_datetime(obs["bucket"], errors="coerce")
        obs = obs.dropna(subset=["bucket"])
        obs["dow"]  = obs["bucket"].dt.dayofweek
        obs["date"] = obs["bucket"].dt.date
        daily_totals = obs.groupby(["date", "dow"])["entry_count"].sum().reset_index()
        dow_stats = daily_totals.groupby("dow")["entry_count"].agg(["mean", "count"]).reset_index()
        dow_stats.columns = ["dow", "obs_mean", "n_days"]
        obs_by_dow = pd.DataFrame({"dow": list(range(7))}).merge(dow_stats, on="dow", how="left").fillna(0)

    labels = [day_names[d] for d in range(7)]
    values = obs_by_dow["obs_mean"].tolist()
    n_days = obs_by_dow["n_days"].astype(int).tolist()
    grand_mean = float(obs_by_dow["obs_mean"].mean()) or 1.0
    colors = ["#4ade80" if v >= grand_mean else "#f87171" for v in values]
    annotations = [
        f"{int(v)}<br><sub>n={n}</sub>" if n > 0 else "—"
        for v, n in zip(values, n_days)
    ]

    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=annotations,
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="%{x}: %{y:.0f} avg arrivals/day<extra></extra>",
        name="Observed",
    ), row=1, col=2)
    fig.add_hline(y=grand_mean, line=dict(color="#64748b", width=1, dash="dot"),
                  annotation_text="avg day", annotation_position="right",
                  annotation_font=dict(color="#94a3b8", size=10), row=1, col=2)

    fig.update_layout(
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(
        gridcolor=GRID,
        title_text="customers vs. avg",
        title_font=dict(size=11, color="#94a3b8"),
    )
    return fig


def fig_wait_estimates(result: dict) -> go.Figure:
    """Two-zone wait time chart.

    PAST zone (left of NOW):
      Blue solid   → actual measured wait from today's snapshots.
      Orange dots  → what was predicted 15 min before each moment (plotted at prediction_for).
      Orange fill  → error band between predicted and observed — shows model accuracy at a glance.

    FUTURE zone (right of NOW):
      Yellow dashed → full 60-min wait forecast curve.
      Yellow dot    → t+15 min point with annotation.

    Accuracy KPI: mean absolute error over past predictions shown in chart title.
    """
    now      = pd.Timestamp.now()
    today    = now.normalize()
    est_svc  = result.get("est_service_min")

    # ── Observed wait (blue) ──────────────────────────────────────────────────
    obs_x, obs_y = [], []
    df_snap = result["df_snapshots"]
    if not df_snap.empty:
        hist = _normalized_snapshot_waits(df_snap, est_service_min=est_svc)
        hist = hist[hist["timestamp"] >= today].copy()
        hist = hist.rename(columns={"timestamp": "ds"})
        hist = _zero_outside_hours(hist, "ds", ["wait_min"])
        hist = _aggregate_time_series(hist, "ds", {"wait_min": "mean"})
        if not hist.empty:
            obs_x = list(hist["ds"])
            obs_y = list(hist["wait_min"])

    # ── Past predictions (orange) — only rows where prediction_for < now ──────
    df_pred = result.get("df_pred_history", pd.DataFrame())
    past_pred_x, past_pred_y = [], []
    mae_val = None
    if not df_pred.empty and "prediction_for" in df_pred.columns:
        ph = df_pred.copy()
        ph["prediction_for"] = pd.to_datetime(ph["prediction_for"])
        ph = ph[(ph["prediction_for"] >= today) & (ph["prediction_for"] < now)].copy()
        ph = ph.sort_values("prediction_for").drop_duplicates("prediction_for")
        if not ph.empty:
            past_pred_x = list(ph["prediction_for"])
            past_pred_y = list(ph["wait_15m"].fillna(0))

            # Compute MAE: match each past prediction to nearest observed snapshot
            if obs_x:
                obs_series = pd.Series(obs_y, index=pd.to_datetime(obs_x)).sort_index()
                errors = []
                for px, py in zip(past_pred_x, past_pred_y):
                    nearest_idx = obs_series.index.get_indexer([px], method="nearest")[0]
                    if nearest_idx >= 0:
                        errors.append(abs(py - obs_series.iloc[nearest_idx]))
                if errors:
                    mae_val = round(float(sum(errors) / len(errors)), 2)

    # ── Future forecast curve (yellow dashed) ────────────────────────────────
    base_waits = result.get("base_wait_estimates", result.get("wait_estimates", []))
    future_x, future_y = [], []
    dot_x, dot_y = None, None
    if base_waits:
        # waits[0] is the t=now anchor — include it as the bridge from past to future
        future_x = [w["ds"] for w in base_waits]
        future_y = [w["wait_min"] for w in base_waits]
        idx_15 = WAIT_15M_INDEX + 1
        if len(future_x) > idx_15:
            dot_x = future_x[idx_15]
            dot_y = future_y[idx_15]

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Error fill — draw observed first as baseline, then predicted on top with fill
    if obs_x and past_pred_x:
        # Resample observed onto past_pred timestamps for the fill
        if obs_x:
            obs_series = pd.Series(obs_y, index=pd.to_datetime(obs_x)).sort_index()
            fill_obs_y = []
            for px in past_pred_x:
                nearest_idx = obs_series.index.get_indexer([px], method="nearest")[0]
                fill_obs_y.append(float(obs_series.iloc[nearest_idx]) if nearest_idx >= 0 else 0.0)

            # Baseline trace (observed at pred timestamps) — invisible, just for fill anchor
            fig.add_trace(go.Scatter(
                x=past_pred_x, y=fill_obs_y,
                mode="lines", name="_obs_fill_base",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            # Predicted trace filled down to the baseline
            fig.add_trace(go.Scatter(
                x=past_pred_x, y=past_pred_y,
                mode="lines+markers",
                name="Predicted (history)",
                fill="tonexty",
                fillcolor="rgba(251,146,60,0.15)",
                line=dict(color="#fb923c", width=1.5, dash="dot"),
                marker=dict(size=5, color="#fb923c"),
            ))
    elif past_pred_x:
        fig.add_trace(go.Scatter(
            x=past_pred_x, y=past_pred_y,
            mode="lines+markers",
            name="Predicted (history)",
            line=dict(color="#fb923c", width=1.5, dash="dot"),
            marker=dict(size=5, color="#fb923c"),
        ))

    # Observed (blue) — drawn on top of the fill so it's clearly readable
    if obs_x:
        fig.add_trace(go.Scatter(
            x=obs_x, y=obs_y,
            mode="lines", name="Observed (actual)",
            line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=0.55),
        ))

    # Future forecast curve (yellow dashed)
    if future_x:
        fig.add_trace(go.Scatter(
            x=future_x, y=future_y,
            mode="lines", name="Forecast (next 60 min)",
            line=dict(color="#fde047", width=2, dash="dash"),
        ))

    # t+15 min dot
    if dot_x is not None:
        fig.add_trace(go.Scatter(
            x=[dot_x], y=[dot_y],
            mode="markers+text",
            name="+15 min",
            text=[f"+15 min: {dot_y:.1f} min"],
            textposition="top center",
            marker=dict(size=13, color="#facc15", line=dict(color="#0f172a", width=1.5)),
            showlegend=False,
        ))

    # ── Reference lines ───────────────────────────────────────────────────────
    fig.add_hline(y=1, line_dash="dot", line_color="#f97316", line_width=1,
                  annotation_text="BUSY", annotation_position="right")
    fig.add_hline(y=2, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="ALERT", annotation_position="right")
    fig.add_vline(x=now.isoformat(), line_dash="dash", line_color="#64748b", line_width=1)

    # ── Title with accuracy KPI ───────────────────────────────────────────────
    acc_label = f"— Précision modèle (MAE) : {mae_val} min" if mae_val is not None else "— précision : en cours d'accumulation"
    fig.update_layout(
        title=f"Attente en caisse — Aujourd'hui {acc_label}",
        xaxis_title="Heure", yaxis_title="Attente (min)",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID, orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_queue_evolution(result: dict) -> go.Figure:
    """Per-bucket diagnostic: shows arrivals vs throughput and the resulting queue/wait build-up.

    Top panel: scaled checkout arrivals (blue bars) vs service throughput (red dashed line).
    Bottom panel: running queue depth (yellow area) and wait_min (green line, right axis).

    Use this to identify which bucket the queue starts diverging and confirm the root cause
    of unrealistic wait estimates.
    """
    import math

    fw = result.get("forecast_wait")
    est_service_min = result.get("est_service_min") or DEFAULT_DWELL_MIN
    lanes = result.get("snap_lanes_formula") or result.get("active_lanes") or 1
    current_queue = result.get("current_queue") or 0

    if fw is None or (hasattr(fw, "empty") and fw.empty):
        return go.Figure().update_layout(
            title="Queue Evolution — no forecast data",
            paper_bgcolor=DARK, plot_bgcolor=DARK, font=dict(color="#e2e8f0"),
        )

    fw = fw.copy()
    fw["ds"] = pd.to_datetime(fw["ds"], errors="coerce")
    fw = fw.dropna(subset=["ds", "yhat"]).reset_index(drop=True)

    svc_min = min(max(float(est_service_min), DWELL_MIN_FLOOR), DWELL_MAX_CAP)
    service_per_bucket = lanes * (BUCKET_MINUTES / svc_min)

    xs, arrivals, queues, waits = [], [], [], []
    running_queue = float(current_queue)
    for row in fw.itertuples(index=False):
        arr = max(0.0, float(row.yhat))
        running_queue = max(0.0, running_queue + arr - service_per_bucket)
        running_queue = min(running_queue, lanes * MAX_QUEUE_PER_LANE)
        wait = (running_queue / service_per_bucket * BUCKET_MINUTES) if service_per_bucket > 0 else 0.0
        wait = min(wait, MAX_WAIT_MIN)
        xs.append(row.ds)
        arrivals.append(round(arr, 2))
        queues.append(round(running_queue, 1))
        waits.append(round(wait, 2))

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"Checkout arrivals vs throughput  (lanes={lanes}, svc={svc_min:.1f} min → cap={service_per_bucket:.1f}/bucket)",
            "Queue depth & wait",
        ],
        vertical_spacing=0.12,
        row_heights=[0.45, 0.55],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # Top: arrivals bars
    fig.add_trace(go.Bar(
        x=xs, y=arrivals,
        name="Checkout arrivals/bucket (model)",
        marker_color="rgba(56,189,248,0.7)",
        hovertemplate="%{x|%H:%M}: %{y:.1f} arrivals<extra></extra>",
    ), row=1, col=1)

    # Top: historical checkout rate reference (what service_events actually shows)
    hist_checkout_rate = result.get("hist_checkout_rate")
    if hist_checkout_rate is not None:
        fig.add_hline(
            y=hist_checkout_rate, line_dash="dot", line_color="#4ade80", line_width=1.5,
            annotation_text=f"hist rate {hist_checkout_rate:.1f}",
            annotation_position="right",
            annotation_font=dict(color="#4ade80", size=10),
            row=1, col=1,
        )

    # Top: throughput reference
    fig.add_hline(
        y=service_per_bucket, line_dash="dash", line_color="#ef4444", line_width=1.5,
        annotation_text=f"throughput {service_per_bucket:.1f}",
        annotation_position="right",
        annotation_font=dict(color="#ef4444", size=10),
        row=1, col=1,
    )

    # Bottom: running queue (area)
    fig.add_trace(go.Scatter(
        x=xs, y=queues,
        name="Queue depth",
        fill="tozeroy", mode="lines",
        line=dict(color="#fbbf24", width=1.5),
        fillcolor="rgba(251,191,36,0.2)",
        hovertemplate="%{x|%H:%M}: %{y:.0f} waiting<extra></extra>",
    ), row=2, col=1, secondary_y=False)

    # Bottom: wait_min line (right axis)
    fig.add_trace(go.Scatter(
        x=xs, y=waits,
        name="Wait (min)",
        mode="lines",
        line=dict(color="#4ade80", width=2.5),
        hovertemplate="%{x|%H:%M}: %{y:.1f} min wait<extra></extra>",
    ), row=2, col=1, secondary_y=True)

    # BUSY/ALERT reference lines on wait axis
    for y_ref, label, color in [(1, "1 min BUSY", "#f97316"), (2, "2 min ALERT", "#ef4444")]:
        fig.add_hline(y=y_ref, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(color=color, size=9),
                      row=2, col=1, secondary_y=True)

    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316", line_width=1)

    fig.update_layout(
        title="Queue Evolution — per 3-min forecast bucket",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID, orientation="h", y=-0.08),
        margin=dict(l=10, r=60, t=50, b=10),
        barmode="overlay",
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="Queue depth", row=2, col=1, secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Wait (min)", row=2, col=1, secondary_y=True,
                     range=[0, max(max(waits) * 1.2, 6)], showgrid=False)
    return fig


def fig_training_wait_comparison(result: dict) -> go.Figure:
    """Observed vs estimated wait over the training-data history window."""
    comp = training_wait_comparison_frame(result)

    fig = go.Figure()
    if comp.empty:
        return fig.update_layout(
            title="Training-Period Wait Comparison",
            plot_bgcolor=DARK,
            paper_bgcolor=DARK,
            font=dict(color="#e2e8f0"),
        )

    # Zero out data outside shop hours so the chart drops to 0 at close/open.
    comp = _zero_outside_hours(comp, "ds", ["observed_wait_min", "estimated_wait_min"])

    fig.add_trace(go.Scatter(
        x=comp["ds"], y=comp["observed_wait_min"],
        mode="lines", name=f"Observed backlog wait ({DISPLAY_FORECAST_BUCKET_MIN} min)",
        line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=0.55),
    ))
    override_lanes = result.get("active_lanes")
    est_label = (
        f"Estimated — {int(override_lanes)} lane(s) (counterfactual)"
        if override_lanes is not None else "Estimated wait"
    )
    fig.add_trace(go.Scatter(
        x=comp["ds"], y=comp["estimated_wait_min"],
        mode="lines", name=est_label,
        line=dict(color="#f472b6", width=1.8, dash="dot", shape="spline", smoothing=0.5),
    ))
    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316", line_width=1)
    fig.update_layout(
        title=f"Training-Period Wait Comparison ({DISPLAY_FORECAST_BUCKET_MIN} min view)",
        xaxis_title="Time", yaxis_title="Wait (min)",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_lane_scenarios(result: dict) -> go.Figure:
    """Wait time forecast for 1–4 open lanes — full planning view.

    Two panels sharing the x-axis (forecast time):

    Top — wait time curves for each lane count.
          Current active lanes drawn with a thicker solid line.
          Other lanes drawn as thinner dashed lines.
          BUSY (2 min) and ALERT (5 min) reference lines.

    Bottom — aggregated arrival forecast (yhat) for context: shows when demand
              spikes so the manager can decide when to open/close lanes.
    """
    scenarios    = result.get("lane_scenarios", {})
    active_lanes = result.get("active_lanes", 2)
    forecast     = forecast_display_frame(result)
    horizon_end  = pd.Timestamp.now() + pd.Timedelta(minutes=FORECAST_HORIZON_MINUTES)
    forecast     = forecast[forecast["ds"] <= horizon_end]

    LANE_COLORS = {
        1: "#ef4444",   # red   — worst
        2: "#f97316",   # orange
        3: "#fbbf24",   # yellow
        4: "#4ade80",   # green — best
        5: "#38bdf8",
    }

    if not scenarios or forecast.empty:
        return go.Figure().update_layout(
            title="No lane scenario data", paper_bgcolor=DARK,
            plot_bgcolor=DARK, font=dict(color="#e2e8f0"),
        )

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Predicted wait by number of open lanes",
                        "Arrival forecast (context)"],
        vertical_spacing=0.10,
        row_heights=[0.70, 0.30],
    )

    # ── Top: wait curves per lane count ──────────────────────────────────────
    for n in sorted(scenarios.keys()):
        waits = scenarios[n]["wait_estimates"]
        if not waits:
            continue
        xs = [w["ds"] for w in waits]
        ys = [w["wait_min"] for w in waits]
        is_active = (n == active_lanes)
        w15 = scenarios[n]["wait_15m"]
        w30 = scenarios[n]["wait_30m"]
        label = (
            f"{n} lane{'s' if n > 1 else ''}"
            + (" ← current" if is_active else "")
            + f"  |  15 min: {w15:.0f} min  30 min: {w30:.0f} min"
            if w15 is not None and w30 is not None
            else f"{n} lane{'s' if n > 1 else ''}"
            + (" ← current" if is_active else "")
        )
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=label,
            line=dict(
                color=LANE_COLORS.get(n, "#e2e8f0"),
                width=3 if is_active else 1.5,
                dash="solid" if is_active else "dash",
            ),
        ), row=1, col=1)

    # Reference lines
    fig.add_hline(y=1,  line_dash="dot", line_color="#f97316", line_width=1,
                  annotation_text="1 min — BUSY",  annotation_position="right",
                  row=1, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="2 min — ALERT", annotation_position="right",
                  row=1, col=1)
    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316", line_width=1)

    # ── Bottom: arrival forecast ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines", fill="tozeroy",
        line=dict(color="#a78bfa", width=1.5),
        fillcolor="rgba(167,139,250,0.12)",
        name="Arrivals forecast", showlegend=False,
    ), row=2, col=1)

    # ── Summary table annotation ──────────────────────────────────────────────
    rows_15 = []
    rows_30 = []
    for n in sorted(scenarios.keys()):
        w15 = scenarios[n].get("wait_15m")
        w30 = scenarios[n].get("wait_30m")
        marker = " ◀" if n == active_lanes else ""
        rows_15.append(f"{n}L: {w15:.0f} min{marker}" if w15 is not None else f"{n}L: —")
        rows_30.append(f"{n}L: {w30:.0f} min{marker}" if w30 is not None else f"{n}L: —")

    fig.update_layout(
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID, orientation="v", x=1.01, y=1.0,
                    xanchor="left", yanchor="top"),
        margin=dict(l=10, r=220, t=45, b=10),
        height=480,
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="Wait (min)", row=1, col=1)
    fig.update_yaxes(title_text="Arrivals", row=2, col=1)
    return fig


def fig_demographics(result: dict,
                      live_actuals: "pd.DataFrame | None" = None) -> go.Figure:
    """3-panel demographics layout.

    Top-left  — Donut: overall M/F split for the loaded period.
    Top-right — Histogram: age distribution (all REAL buckets with avg_age data).
    Bottom    — Grouped bars: hourly M/F entry counts for today only.
    """
    df = result["df_arrivals"].copy()

    if live_actuals is not None and not live_actuals.empty:
        today = pd.Timestamp.now().normalize()
        df = df[df["bucket"] < today]
        df = pd.concat([df, live_actuals.copy()], ignore_index=True)

    needed = {"male_count", "female_count", "avg_age"}
    if not needed.issubset(df.columns):
        return go.Figure().update_layout(
            title="Demographics — no gender/age data available",
            paper_bgcolor=DARK, plot_bgcolor=DARK, font=dict(color="#e2e8f0"),
        )

    # Only REAL rows
    if "source" in df.columns:
        real = df[df["source"] != "SIM"].copy()
    else:
        real = df.copy()

    # ── Totals for donut ──────────────────────────────────────────────────────
    total_male   = int(real["male_count"].sum())
    total_female = int(real["female_count"].sum())

    # ── Age values (one per bucket where avg_age is available) ───────────────
    age_vals = real["avg_age"].dropna()

    # ── Today hourly gender bars ──────────────────────────────────────────────
    today = pd.Timestamp.now().normalize()
    today_df = real[real["bucket"] >= today].copy()
    if not today_df.empty:
        today_df["hour"] = today_df["bucket"].dt.floor("1h")
        hourly = (
            today_df.groupby("hour")
            .agg(male=("male_count", "sum"), female=("female_count", "sum"))
            .reset_index()
        )
    else:
        hourly = pd.DataFrame(columns=["hour", "male", "female"])

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = sp.make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=[
            "Gender split (period total)",
            "Age distribution",
            "Hourly gender breakdown — today",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # Donut
    fig.add_trace(go.Pie(
        labels=["Male", "Female"],
        values=[total_male, total_female],
        hole=0.55,
        marker=dict(colors=["#38bdf8", "#f472b6"],
                    line=dict(color=DARK, width=2)),
        textinfo="label+percent",
        textfont=dict(size=13),
        showlegend=False,
    ), row=1, col=1)

    # Age histogram
    if not age_vals.empty:
        fig.add_trace(go.Histogram(
            x=age_vals,
            nbinsx=20,
            marker_color="#fbbf24",
            marker_line=dict(color=DARK, width=1),
            opacity=0.85,
            name="Age",
            showlegend=False,
        ), row=1, col=2)

    # Hourly gender bars
    if not hourly.empty:
        hour_labels = hourly["hour"].dt.strftime("%H:%M").tolist()
        fig.add_trace(go.Bar(
            x=hour_labels, y=hourly["male"],
            name="Male", marker_color="#38bdf8",
            marker_line=dict(color=DARK, width=0.5),
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=hour_labels, y=hourly["female"],
            name="Female", marker_color="#f472b6",
            marker_line=dict(color=DARK, width=0.5),
        ), row=2, col=1)
    else:
        fig.add_annotation(
            text="No data yet today", xref="x3", yref="y3",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#94a3b8", size=13),
        )

    fig.update_layout(
        barmode="group",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID, orientation="h", x=0.5, xanchor="center", y=-0.05),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    # Age axis label
    fig.update_xaxes(title_text="Age (years)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Entries", row=2, col=1)
    return fig


def fig_queue_history(result: dict, window_hours: int = 24) -> go.Figure:
    """Queue depth history — aggregated 15-min buckets, smoothed trend, wait estimate.

    Three stacked panels sharing the x-axis:

    Top    — Queue depth: faint raw scatter + P25/P75 band + median line (15-min buckets).
             Derived wait estimate (queue×service/lanes, capped) overlaid as dashed line.
    Middle — Active lanes as a step chart so lane changes are visible.
    Bottom — Avg dwell (min) smoothed over 15-min buckets.

    Data is limited to the last `window_hours` hours to avoid a dense multi-day blob.
    """
    from prediction.core import DWELL_MIN_FLOOR, DWELL_MAX_CAP

    df_raw = result.get("df_snapshots", pd.DataFrame())
    est_service_min = float(result.get("est_service_min", 3.0))
    BUCKET = "15min"

    if df_raw.empty:
        return go.Figure().update_layout(
            title="No queue snapshot data", paper_bgcolor=DARK,
            plot_bgcolor=DARK, font=dict(color="#e2e8f0"),
        )

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Limit to the requested window
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=window_hours)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return go.Figure().update_layout(
            title=f"No queue data in last {window_hours} h", paper_bgcolor=DARK,
            plot_bgcolor=DARK, font=dict(color="#e2e8f0"),
        )

    df["queue_count"]  = pd.to_numeric(df["queue_count"],  errors="coerce")
    df["active_lanes"] = pd.to_numeric(df["active_lanes"], errors="coerce").fillna(2)

    # Zero out any rows outside shop hours before deriving wait or aggregating
    df = _zero_outside_hours(df, "timestamp", ["queue_count", "wait_min"])
    df["queue_count"] = df["queue_count"].fillna(0)
    df["active_lanes"] = df["active_lanes"].fillna(2)
    df["bucket"] = df["timestamp"].dt.floor(BUCKET)

    # Derived wait estimate per snapshot row
    waiting = (df["queue_count"] - df["active_lanes"]).clip(lower=0)
    df["wait_min"] = (waiting * est_service_min / df["active_lanes"].clip(lower=1)).clip(upper=MAX_WAIT_MIN)

    # 15-min aggregation: median + P25/P75 for queue depth and wait
    agg = (
        df.groupby("bucket")
        .agg(
            q_median=("queue_count", "median"),
            q_p25=("queue_count",    lambda s: s.quantile(0.25)),
            q_p75=("queue_count",    lambda s: s.quantile(0.75)),
            wait_median=("wait_min", "median"),
            lanes_median=("active_lanes", "median"),
            dwell_median=("avg_dwell_sec", "median") if "avg_dwell_sec" in df.columns else ("queue_count", "count"),
        )
        .reset_index()
    )
    has_dwell = "avg_dwell_sec" in df.columns

    n_panels = 3 if has_dwell else 2
    row_heights = [0.55, 0.20, 0.25] if has_dwell else [0.70, 0.30]
    subplot_titles = (
        ["Queue depth (15-min buckets)", "Active lanes", "Avg dwell (min)"]
        if has_dwell
        else ["Queue depth (15-min buckets)", "Active lanes"]
    )

    fig = sp.make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        row_heights=row_heights,
    )

    # ── Panel 1: queue depth ─────────────────────────────────────────────────
    # P25/P75 band (shows spread across raw snapshots in each bucket)
    fig.add_trace(go.Scatter(
        x=pd.concat([agg["bucket"], agg["bucket"].iloc[::-1]]),
        y=pd.concat([agg["q_p75"],  agg["q_p25"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(249,115,22,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Queue P25–P75",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # Median line
    fig.add_trace(go.Scatter(
        x=agg["bucket"], y=agg["q_median"],
        mode="lines",
        line=dict(color="#f97316", width=2.5),
        name="Queue depth (median)",
        hovertemplate="%{x|%H:%M}  %{y:.0f} people<extra></extra>",
    ), row=1, col=1)

    # Raw scatter (faint, for reference — shows individual snapshot noise)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["queue_count"],
        mode="markers",
        marker=dict(color="#f97316", size=3, opacity=0.18),
        name="Raw snapshots",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # Derived wait estimate (secondary context)
    fig.add_trace(go.Scatter(
        x=agg["bucket"], y=agg["wait_median"],
        mode="lines",
        line=dict(color="#fbbf24", width=1.5, dash="dash"),
        name="Est. wait (min)",
        hovertemplate="%{x|%H:%M}  ~%{y:.1f} min wait<extra></extra>",
    ), row=1, col=1)

    # Wait reference lines
    fig.add_hline(y=2, line_dash="dot", line_color="#f97316", line_width=1,
                  annotation_text="2 min", annotation_position="right",
                  annotation_font_color="#64748b", row=1, col=1)
    fig.add_hline(y=5, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="5 min", annotation_position="right",
                  annotation_font_color="#64748b", row=1, col=1)

    # ── Panel 2: active lanes (step chart) ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=agg["bucket"], y=agg["lanes_median"],
        mode="lines",
        line=dict(color="#38bdf8", width=2, shape="hv"),  # step
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.08)",
        name="Active lanes",
        hovertemplate="%{x|%H:%M}  %{y:.0f} lanes<extra></extra>",
    ), row=2, col=1)

    # ── Panel 3: dwell ────────────────────────────────────────────────────────
    if has_dwell:
        dwell_min = df.copy()
        dwell_min["dwell_bucket"] = dwell_min["timestamp"].dt.floor(BUCKET)
        dwell_agg = (
            dwell_min.groupby("dwell_bucket")["avg_dwell_sec"]
            .median()
            .div(60)
            .clip(lower=0)
            .reset_index()
            .rename(columns={"dwell_bucket": "bucket", "avg_dwell_sec": "dwell_min"})
        )
        fig.add_trace(go.Scatter(
            x=dwell_agg["bucket"], y=dwell_agg["dwell_min"],
            mode="lines",
            line=dict(color="#a78bfa", width=2),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.10)",
            name="Avg dwell (min)",
            hovertemplate="%{x|%H:%M}  %{y:.1f} min<extra></extra>",
        ), row=3, col=1)

    n_raw = len(df)
    n_buckets = len(agg)
    fig.update_layout(
        title=f"Queue History — last {window_hours} h  ({n_raw} snapshots → {n_buckets} × 15-min buckets)",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID, orientation="h", x=0.5, xanchor="center", y=-0.04),
        margin=dict(l=10, r=80, t=55, b=10),
        height=480,
    )
    fig.update_xaxes(gridcolor=GRID, tickformat="%H:%M")
    fig.update_yaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="People / min", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Lanes", rangemode="tozero", dtick=1, row=2, col=1)
    if has_dwell:
        fig.update_yaxes(title_text="Min", rangemode="tozero", row=3, col=1)
    return fig


def fig_dwell_analysis(result: dict) -> go.Figure:
    """Three-panel dwell analysis — all derived from entrance_events + snapshots.

    Top (full width) — Dwell time series last 24 h:
        · scatter dots : individual entrance_events.dwell_seconds/60 (ground truth)
        · solid line   : snapshot rolling avg_dwell_sec/60 (system live view)
        · dashed line  : clamped model input [DWELL_MIN_FLOOR, DWELL_MAX_CAP]
        · dotted ref   : DEFAULT_DWELL_MIN assumed prior

    Bottom-left — Dwell distribution (histogram, last N days):
        Shape of individual queue wait times — single peak, bimodal, long tail.

    Bottom-right — Dwell by hour of day (bar + error bar, last N days):
        Median wait per hour across all real days loaded.
        Shows whether morning / lunch / evening queues are systematically different.
    """
    df_dwell     = result.get("df_dwell",     pd.DataFrame())
    df_snap      = result.get("df_snapshots", pd.DataFrame())
    df_service   = result.get("df_service",   pd.DataFrame())
    browsing_est = result.get("browsing_est", {})

    cutoff_24h = pd.Timestamp.now() - pd.Timedelta(hours=24)
    days_label = result.get("days", "?")

    fig = sp.make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=[
            "Entrance queue dwell — last 24 h",
            f"Dwell distributions (last {days_label} days)",
            "Estimated shop time breakdown",
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
        row_heights=[0.55, 0.45],
    )

    # ── Top: dwell time series ─────────────────────────────────────────────────
    has_top_data = False

    if not df_dwell.empty and "timestamp" in df_dwell.columns:
        recent = df_dwell[df_dwell["timestamp"] >= cutoff_24h]
        if not recent.empty:
            fig.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["dwell_min"],
                mode="markers", name="Individual dwell",
                marker=dict(color="#38bdf8", size=5, opacity=0.45),
            ), row=1, col=1)
            has_top_data = True

    if not df_snap.empty and "avg_dwell_sec" in df_snap.columns:
        snap_r = df_snap[df_snap["timestamp"] >= cutoff_24h].copy().sort_values("timestamp")
        if not snap_r.empty:
            avg_min = snap_r["avg_dwell_sec"] / 60.0
            fig.add_trace(go.Scatter(
                x=snap_r["timestamp"], y=avg_min,
                mode="lines", name="Snapshot avg (rolling)",
                line=dict(color="#f97316", width=2),
            ), row=1, col=1)
            clamped = avg_min.clip(lower=0.5, upper=10.0)
            fig.add_trace(go.Scatter(
                x=snap_r["timestamp"], y=clamped,
                mode="lines", name="Model input (clamped)",
                line=dict(color="#ef4444", width=1.5, dash="dash"),
            ), row=1, col=1)
            has_top_data = True

    # Reference lines as Scatter traces (reliable on colspan subplots)
    if has_top_data:
        all_ts = []
        if not df_dwell.empty and "timestamp" in df_dwell.columns:
            all_ts = df_dwell["timestamp"].tolist()
        elif not df_snap.empty:
            all_ts = df_snap["timestamp"].tolist()
        if all_ts:
            x0, x1 = min(all_ts), max(all_ts)
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[DEFAULT_DWELL_MIN, DEFAULT_DWELL_MIN],
                mode="lines", name=f"Prior default ({DEFAULT_DWELL_MIN} min)",
                line=dict(color="#64748b", width=1, dash="dot"),
            ), row=1, col=1)
        # Now line as a shape — does not affect y-axis autorange
        fig.add_shape(
            type="line",
            x0=pd.Timestamp.now(), x1=pd.Timestamp.now(),
            y0=0, y1=1, yref="y domain",
            line=dict(color="#f97316", width=1, dash="dash"),
            row=1, col=1,
        )
    else:
        fig.add_annotation(
            text="No dwell data in last 24 h",
            xref="paper", yref="paper", x=0.5, y=0.78,
            showarrow=False, font=dict(color="#94a3b8", size=13),
        )

    # ── Bottom-left: overlapping dwell distributions ──────────────────────────
    has_entrance_dist = False
    has_service_dist  = False

    if not df_dwell.empty and "dwell_min" in df_dwell.columns:
        vals = df_dwell["dwell_min"].dropna()
        vals = vals[(vals > 0) & (vals < 60)]          # cap outliers for readability
        if not vals.empty:
            fig.add_trace(go.Histogram(
                x=vals, nbinsx=30,
                marker_color="#38bdf8",
                marker_line=dict(color=DARK, width=0.5),
                opacity=0.65, name=f"Entrance queue (med {vals.median():.1f} min)",
            ), row=2, col=1)
            has_entrance_dist = True

    if not df_service.empty and "service_min" in df_service.columns:
        svals = df_service["service_min"].dropna()
        svals = svals[(svals > 0) & (svals < 60)]
        if not svals.empty:
            fig.add_trace(go.Histogram(
                x=svals, nbinsx=30,
                marker_color="#a78bfa",
                marker_line=dict(color=DARK, width=0.5),
                opacity=0.65, name=f"Service/checkout (med {svals.median():.1f} min)",
            ), row=2, col=1)
            has_service_dist = True

    if not has_entrance_dist and not has_service_dist:
        fig.add_annotation(
            text="No dwell data available",
            xref="paper", yref="paper", x=0.25, y=0.18,
            showarrow=False, font=dict(color="#94a3b8", size=12),
        )

    # ── Bottom-right: shop time breakdown bar ─────────────────────────────────
    avg_e   = browsing_est.get("avg_entrance_min",  None)
    avg_s   = browsing_est.get("avg_service_min",   None)
    gap     = browsing_est.get("peak_lag_min",       None)
    r_val   = browsing_est.get("correlation",        None)
    total   = browsing_est.get("est_total_min",      None)

    # Fallback: compute avgs directly from loaded data if browsing_est is sparse
    if avg_e is None and not df_dwell.empty and "dwell_min" in df_dwell.columns:
        avg_e = round(float(df_dwell["dwell_min"].mean()), 1)
    if avg_s is None and not df_service.empty and "service_min" in df_service.columns:
        avg_s = round(float(df_service["service_min"].mean()), 1)

    has_breakdown = avg_e is not None or avg_s is not None

    if has_breakdown:
        segments, colors, labels = [], [], []
        if avg_e:
            segments.append(avg_e)
            colors.append("#38bdf8")
            labels.append(f"Queue wait<br>{avg_e} min")
        if gap is not None:
            segments.append(gap)
            colors.append("#64748b")
            labels.append(f"Browsing gap<br>{gap} min (r={r_val:.2f})")
        else:
            segments.append(0)
            colors.append("#334155")
            labels.append("Browsing gap<br>(not enough data)")
        if avg_s:
            segments.append(avg_s)
            colors.append("#a78bfa")
            labels.append(f"Service<br>{avg_s} min")

        fig.add_trace(go.Bar(
            x=labels, y=segments,
            marker_color=colors,
            marker_line=dict(color=DARK, width=1),
            text=[f"{v:.1f} min" if v > 0 else "?" for v in segments],
            textposition="auto",
            name="Shop time components", showlegend=False,
        ), row=2, col=2)

        if total:
            fig.add_annotation(
                text=f"<b>Est. total: {total} min</b>",
                xref="paper", yref="paper",
                x=0.75, y=0.05,
                showarrow=False,
                font=dict(color="#fbbf24", size=13),
                bgcolor=GRID, bordercolor="#fbbf24", borderwidth=1,
            )
    else:
        fig.add_annotation(
            text="No data for shop time breakdown",
            xref="paper", yref="paper", x=0.75, y=0.18,
            showarrow=False, font=dict(color="#94a3b8", size=12),
        )

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID, orientation="h", x=0.5, xanchor="center", y=-0.05),
        margin=dict(l=10, r=10, t=55, b=40),
        height=540,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    # Top panel y-axis: autorange from data, not from 0–30 sentinel line
    fig.update_yaxes(title_text="Dwell (min)", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="Dwell (min)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Minutes", row=2, col=2)
    return fig


def fig_active_lanes_history(result: dict) -> go.Figure:
    """Bar chart: observed active lanes per 30-min time-of-day slot.

    Bars — Phase 1 (blue): service_events median lanes (lane_id or throughput).
           Phase 2 (gray): snapshot active_lanes fallback where Phase 1 is absent.
    Dashed line — Phase 3: demand-implied float (arrivals × checkout rate × service).
    """
    import math
    from prediction.core import SHOP_OPEN, SHOP_CLOSE, SHOP_OPEN_TOT, SHOP_CLOSE_TOT

    _MAX_LANES = 4
    agg = result.get("df_lane_history", pd.DataFrame())
    days = result.get("days", 30)
    # Use historical_checkout_fraction (stable, from full training dataset) so the
    # yellow curve matches exactly what _build_lane_schedule uses for Phase 3.
    # Fall back to checkout_fraction (current-window) only when not available.
    hist_cf = result.get("historical_checkout_fraction")
    checkout_fraction = float(hist_cf if hist_cf is not None else result.get("checkout_fraction", 1.0))
    est_service_min = float(result.get("est_service_min", 3.0))
    checkout_service_min = float(result.get("checkout_service_min", est_service_min))

    # Build full slot index (minute-accurate: 08:30 to 20:00)
    all_labels = []
    h = SHOP_OPEN
    while h < SHOP_CLOSE:
        for m in (0, 30):
            if SHOP_OPEN_TOT <= h * 60 + m < SHOP_CLOSE_TOT:
                all_labels.append(f"{h:02d}:{m:02d}")
        h += 1
    full_index = pd.DataFrame({"slot_label": all_labels})

    # Drop any out-of-hours slots that may survive from stale DB data
    if not agg.empty and "slot_label" in agg.columns:
        agg = agg[agg["slot_label"].apply(
            lambda s: SHOP_OPEN_TOT <= int(s.split(":")[0]) * 60 + int(s.split(":")[1]) < SHOP_CLOSE_TOT
        )].copy()

    fill_defaults = {
        "median_lanes": 0.0, "lanes_source": "none", "n_days_observed": 0,
        "snapshot_lanes": 0.0, "n_days_snapshot": 0,
        "median_entrance_per_30min": 0.0, "n_days_entrance": 0,
    }
    if agg.empty:
        merged = full_index.copy()
        for col, val in fill_defaults.items():
            merged[col] = val
    else:
        merged = full_index.merge(agg, on="slot_label", how="left")
        for col, val in fill_defaults.items():
            if col not in merged.columns:
                merged[col] = val
        merged = merged.fillna(fill_defaults)

    # Phase 3: demand-implied — mirrors _build_lane_schedule Phase 3 exactly.
    # capacity_per_30 = how many customers one lane can serve in 30 min
    capacity_per_30 = 30.0 / max(checkout_service_min, 0.5)
    implied = (
        merged["median_entrance_per_30min"] * checkout_fraction / capacity_per_30
    ).clip(lower=0, upper=_MAX_LANES)   # hard cap matches pipeline max_lanes
    max_n_p3 = int(merged["n_days_entrance"].max()) if not agg.empty else 0

    # Priority: Phase 1 → Phase 2 → Phase 3  (matches _build_lane_schedule)
    # P1 = service_events (any source: lane_id or throughput, both shown blue)
    # P2 = snapshot active_lanes (directly observed — same source as Training-Period Wait Comparison)
    # P3 = demand-implied (fallback when no observed data)
    use_p1 = merged["median_lanes"] > 0
    use_p2 = (~use_p1) & (merged["snapshot_lanes"] > 0)
    use_p3 = (~use_p1) & (~use_p2) & (implied > 0) & (checkout_fraction > 0)

    best_lanes = merged["median_lanes"].where(
        use_p1,
        merged["snapshot_lanes"].round().where(use_p2,
        implied.apply(lambda v: math.ceil(v)).where(use_p3, 0.0))
    ).clip(lower=0, upper=_MAX_LANES)
    bar_colors = [
        "#38bdf8" if p1 else ("#94a3b8" if p2 else ("#f59e0b" if p3 else "#1e293b"))
        for p1, p2, p3 in zip(use_p1, use_p2, use_p3)
    ]
    n_days_label = merged["n_days_observed"].where(
        use_p1,
        merged["n_days_snapshot"].where(use_p2,
        merged["n_days_entrance"].where(use_p3, 0))
    )
    bar_labels = [f"n={int(n)}d" if n > 0 else "" for n in n_days_label]

    has_p1 = use_p1.any()
    has_p2 = use_p2.any()
    has_p3 = use_p3.any()
    max_n_p1 = int(merged.loc[use_p1, "n_days_observed"].max()) if has_p1 else 0
    max_n_p2 = int(merged.loc[use_p2, "n_days_snapshot"].max()) if has_p2 else 0
    max_n_p3 = int(merged.loc[use_p3, "n_days_entrance"].max()) if has_p3 else 0

    fig = go.Figure()

    phase_labels = [
        "service events" if p1 else ("snapshot" if p2 else ("demand-implied" if p3 else "no data"))
        for p1, p2, p3 in zip(use_p1, use_p2, use_p3)
    ]
    hover_texts = [
        f"{slot}<br><b>{int(lanes)} lane{'s' if lanes != 1 else ''}</b> ({src})<extra></extra>"
        for slot, lanes, src in zip(merged["slot_label"], best_lanes, phase_labels)
    ]
    fig.add_trace(go.Bar(
        x=merged["slot_label"],
        y=best_lanes,
        name="Active lanes (used for wait estimate)",
        marker_color=bar_colors,
        opacity=0.85,
        text=bar_labels,
        textposition="outside",
        textfont=dict(size=9, color="#94a3b8"),
        hovertemplate=hover_texts,
    ))

    # Yellow curve: demand-implied float shown for P2 (snapshot) and P3 slots so the
    # user can compare the demand-implied estimate against the observed snapshot or bar.
    # Hidden for P1 slots where service_events already give a direct observation.
    implied_display = implied.where(use_p2 | use_p3, other=float("nan"))
    if implied_display.notna().any() and implied_display.max() > 0:
        fig.add_trace(go.Scatter(
            x=merged["slot_label"],
            y=implied_display,
            mode="lines+markers",
            name=f"Demand-implied float (n={max_n_p3}d)",
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
            marker=dict(size=5, color="#f59e0b"),
            connectgaps=False,
            hovertemplate="%{x}<br>%{y:.2f} implied lanes (pre-ceil)<extra></extra>",
        ))

    snap_lanes = result.get("active_lanes")
    if snap_lanes is not None:
        fig.add_hline(
            y=snap_lanes, line_dash="dot", line_color="#64748b",
            annotation_text=f"Current snapshot: {snap_lanes}",
            annotation_position="top right",
            annotation_font_color="#64748b",
        )

    if has_p1:
        fig.add_trace(go.Bar(x=[], y=[], name=f"Service events lane_id (n={max_n_p1}d)",
                             marker_color="#38bdf8", showlegend=True))
    if has_p2:
        fig.add_trace(go.Bar(x=[], y=[], name=f"Snapshot observed (n={max_n_p2}d)",
                             marker_color="#94a3b8", showlegend=True))
    if has_p3:
        fig.add_trace(go.Bar(x=[], y=[], name=f"Demand-implied ceil (n={max_n_p3}d)",
                             marker_color="#f59e0b", showlegend=True))

    source_parts = []
    if has_p1: source_parts.append("service events")
    if has_p2: source_parts.append("snapshots")
    if has_p3: source_parts.append("demand-implied")
    source_note = " + ".join(source_parts) or "no observed data yet"
    fig.update_layout(
        title=f"Active lanes by time of day — last {days} days  ({source_note})",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        height=320,
        margin=dict(l=10, r=10, t=55, b=50),
        yaxis=dict(title="Active lanes", gridcolor=GRID, dtick=1, rangemode="tozero"),
        xaxis=dict(title="Time of day", gridcolor=GRID, tickangle=-45),
        legend=dict(bgcolor=GRID, orientation="h", x=0.5, xanchor="center", y=-0.20),
        barmode="overlay",
    )
    return fig
