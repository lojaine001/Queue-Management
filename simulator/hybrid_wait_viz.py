"""Plotly chart builders for the separate hybrid wait dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from prediction.core import BUCKET_MINUTES, WAIT_15M_INDEX, WAIT_30M_INDEX  # noqa: E402
import simulator.predict_viz as base_viz  # noqa: E402

DARK = "#0f172a"
GRID = "#1e293b"
DISPLAY_BUCKET_MIN = 15
OPEN_HOUR = 9
CLOSE_HOUR = 20


def _local_now_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now().astimezone()).tz_localize(None)


def _aggregate_time_series(
    df: pd.DataFrame,
    time_col: str,
    agg_map: dict[str, str],
    *,
    freq_min: int = DISPLAY_BUCKET_MIN,
) -> pd.DataFrame:
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


def _add_open_hours_overlay(fig: go.Figure, *frames: pd.DataFrame) -> None:
    timestamps: list[pd.Timestamp] = []
    for frame in frames:
        if frame is None or frame.empty or "ds" not in frame.columns:
            continue
        ds = pd.to_datetime(frame["ds"], errors="coerce").dropna()
        timestamps.extend(ds.tolist())
    if not timestamps:
        return

    min_day = min(timestamps).normalize()
    max_day = max(timestamps).normalize()
    day = min_day
    while day <= max_day:
        open_ts = day + pd.Timedelta(hours=OPEN_HOUR)
        close_ts = day + pd.Timedelta(hours=CLOSE_HOUR)
        fig.add_vline(x=open_ts, line_dash="dot", line_color="#22c55e", line_width=1)
        fig.add_vline(x=close_ts, line_dash="dot", line_color="#ef4444", line_width=1)
        fig.add_vrect(
            x0=day,
            x1=open_ts,
            fillcolor="rgba(71,85,105,0.18)",
            line_width=0,
            layer="below",
        )
        fig.add_vrect(
            x0=close_ts,
            x1=day + pd.Timedelta(days=1),
            fillcolor="rgba(71,85,105,0.18)",
            line_width=0,
            layer="below",
        )
        day += pd.Timedelta(days=1)


def future_wait_display_frame(result: dict) -> pd.DataFrame:
    waits = result.get("hybrid_wait_curve", pd.DataFrame()).copy()
    if waits.empty:
        return pd.DataFrame(columns=["ds", "wait_min"])
    waits["ds"] = pd.to_datetime(waits["ds"], errors="coerce")
    waits["wait_min"] = pd.to_numeric(waits["wait_min"], errors="coerce")
    waits = waits.dropna(subset=["ds", "wait_min"])
    if waits.empty:
        return pd.DataFrame(columns=["ds", "wait_min"])
    return _aggregate_time_series(waits, "ds", {"wait_min": "first"})


def future_values_frame(result: dict) -> pd.DataFrame:
    waits = result.get("hybrid_wait_curve", pd.DataFrame()).copy()
    if waits.empty:
        return pd.DataFrame(
            columns=[
                "ds",
                "wait_min",
                "queue_after",
                "lanes_used",
                "service_min_used",
                "service_capacity",
                "net_queue_change",
                "current_weight",
                "history_weight",
                "recent_real_checkout_rate",
                "forecast_checkout_pred",
                "historical_checkout_rate",
                "historical_pred_checkout_rate",
                "blended_checkout_rate",
            ]
        )

    waits["ds"] = pd.to_datetime(waits["ds"], errors="coerce")
    numeric_cols = [
        "wait_min",
        "queue_after",
        "lanes_used",
        "service_min_used",
        "service_capacity",
        "net_queue_change",
        "current_weight",
        "history_weight",
        "recent_real_checkout_rate",
        "forecast_checkout_pred",
        "historical_checkout_rate",
        "historical_pred_checkout_rate",
        "blended_checkout_rate",
    ]
    for col in numeric_cols:
        if col in waits.columns:
            waits[col] = pd.to_numeric(waits[col], errors="coerce")
    return waits.dropna(subset=["ds"]).reset_index(drop=True)


def inflow_display_frame(result: dict) -> pd.DataFrame:
    waits = future_values_frame(result)
    if waits.empty:
        return pd.DataFrame(
            columns=[
                "ds",
                "recent_real_checkout_rate",
                "forecast_checkout_pred",
                "historical_checkout_rate",
                "historical_pred_checkout_rate",
                "blended_checkout_rate",
            ]
        )
    inflow = waits.iloc[1:].copy() if len(waits) > 1 else waits.copy()
    return _aggregate_time_series(
        inflow,
        "ds",
        {
            "recent_real_checkout_rate": "mean",
            "forecast_checkout_pred": "mean",
            "historical_checkout_rate": "mean",
            "historical_pred_checkout_rate": "mean",
            "blended_checkout_rate": "mean",
        },
    )


def training_values_frame(result: dict) -> pd.DataFrame:
    comp = result.get("training_wait_comparison", pd.DataFrame()).copy()
    if comp.empty:
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
    comp["ds"] = pd.to_datetime(comp["ds"], errors="coerce")
    for col in [
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
    ]:
        if col in comp.columns:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")
    return comp.dropna(subset=["ds"]).reset_index(drop=True)


def backtest_factor_profile_frame(result: dict) -> pd.DataFrame:
    comp = training_values_frame(result)
    if comp.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "median_factor",
                "mean_observed_wait_min",
                "mean_estimated_wait_min",
                "mean_wait_delta",
                "mean_service_ratio",
                "mean_lane_ratio",
                "bucket_count",
            ]
        )

    prof = comp.copy()
    prof["label"] = pd.to_datetime(prof["ds"], errors="coerce").dt.strftime("%H:%M")
    prof = prof.dropna(subset=["label"])
    if prof.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "median_factor",
                "mean_observed_wait_min",
                "mean_estimated_wait_min",
                "mean_wait_delta",
                "mean_service_ratio",
                "mean_lane_ratio",
                "bucket_count",
            ]
        )

    prof = (
        prof.groupby("label", as_index=False)
        .agg(
            median_factor=("factor", "median"),
            mean_observed_wait_min=("observed_wait_min", "mean"),
            mean_estimated_wait_min=("estimated_wait_min", "mean"),
            mean_wait_delta=("wait_delta", "mean"),
            mean_service_ratio=("service_ratio", "mean"),
            mean_lane_ratio=("lane_ratio", "mean"),
            bucket_count=("label", "size"),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )
    return prof


def backtest_root_cause_frame(result: dict) -> pd.DataFrame:
    comp = training_values_frame(result)
    if comp.empty:
        return pd.DataFrame(
            columns=[
                "ds",
                "label",
                "factor",
                "wait_delta",
                "service_ratio",
                "lane_ratio",
                "root_cause",
            ]
        )

    root = comp.copy()
    root["label"] = pd.to_datetime(root["ds"], errors="coerce").dt.strftime("%H:%M")
    root["factor"] = pd.to_numeric(root["factor"], errors="coerce")
    root["wait_delta"] = pd.to_numeric(root["wait_delta"], errors="coerce")
    root["service_ratio"] = pd.to_numeric(root["service_ratio"], errors="coerce")
    root["lane_ratio"] = pd.to_numeric(root["lane_ratio"], errors="coerce")

    def _classify(row: pd.Series) -> str:
        factor = float(row.get("factor") or 0.0)
        service_ratio = float(row.get("service_ratio") or 1.0)
        lane_ratio = float(row.get("lane_ratio") or 1.0)
        wait_delta = float(row.get("wait_delta") or 0.0)
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

    root["root_cause"] = root.apply(_classify, axis=1)
    return root


def backtest_root_cause_profile_frame(result: dict) -> pd.DataFrame:
    root = backtest_root_cause_frame(result)
    if root.empty:
        return pd.DataFrame(columns=["label", "root_cause", "bucket_count"])
    prof = (
        root.groupby(["label", "root_cause"], as_index=False)
        .size()
        .rename(columns={"size": "bucket_count"})
        .sort_values(["label", "root_cause"])
        .reset_index(drop=True)
    )
    return prof


def fig_hybrid_wait_forecast(result: dict) -> go.Figure:
    wait_agg = future_wait_display_frame(result)
    raw_waits = future_values_frame(result)
    df_snap = result.get("df_snapshots", pd.DataFrame())
    service_for_history = result.get("service_median_min") or result.get("service_fallback_min")
    now_ts = pd.to_datetime(result.get("current_clock_ts"), errors="coerce")
    if pd.isna(now_ts):
        now_ts = _local_now_ts()

    fig = go.Figure()

    if not df_snap.empty:
        today = now_ts.normalize()
        hist = base_viz._normalized_snapshot_waits(df_snap, est_service_min=service_for_history)
        hist = hist[hist["timestamp"] >= today].copy()
        if not hist.empty:
            hist = hist.rename(columns={"timestamp": "ds"})
            hist = _aggregate_time_series(hist, "ds", {"wait_min": "mean"})
            fig.add_trace(go.Scatter(
                x=hist["ds"],
                y=hist["wait_min"],
                mode="lines",
                name=f"Observed ({DISPLAY_BUCKET_MIN} min)",
                line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=0.55),
            ))

    if not wait_agg.empty:
        fig.add_trace(go.Scatter(
            x=wait_agg["ds"],
            y=wait_agg["wait_min"],
            mode="lines",
            name=f"Hybrid forecast ({DISPLAY_BUCKET_MIN} min)",
            line=dict(color="#a78bfa", width=3, shape="spline", smoothing=0.6),
        ))

    if not raw_waits.empty:
        future_only = raw_waits.iloc[1:].reset_index(drop=True) if len(raw_waits) > 1 else raw_waits.copy()
        highlight_x: list[pd.Timestamp] = []
        highlight_y: list[float] = []
        highlight_text: list[str] = []
        if len(future_only) > WAIT_15M_INDEX:
            highlight_x.append(pd.Timestamp(future_only.iloc[WAIT_15M_INDEX]["ds"]))
            highlight_y.append(float(future_only.iloc[WAIT_15M_INDEX]["wait_min"]))
            highlight_text.append(f"+15 min: {future_only.iloc[WAIT_15M_INDEX]['wait_min']:.2f} min")
        if len(future_only) > WAIT_30M_INDEX:
            highlight_x.append(pd.Timestamp(future_only.iloc[WAIT_30M_INDEX]["ds"]))
            highlight_y.append(float(future_only.iloc[WAIT_30M_INDEX]["wait_min"]))
            highlight_text.append(f"+30 min: {future_only.iloc[WAIT_30M_INDEX]['wait_min']:.2f} min")
        if highlight_x:
            fig.add_trace(go.Scatter(
                x=highlight_x,
                y=highlight_y,
                mode="markers+text",
                name="KPI checkpoints",
                text=highlight_text,
                textposition="top center",
                marker=dict(size=9, color="#facc15", line=dict(color="#0f172a", width=1)),
            ))

    fig.add_hline(
        y=2,
        line_dash="dot",
        line_color="#f97316",
        line_width=1,
        annotation_text="2 min - BUSY",
        annotation_position="right",
    )
    fig.add_hline(
        y=5,
        line_dash="dot",
        line_color="#ef4444",
        line_width=1,
        annotation_text="5 min - ALERT",
        annotation_position="right",
    )
    fig.add_vline(x=now_ts.isoformat(), line_dash="dash", line_color="#f97316", line_width=1)
    _add_open_hours_overlay(fig, wait_agg, raw_waits.rename(columns={"timestamp": "ds"}) if "timestamp" in raw_waits.columns else raw_waits)
    fig.update_layout(
        title=f"Hybrid Wait Forecast ({DISPLAY_BUCKET_MIN} min view)",
        xaxis_title="Time",
        yaxis_title="Wait (min)",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_hybrid_inflow_components(result: dict) -> go.Figure:
    inflow = inflow_display_frame(result)
    now_ts = pd.to_datetime(result.get("current_clock_ts"), errors="coerce")
    if pd.isna(now_ts):
        now_ts = _local_now_ts()
    fig = go.Figure()

    if not inflow.empty:
        fig.add_trace(go.Scatter(
            x=inflow["ds"],
            y=inflow["recent_real_checkout_rate"],
            mode="lines",
            name="Recent REAL checkout rate",
            line=dict(color="#38bdf8", width=2.2, shape="spline", smoothing=0.55),
        ))
        fig.add_trace(go.Scatter(
            x=inflow["ds"],
            y=inflow["historical_pred_checkout_rate"],
            mode="lines",
            name="Historical/predicted inflow",
            line=dict(color="#f472b6", width=1.8, dash="dot", shape="spline", smoothing=0.5),
        ))
        fig.add_trace(go.Scatter(
            x=inflow["ds"],
            y=inflow["blended_checkout_rate"],
            mode="lines",
            name="Blended inflow used",
            line=dict(color="#a78bfa", width=3, shape="spline", smoothing=0.6),
        ))

    fig.add_vline(x=now_ts.isoformat(), line_dash="dash", line_color="#f97316", line_width=1)
    _add_open_hours_overlay(fig, inflow)
    fig.update_layout(
        title=f"Checkout Inflow Blend ({DISPLAY_BUCKET_MIN} min view)",
        xaxis_title="Time",
        yaxis_title=f"Customers / {BUCKET_MINUTES} min",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_hybrid_training_backtest(result: dict) -> go.Figure:
    comp = training_values_frame(result)
    fig = go.Figure()

    if not comp.empty:
        fig.add_trace(go.Scatter(
            x=comp["ds"],
            y=comp["observed_wait_min"],
            mode="lines",
            name=f"Observed wait proxy ({DISPLAY_BUCKET_MIN} min)",
            line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=0.55),
        ))
        fig.add_trace(go.Scatter(
            x=comp["ds"],
            y=comp["estimated_wait_min"],
            mode="lines",
            name="Hybrid estimate",
            line=dict(color="#a78bfa", width=3, shape="spline", smoothing=0.6),
        ))

    _add_open_hours_overlay(fig, comp)
    fig.update_layout(
        title=f"Training-Period Wait Backtest ({DISPLAY_BUCKET_MIN} min view)",
        xaxis_title="Time",
        yaxis_title="Wait (min)",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_backtest_factor_profile(result: dict) -> go.Figure:
    prof = backtest_factor_profile_frame(result)
    fig = go.Figure()

    if not prof.empty:
        fig.add_trace(go.Bar(
            x=prof["label"],
            y=prof["median_factor"],
            name="Median factor",
            marker_color="#a78bfa",
            opacity=0.8,
            hovertemplate="%{x}<br>median factor=%{y:.2f}x<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=prof["label"],
            y=prof["mean_wait_delta"],
            mode="lines+markers",
            name="Mean wait delta",
            line=dict(color="#38bdf8", width=2.2, shape="spline", smoothing=0.45),
            marker=dict(size=5),
            yaxis="y2",
            hovertemplate="%{x}<br>mean delta=%{y:.2f} min<extra></extra>",
        ))

    fig.add_hline(
        y=1,
        line_dash="dot",
        line_color="#22c55e",
        line_width=1,
        annotation_text="1.0x = aligned",
        annotation_position="right",
    )
    fig.update_layout(
        title="Backtest Factor by Time of Day",
        xaxis_title="Time of day",
        yaxis_title="Median factor (x)",
        yaxis2=dict(
            title="Mean wait delta (min)",
            overlaying="y",
            side="right",
            rangemode="tozero",
            gridcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID, tickangle=-45)
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero")
    return fig


def fig_backtest_root_cause_profile(result: dict) -> go.Figure:
    prof = backtest_root_cause_profile_frame(result)
    fig = go.Figure()
    if prof.empty:
        fig.update_layout(
            title="Backtest Root Cause by Time of Day",
            plot_bgcolor=DARK,
            paper_bgcolor=DARK,
            font=dict(color="#e2e8f0"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    palette = {
        "Aligned": "#22c55e",
        "Service-led": "#38bdf8",
        "Lane-led": "#f59e0b",
        "Service + lanes": "#ef4444",
        "Other inflation": "#a78bfa",
    }

    for cause in ["Aligned", "Service-led", "Lane-led", "Service + lanes", "Other inflation"]:
        block = prof[prof["root_cause"] == cause].copy()
        if block.empty:
            continue
        fig.add_trace(go.Bar(
            x=block["label"],
            y=block["bucket_count"],
            name=cause,
            marker_color=palette.get(cause, "#e2e8f0"),
            hovertemplate="%{x}<br>%{y} buckets<extra>" + cause + "</extra>",
        ))

    fig.update_layout(
        title="Backtest Root Cause by Time of Day",
        xaxis_title="Time of day",
        yaxis_title="Bucket count",
        barmode="stack",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID, tickangle=-45)
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero")
    return fig


def fig_lane_strategy_comparison(strategy_views: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    first_view = next((view for view in strategy_views.values() if view), None)
    now_ts = pd.to_datetime(first_view.get("current_clock_ts"), errors="coerce") if first_view else pd.NaT
    if pd.isna(now_ts):
        now_ts = _local_now_ts()

    palette = {
        "auto": "#a78bfa",
        "snapshot": "#38bdf8",
        "inferred": "#f472b6",
        "manual": "#f59e0b",
    }

    for strategy_name in ["auto", "snapshot", "inferred", "manual"]:
        view = strategy_views.get(strategy_name)
        if not view:
            continue
        wait_agg = future_wait_display_frame(view)
        if wait_agg.empty:
            continue
        active_lanes_now = view.get("active_lanes_now", "—")
        lane_source = view.get("active_lane_source", "unknown")
        fig.add_trace(go.Scatter(
            x=wait_agg["ds"],
            y=wait_agg["wait_min"],
            mode="lines",
            name=f"{strategy_name} ({active_lanes_now} lanes)",
            line=dict(
                color=palette.get(strategy_name, "#e2e8f0"),
                width=3 if strategy_name == "auto" else 2.2,
                dash="solid" if strategy_name in {"auto", "snapshot"} else "dot",
                shape="spline",
                smoothing=0.55,
            ),
            hovertemplate=(
                f"strategy={strategy_name}"
                f"<br>lane_source={lane_source}"
                "<br>%{x|%H:%M}: %{y:.2f} min<extra></extra>"
            ),
        ))

    frames = [future_wait_display_frame(view) for view in strategy_views.values() if view]
    _add_open_hours_overlay(fig, *frames)
    fig.add_hline(
        y=2,
        line_dash="dot",
        line_color="#f97316",
        line_width=1,
        annotation_text="2 min - BUSY",
        annotation_position="right",
    )
    fig.add_hline(
        y=5,
        line_dash="dot",
        line_color="#ef4444",
        line_width=1,
        annotation_text="5 min - ALERT",
        annotation_position="right",
    )
    fig.add_vline(x=now_ts.isoformat(), line_dash="dash", line_color="#f97316", line_width=1)
    fig.update_layout(
        title=f"Lane Strategy Wait Comparison ({DISPLAY_BUCKET_MIN} min view)",
        xaxis_title="Time",
        yaxis_title="Wait (min)",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_calibration_scenario_comparison(scenario_views: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    first_view = next((view for view in scenario_views.values() if view), None)
    now_ts = pd.to_datetime(first_view.get("current_clock_ts"), errors="coerce") if first_view else pd.NaT
    if pd.isna(now_ts):
        now_ts = _local_now_ts()

    palette = [
        "#38bdf8",
        "#a78bfa",
        "#22c55e",
        "#f59e0b",
        "#ef4444",
        "#f472b6",
    ]

    for idx, (scenario_name, view) in enumerate(scenario_views.items()):
        if not view:
            continue
        wait_agg = future_wait_display_frame(view)
        if wait_agg.empty:
            continue
        fig.add_trace(go.Scatter(
            x=wait_agg["ds"],
            y=wait_agg["wait_min"],
            mode="lines",
            name=scenario_name,
            line=dict(
                color=palette[idx % len(palette)],
                width=3 if idx == 0 else 2.2,
                dash="solid" if idx < 2 else "dot",
                shape="spline",
                smoothing=0.55,
            ),
            hovertemplate=f"{scenario_name}<br>%{{x|%H:%M}}: %{{y:.2f}} min<extra></extra>",
        ))

    frames = [future_wait_display_frame(view) for view in scenario_views.values() if view]
    _add_open_hours_overlay(fig, *frames)
    fig.add_hline(
        y=2,
        line_dash="dot",
        line_color="#f97316",
        line_width=1,
        annotation_text="2 min - BUSY",
        annotation_position="right",
    )
    fig.add_hline(
        y=5,
        line_dash="dot",
        line_color="#ef4444",
        line_width=1,
        annotation_text="5 min - ALERT",
        annotation_position="right",
    )
    fig.add_vline(x=now_ts.isoformat(), line_dash="dash", line_color="#f97316", line_width=1)
    fig.update_layout(
        title=f"Calibration Scenario Wait Comparison ({DISPLAY_BUCKET_MIN} min view)",
        xaxis_title="Time",
        yaxis_title="Wait (min)",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_recent_window_validation(validation_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if validation_frame.empty:
        fig.update_layout(
            title="Recent Window Validation Comparison",
            plot_bgcolor=DARK,
            paper_bgcolor=DARK,
            font=dict(color="#e2e8f0"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    frame = validation_frame.copy()
    for col in ["Current MAE", "Suggested MAE", "Current factor", "Suggested factor"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["Window label"] = frame["Window (days)"].astype(str) + "d"

    fig.add_trace(go.Bar(
        x=frame["Window label"],
        y=frame["Current MAE"],
        name="Current MAE",
        marker_color="#64748b",
        opacity=0.8,
        hovertemplate="%{x}<br>current MAE=%{y:.2f} min<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=frame["Window label"],
        y=frame["Suggested MAE"],
        name="Suggested MAE",
        marker_color="#22c55e",
        opacity=0.85,
        hovertemplate="%{x}<br>suggested MAE=%{y:.2f} min<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["Window label"],
        y=frame["Current factor"],
        mode="lines+markers",
        name="Current factor",
        line=dict(color="#f59e0b", width=2.0),
        marker=dict(size=6),
        yaxis="y2",
        hovertemplate="%{x}<br>current factor=%{y:.2f}x<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["Window label"],
        y=frame["Suggested factor"],
        mode="lines+markers",
        name="Suggested factor",
        line=dict(color="#38bdf8", width=2.2),
        marker=dict(size=6),
        yaxis="y2",
        hovertemplate="%{x}<br>suggested factor=%{y:.2f}x<extra></extra>",
    ))

    fig.add_hline(
        y=1,
        line_dash="dot",
        line_color="#22c55e",
        line_width=1,
        yref="y2",
        annotation_text="1.0x target",
        annotation_position="right",
    )
    fig.update_layout(
        title="Recent Window Validation Comparison",
        xaxis_title="History window",
        yaxis_title="MAE (min)",
        yaxis2=dict(
            title="Median factor (x)",
            overlaying="y",
            side="right",
            rangemode="tozero",
            gridcolor="rgba(0,0,0,0)",
        ),
        barmode="group",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero")
    return fig
