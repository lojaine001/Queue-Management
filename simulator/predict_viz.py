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
)

# ── Chart builders ────────────────────────────────────────────────────────────

DARK = "#0f172a"
GRID = "#1e293b"


def fig_training_data(result: dict,
                       live_actuals: "pd.DataFrame | None" = None) -> go.Figure:
    """Arrival counts coloured by source (REAL vs SIM).

    live_actuals — today's fresh DB query (bucket, entry_count).  When provided,
    today's REAL rows in df_arrivals are replaced with it so the chart always
    extends to the current minute regardless of when training last ran.
    """
    df = result["df_arrivals"].copy()

    if live_actuals is not None and not live_actuals.empty:
        # Drop today's cached rows and substitute the live query with proper source tags.
        today = pd.Timestamp.now().normalize()
        df = df[df["bucket"] < today]
        live = live_actuals.copy()
        df = pd.concat([df, live], ignore_index=True)

    fig = go.Figure()
    for src, color in [("REAL", "#38bdf8"), ("SIM", "#a78bfa")]:
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        agg = sub.groupby("bucket")["entry_count"].sum().reset_index()
        alpha = "rgba(56,189,248,0.10)" if src == "REAL" else "rgba(167,139,250,0.10)"
        fig.add_trace(go.Scatter(
            x=agg["bucket"], y=agg["entry_count"],
            mode="lines", name=src, line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=alpha,
        ))
    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316", line_width=1)
    fig.update_layout(
        title=f"Training Data — arrivals / {BUCKET_MINUTES} min",
        xaxis_title="Time", yaxis_title="Count",
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
    fc     = result["forecast"]

    # Actuals: prefer live query; fall back to today's slice of training set
    today_date = pd.Timestamp.now().normalize()
    if live_actuals is not None and not live_actuals.empty:
        hist = (
            live_actuals.groupby("bucket")["entry_count"]
            .sum()
            .reset_index()
            .rename(columns={"bucket": "ds", "entry_count": "y"})
        )
    else:
        hist = result["df_combined"][result["df_combined"]["ds"] >= today_date]
    insamp = result["df_insample"]   # already scoped to today in the pipeline

    fig = go.Figure()
    # Historical actuals
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["y"],
        mode="markers+lines", name="Actual (train)",
        marker=dict(size=4, color="#38bdf8"),
        line=dict(color="#38bdf8", width=1),
    ))
    # In-sample fit
    fig.add_trace(go.Scatter(
        x=insamp["ds"], y=insamp["yhat"],
        mode="lines", name="Prophet fit",
        line=dict(color="#f472b6", width=1, dash="dot"),
    ))
    # Confidence band
    ds_fwd = fc["ds"].tolist()
    ds_rev = fc["ds"].tolist()[::-1]
    yu = fc["yhat_upper"].tolist()
    yl = fc["yhat_lower"].tolist()[::-1]
    fig.add_trace(go.Scatter(
        x=ds_fwd + ds_rev,
        y=yu + yl,
        fill="toself", fillcolor="rgba(167,139,250,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="80% CI",
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=fc["ds"], y=fc["yhat"],
        mode="lines+markers", name="Forecast (next 60 min)",
        line=dict(color="#a78bfa", width=2),
        marker=dict(size=5),
    ))
    # Now line
    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316")

    fig.update_layout(
        title="Prophet — Today's Forecast",
        xaxis_title="Time", yaxis_title=f"Arrivals / {BUCKET_MINUTES} min",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_prophet_components(result: dict) -> go.Figure:
    """Prophet daily + weekly seasonality components."""
    comp = result["comp_df"]

    fig = sp.make_subplots(rows=1, cols=2,
                           subplot_titles=["Daily Pattern", "Weekly Pattern"])

    # Daily (0–24h pivot)
    daily_df = comp.copy()
    daily_df["hour"] = daily_df["ds"].dt.hour + daily_df["ds"].dt.minute / 60
    daily_agg = daily_df.groupby("hour")["daily"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=daily_agg["hour"], y=daily_agg["daily"],
        mode="lines", line=dict(color="#38bdf8", width=2), name="Daily",
    ), row=1, col=1)

    # Weekly (day-of-week pivot)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_df = comp.copy()
    weekly_df["dow"] = weekly_df["ds"].dt.dayofweek
    weekly_agg = weekly_df.groupby("dow")["weekly"].mean().reset_index()
    fig.add_trace(go.Bar(
        x=[day_names[d] for d in weekly_agg["dow"]],
        y=weekly_agg["weekly"],
        marker_color="#a78bfa", name="Weekly",
    ), row=1, col=2)

    fig.update_layout(
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_wait_estimates(result: dict) -> go.Figure:
    """Wait time: observed history (snapshots) + predicted remainder of day.

    Left of the orange 'now' line  → actual measured wait from queue snapshots.
    Right of the 'now' line        → model-predicted wait derived from Prophet
                                     arrival forecast + current queue backlog.
    """
    waits = result["wait_estimates"]
    df_snap = result["df_snapshots"]

    fig = go.Figure()

    # ── Historical wait from snapshots ────────────────────────────────────────
    if not df_snap.empty and "avg_dwell_sec" in df_snap.columns:
        # Use avg_dwell_sec / 60 as a proxy for observed service wait
        today = pd.Timestamp.now().normalize()
        hist = df_snap[df_snap["timestamp"] >= today].copy()
        if not hist.empty:
            hist_wait = hist["avg_dwell_sec"] / 60
            fig.add_trace(go.Scatter(
                x=hist["timestamp"], y=hist_wait,
                mode="lines", name="Observed dwell (min)",
                line=dict(color="#38bdf8", width=1.5),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
            ))

    # ── Predicted wait ────────────────────────────────────────────────────────
    if waits:
        pred_x = [w["ds"] for w in waits]
        pred_y = [w["wait_min"] for w in waits]

        # Confidence-style band (±20 % around the estimate)
        upper = [round(v * 1.2, 1) for v in pred_y]
        lower = [round(v * 0.8, 1) for v in pred_y]
        fig.add_trace(go.Scatter(
            x=pred_x + pred_x[::-1],
            y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(249,115,22,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="±20% band",
        ))
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode="lines+markers", name="Predicted wait",
            line=dict(color="#f97316", width=2),
            marker=dict(size=4),
        ))

    # ── Reference lines ───────────────────────────────────────────────────────
    fig.add_hline(y=5,  line_dash="dot", line_color="#f97316", line_width=1,
                  annotation_text="5 min — BUSY",  annotation_position="right")
    fig.add_hline(y=10, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="10 min — ALERT", annotation_position="right")
    fig.add_vline(x=datetime.now().isoformat(), line_dash="dash",
                  line_color="#f97316", line_width=1)

    fig.update_layout(
        title="Queue Wait — Today",
        xaxis_title="Time", yaxis_title="Wait (min)",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(rangemode="nonnegative"),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def fig_lane_scenarios(result: dict) -> go.Figure:
    """Wait time forecast for 1–5 open lanes — full planning view.

    Two panels sharing the x-axis (forecast time):

    Top — wait time curves for each lane count.
          Current active lanes drawn with a thicker solid line.
          Other lanes drawn as thinner dashed lines.
          BUSY (5 min) and ALERT (10 min) reference lines.

    Bottom — arrival forecast (yhat) for context: shows when demand
              spikes so the manager can decide when to open/close lanes.
    """
    scenarios    = result.get("lane_scenarios", {})
    active_lanes = result.get("active_lanes", 2)
    forecast     = result.get("forecast", pd.DataFrame())

    LANE_COLORS = {
        1: "#ef4444",   # red   — worst
        2: "#f97316",   # orange
        3: "#fbbf24",   # yellow
        4: "#4ade80",   # green
        5: "#38bdf8",   # blue  — best
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
    fig.add_hline(y=5,  line_dash="dot", line_color="#f97316", line_width=1,
                  annotation_text="5 min — BUSY",  annotation_position="right",
                  row=1, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="10 min — ALERT", annotation_position="right",
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


def fig_queue_history(result: dict) -> go.Figure:
    """Queue depth + dwell — two stacked subplots, shared x-axis, full width.

    Top panel   — queue count from snapshots (filled area) overlaid with
                  max_lane_depth per bucket from entrance_events (dotted markers).
                  Both are "people" on the same axis — no dual-axis confusion.
    Bottom panel — avg dwell (min) from snapshots as a smooth filled area.
    """
    df     = result["df_snapshots"]
    df_arr = result.get("df_arrivals", pd.DataFrame())

    # Derive per-bucket lane depth (REAL only, today)
    today = pd.Timestamp.now().normalize()
    lane_depth: "pd.DataFrame | None" = None
    if not df_arr.empty and "max_lane_depth" in df_arr.columns:
        src_mask = (
            df_arr["source"] != "SIM"
            if "source" in df_arr.columns
            else pd.Series(True, index=df_arr.index)
        )
        real_arr = df_arr[src_mask & (df_arr["bucket"] >= today)].copy()
        if not real_arr.empty:
            lane_depth = (
                real_arr.groupby("bucket")["max_lane_depth"]
                .max()
                .reset_index()
            )

    has_snaps = not df.empty
    has_dwell = has_snaps and "avg_dwell_sec" in df.columns

    if not has_snaps and lane_depth is None:
        return go.Figure().update_layout(
            title="No queue data", paper_bgcolor=DARK,
            plot_bgcolor=DARK, font=dict(color="#e2e8f0"),
        )

    # Two rows if we have dwell data, otherwise single row
    if has_dwell:
        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=["Queue depth (people)", "Avg dwell (min)"],
            vertical_spacing=0.10,
            row_heights=[0.65, 0.35],
        )
    else:
        fig = sp.make_subplots(rows=1, cols=1,
                               subplot_titles=["Queue depth (people)"])

    # ── Top: snapshot queue count ─────────────────────────────────────────────
    if has_snaps:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["queue_count"],
            mode="lines", fill="tozeroy",
            line=dict(color="#f97316", width=2),
            fillcolor="rgba(249,115,22,0.15)",
            name="Queue (snapshot)",
        ), row=1, col=1)

    # ── Top: lane depth from entrance events ──────────────────────────────────
    if lane_depth is not None and not lane_depth.empty:
        fig.add_trace(go.Scatter(
            x=lane_depth["bucket"], y=lane_depth["max_lane_depth"],
            mode="lines+markers",
            line=dict(color="#fbbf24", width=1.5, dash="dot"),
            marker=dict(size=5, symbol="circle-open"),
            name="Max lane depth (exits)",
        ), row=1, col=1)

    # ── Bottom: avg dwell ─────────────────────────────────────────────────────
    if has_dwell:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["avg_dwell_sec"] / 60,
            mode="lines", fill="tozeroy",
            line=dict(color="#a78bfa", width=1.5),
            fillcolor="rgba(167,139,250,0.12)",
            name="Avg dwell (min)",
        ), row=2, col=1)

    fig.update_layout(
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor=GRID, orientation="h", x=0.5, xanchor="center", y=-0.04),
        margin=dict(l=10, r=10, t=45, b=10),
        height=420,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="People", row=1, col=1)
    if has_dwell:
        fig.update_yaxes(title_text="Min", row=2, col=1)
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
