"""Separate Streamlit dashboard for hybrid REAL-only wait forecasting.

Run with:
    streamlit run simulator/app_hybrid_wait.py
"""

from __future__ import annotations

import os
import sys
import inspect
from time import perf_counter
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import simulator.hybrid_wait_viz as hybrid_viz
import simulator.predict_viz as predict_viz
from prediction.core import WAIT_15M_INDEX, WAIT_30M_INDEX
from prediction import hybrid_wait


st.set_page_config(layout="wide", page_title="Hybrid Wait Forecast")


def _phase_log(message: str, *, started_at: float | None = None, status_box=None) -> None:
    now_txt = datetime.now().strftime("%H:%M:%S")
    if started_at is not None:
        elapsed = perf_counter() - started_at
        message = f"{message} ({elapsed:.2f}s)"
    print(f"[HYBRID APP {now_txt}] {message}", flush=True)
    if status_box is not None:
        status_box.info(message)


def _hybrid_supports_calibration() -> bool:
    params = inspect.signature(hybrid_wait.build_hybrid_wait_view).parameters
    return all(
        key in params
        for key in ["inflow_multiplier", "service_multiplier", "lane_multiplier"]
    )


@st.cache_data(show_spinner="Loading REAL base forecast...", ttl=300)
def _load_base_result(
    days: int,
    use_bootstrap: bool,
    data_since_iso: str | None,
) -> dict:
    data_since = pd.Timestamp(data_since_iso).to_pydatetime() if data_since_iso else None
    return hybrid_wait.load_base_real_prediction(
        days=days,
        use_bootstrap=use_bootstrap,
        data_since=data_since,
    )


def _fmt_wait(value: float | None) -> str:
    return f"{float(value):.2f} min" if value is not None else "—"


def _fmt_num(value: float | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.2f}{suffix}"


def _backtest_metrics(comp: pd.DataFrame) -> dict[str, float | None]:
    if comp.empty:
        return {"median_factor": None, "mae": None, "p90_abs_error": None}

    frame = comp.copy()
    frame["observed_wait_min"] = pd.to_numeric(frame["observed_wait_min"], errors="coerce")
    frame["estimated_wait_min"] = pd.to_numeric(frame["estimated_wait_min"], errors="coerce")
    frame = frame.dropna(subset=["observed_wait_min", "estimated_wait_min"])
    if frame.empty:
        return {"median_factor": None, "mae": None, "p90_abs_error": None}

    frame["abs_error"] = (frame["estimated_wait_min"] - frame["observed_wait_min"]).abs()
    valid_factor = frame[frame["observed_wait_min"] > 0].copy()
    median_factor = float(valid_factor["estimated_wait_min"].div(valid_factor["observed_wait_min"]).median()) if not valid_factor.empty else None
    mae = float(frame["abs_error"].mean()) if not frame.empty else None
    p90 = float(frame["abs_error"].quantile(0.90)) if not frame.empty else None
    return {"median_factor": median_factor, "mae": mae, "p90_abs_error": p90}


def _backtest_driver_summary(comp: pd.DataFrame) -> dict[str, object]:
    if comp.empty:
        return {
            "overestimate_share": None,
            "median_service_ratio": None,
            "median_lane_ratio": None,
            "worst_time_label": None,
        }

    frame = comp.copy()
    for col in ["factor", "service_ratio", "lane_ratio", "wait_delta"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    valid_factor = frame["factor"].dropna() if "factor" in frame.columns else pd.Series(dtype=float)
    overestimate_share = float((valid_factor > 1.25).mean() * 100.0) if not valid_factor.empty else None
    median_service_ratio = float(frame["service_ratio"].dropna().median()) if "service_ratio" in frame.columns and not frame["service_ratio"].dropna().empty else None
    median_lane_ratio = float(frame["lane_ratio"].dropna().median()) if "lane_ratio" in frame.columns and not frame["lane_ratio"].dropna().empty else None

    worst_time_label = None
    factor_profile = hybrid_viz.backtest_factor_profile_frame({"training_wait_comparison": comp})
    if not factor_profile.empty:
        sortable = factor_profile.dropna(subset=["median_factor"]).sort_values(
            ["median_factor", "mean_wait_delta", "bucket_count"],
            ascending=[False, False, False],
        )
        if not sortable.empty:
            worst_row = sortable.iloc[0]
            worst_time_label = f"{worst_row['label']} ({worst_row['median_factor']:.2f}x)"

    return {
        "overestimate_share": overestimate_share,
        "median_service_ratio": median_service_ratio,
        "median_lane_ratio": median_lane_ratio,
        "worst_time_label": worst_time_label,
    }


def _root_cause_summary(result: dict) -> pd.DataFrame:
    root = hybrid_viz.backtest_root_cause_frame(result)
    if root.empty:
        return pd.DataFrame(columns=["Root cause", "Buckets", "Share %"])
    summary = (
        root.groupby("root_cause", as_index=False)
        .size()
        .rename(columns={"root_cause": "Root cause", "size": "Buckets"})
        .sort_values("Buckets", ascending=False)
        .reset_index(drop=True)
    )
    total = int(summary["Buckets"].sum()) if not summary.empty else 0
    summary["Share %"] = ((summary["Buckets"] / total) * 100.0).round(1) if total > 0 else 0.0
    return summary


def _tuning_guidance(result: dict, driver_summary: dict[str, object], root_summary: pd.DataFrame) -> list[dict[str, str]]:
    guidance: list[dict[str, str]] = []
    if root_summary.empty:
        return guidance

    lead = root_summary.iloc[0]
    lead_cause = str(lead["Root cause"])
    lead_share = float(lead["Share %"])
    worst_time = str(driver_summary.get("worst_time_label") or "—")
    service_ratio = driver_summary.get("median_service_ratio")
    lane_ratio = driver_summary.get("median_lane_ratio")
    overestimate_share = driver_summary.get("overestimate_share")

    if lead_cause == "Service-led":
        guidance.append(
            {
                "Priority": "1",
                "Focus": "Tighten service estimation",
                "Why": f"Service-led inflation is the largest bucket group at {lead_share:.1f}%.",
                "Next move": "Reduce reliance on fallback service minutes, favor recent REAL service medians, and inspect service drift around the worst time bucket.",
            }
        )
    elif lead_cause == "Lane-led":
        guidance.append(
            {
                "Priority": "1",
                "Focus": "Tighten lane policy",
                "Why": f"Lane-led inflation is the largest bucket group at {lead_share:.1f}%.",
                "Next move": "Trust fresh snapshot lanes more aggressively, soften historical lane blending, and compare snapshot vs manual strategies around the worst time bucket.",
            }
        )
    elif lead_cause == "Service + lanes":
        guidance.append(
            {
                "Priority": "1",
                "Focus": "Calibrate service and lanes together",
                "Why": f"Combined service-plus-lanes inflation leads at {lead_share:.1f}%.",
                "Next move": "Reduce simultaneous upward blending of service and lanes, and inspect whether historical profiles are too influential during the worst time bucket.",
            }
        )
    elif lead_cause == "Other inflation":
        guidance.append(
            {
                "Priority": "1",
                "Focus": "Inspect inflow and backlog assumptions",
                "Why": f"Inflation is not mainly explained by service or lanes in {lead_share:.1f}% of dominant buckets.",
                "Next move": "Review blended inflow and backlog carry-forward around the worst time bucket, especially when recent REAL throughput and historical inflow diverge.",
            }
        )

    if service_ratio is not None and float(service_ratio) >= 1.10:
        guidance.append(
            {
                "Priority": "2",
                "Focus": "Service profile is running hot",
                "Why": f"Median service ratio is {float(service_ratio):.2f}x.",
                "Next move": "Bias the hybrid model more toward recent REAL service time and less toward slower historical service during short horizons.",
            }
        )

    if lane_ratio is not None and float(lane_ratio) <= 0.95:
        guidance.append(
            {
                "Priority": "2",
                "Focus": "Lane profile is likely too thin",
                "Why": f"Median lane ratio is {float(lane_ratio):.2f}x estimated/observed.",
                "Next move": "Keep snapshot-preferred lane selection, reduce downward lane blending from history, and test a modest lane uplift in the calibration sandbox.",
            }
        )

    if overestimate_share is not None and float(overestimate_share) >= 40.0:
        guidance.append(
            {
                "Priority": "3",
                "Focus": "Inflation is broad, not isolated",
                "Why": f"{float(overestimate_share):.1f}% of backtest buckets overestimate by more than 1.25x.",
                "Next move": f"Prioritize fixing the dominant cause first, then re-check the worst time bucket at {worst_time} before changing more than one model assumption at once.",
            }
        )

    if not guidance:
        guidance.append(
            {
                "Priority": "1",
                "Focus": "Model is relatively aligned",
                "Why": "The current backtest does not show a single dominant inflation mode.",
                "Next move": "Keep the hybrid structure stable and tune only the worst time-of-day pocket if needed.",
            }
        )

    return guidance


def _suggested_calibration(
    metrics: dict[str, float | None],
    driver_summary: dict[str, object],
    root_summary: pd.DataFrame,
) -> dict[str, object]:
    service_ratio = driver_summary.get("median_service_ratio")
    lane_ratio = driver_summary.get("median_lane_ratio")
    median_factor = metrics.get("median_factor")
    dominant_cause = str(root_summary.iloc[0]["Root cause"]) if not root_summary.empty else "Aligned"

    inflow = 1.0
    service = 1.0
    lanes = 1.0

    if service_ratio is not None and dominant_cause in {"Service-led", "Service + lanes"}:
        service = min(max(1.0 / max(float(service_ratio), 0.5), 0.75), 1.0)
    if lane_ratio is not None and dominant_cause in {"Lane-led", "Service + lanes"}:
        lanes = min(max(1.0 / max(float(lane_ratio), 0.5), 1.0), 1.35)
    if median_factor is not None and dominant_cause in {"Other inflation", "Aligned"} and float(median_factor) > 1.10:
        inflow = min(max(1.0 / float(median_factor), 0.80), 1.0)

    reason_parts: list[str] = []
    if service != 1.0:
        reason_parts.append("service reduction")
    if lanes != 1.0:
        reason_parts.append("lane uplift")
    if inflow != 1.0:
        reason_parts.append("inflow reduction")
    reason = ", ".join(reason_parts) if reason_parts else "no strong correction suggested"

    return {
        "dominant_cause": dominant_cause,
        "suggested_inflow_multiplier": round(inflow, 2),
        "suggested_service_multiplier": round(service, 2),
        "suggested_lane_multiplier": round(lanes, 2),
        "reason": reason,
    }


def _scenario_summary_row(name: str, view: dict, *, inflow: float, service: float, lanes: float) -> dict[str, object]:
    training_vals = hybrid_viz.training_values_frame(view)
    metrics = _backtest_metrics(training_vals)
    return {
        "Scenario": name,
        "Inflow x": inflow,
        "Service x": service,
        "Lanes x": lanes,
        "Wait @ +15 min": view.get("wait_15m"),
        "Wait @ +30 min": view.get("wait_30m"),
        "Median factor": metrics.get("median_factor"),
        "MAE (min)": metrics.get("mae"),
        "Current backlog": view.get("current_waiting_backlog"),
        "Active lanes now": view.get("active_lanes_now"),
    }


def _rank_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ranked = frame.copy()
    numeric_cols = ["Wait @ +15 min", "Wait @ +30 min", "Median factor", "MAE (min)"]
    for col in numeric_cols:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
    ranked["Current backlog"] = pd.to_numeric(ranked.get("Current backlog"), errors="coerce")
    ranked["Active lanes now"] = pd.to_numeric(ranked.get("Active lanes now"), errors="coerce")

    ranked["factor_distance"] = (ranked["Median factor"] - 1.0).abs()
    ranked["underestimate_penalty"] = ((0.92 - ranked["Median factor"]).clip(lower=0.0) * 100.0).fillna(0.0)
    ranked["backlog_floor_penalty"] = (
        ((ranked["Current backlog"].fillna(0.0) - 2.0).clip(lower=0.0))
        * ((0.75 - ranked["Wait @ +15 min"].fillna(99.0)).clip(lower=0.0))
        * 8.0
    )
    ranked["score"] = (
        ranked["factor_distance"].fillna(99.0) * 100.0
        + ranked["underestimate_penalty"]
        + ranked["backlog_floor_penalty"]
        + ranked["MAE (min)"].fillna(99.0) * 10.0
        + ranked["Wait @ +15 min"].fillna(99.0)
    )
    ranked = ranked.sort_values(
        ["score", "factor_distance", "underestimate_penalty", "MAE (min)", "Wait @ +15 min"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return ranked


def _scenario_delta_frame(ranked: pd.DataFrame, baseline_name: str = "Current sandbox") -> pd.DataFrame:
    if ranked.empty or "Scenario" not in ranked.columns:
        return pd.DataFrame()

    baseline = ranked[ranked["Scenario"] == baseline_name]
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]

    delta = ranked.copy()
    numeric_cols = ["Wait @ +15 min", "Wait @ +30 min", "Median factor", "MAE (min)", "score"]
    for col in numeric_cols:
        delta[col] = pd.to_numeric(delta[col], errors="coerce")

    delta["Delta +15 (min)"] = delta["Wait @ +15 min"] - float(pd.to_numeric(base["Wait @ +15 min"], errors="coerce"))
    delta["Delta +30 (min)"] = delta["Wait @ +30 min"] - float(pd.to_numeric(base["Wait @ +30 min"], errors="coerce"))
    delta["Delta factor"] = delta["Median factor"] - float(pd.to_numeric(base["Median factor"], errors="coerce"))
    delta["Delta MAE (min)"] = delta["MAE (min)"] - float(pd.to_numeric(base["MAE (min)"], errors="coerce"))
    delta["Delta score"] = delta["score"] - float(pd.to_numeric(base["score"], errors="coerce"))
    return delta


def _best_scenario_reason(best_row: pd.Series, current_row: pd.Series | None) -> str:
    if current_row is None:
        return "This scenario ranks first on the combined calibration score."

    delta_factor = pd.to_numeric(pd.Series([best_row["Median factor"]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([current_row["Median factor"]]), errors="coerce").iloc[0]
    delta_mae = pd.to_numeric(pd.Series([best_row["MAE (min)"]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([current_row["MAE (min)"]]), errors="coerce").iloc[0]
    delta_wait = pd.to_numeric(pd.Series([best_row["Wait @ +15 min"]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([current_row["Wait @ +15 min"]]), errors="coerce").iloc[0]

    reasons: list[str] = []
    if pd.notna(delta_factor) and delta_factor < -0.02:
        reasons.append(f"improves median factor by {abs(delta_factor):.2f}x")
    if pd.notna(delta_mae) and delta_mae < -0.05:
        reasons.append(f"reduces MAE by {abs(delta_mae):.2f} min")
    if pd.notna(delta_wait) and delta_wait < -0.05:
        reasons.append(f"lowers +15 wait by {abs(delta_wait):.2f} min")
    if not reasons:
        reasons.append("edges out the current sandbox on the combined ranking score")
    return ", ".join(reasons)


@st.cache_data(show_spinner=False, ttl=300)
def _validation_window_row(
    *,
    window_days: int,
    use_bootstrap: bool,
    data_since_iso: str | None,
    current_weight_pct: int,
    auto_shift_history: bool,
    recent_window_min: int,
    lane_strategy: str,
    manual_active_lanes: int | None,
    applied_inflow_multiplier: float,
    applied_service_multiplier: float,
    applied_lane_multiplier: float,
    use_profile_calibration: bool,
    profile_strength: float,
) -> dict[str, object]:
    base = _load_base_result(window_days, use_bootstrap, data_since_iso)
    current_view = _build_strategy_view(
        base,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
        inflow_multiplier=applied_inflow_multiplier,
        service_multiplier=applied_service_multiplier,
        lane_multiplier=applied_lane_multiplier,
        use_profile_calibration=use_profile_calibration,
        profile_strength=profile_strength,
    )
    current_training = hybrid_viz.training_values_frame(current_view)
    current_metrics = _backtest_metrics(current_training)
    current_driver = _backtest_driver_summary(current_training)
    current_root = _root_cause_summary(current_view)
    suggested = _suggested_calibration(current_metrics, current_driver, current_root)

    scenario_rows = [
        _scenario_summary_row(
            "Current sandbox",
            current_view,
            inflow=applied_inflow_multiplier,
            service=applied_service_multiplier,
            lanes=applied_lane_multiplier,
        )
    ]

    suggested_view = None
    if any(
        abs(float(suggested[key]) - 1.0) > 1e-9
        for key in ["suggested_inflow_multiplier", "suggested_service_multiplier", "suggested_lane_multiplier"]
    ):
        suggested_view = _build_strategy_view(
            base,
            current_weight_pct=current_weight_pct,
            auto_shift_history=auto_shift_history,
            recent_window_min=recent_window_min,
            lane_strategy=lane_strategy,
            manual_active_lanes=manual_active_lanes,
            inflow_multiplier=float(suggested["suggested_inflow_multiplier"]),
            service_multiplier=float(suggested["suggested_service_multiplier"]),
            lane_multiplier=float(suggested["suggested_lane_multiplier"]),
            use_profile_calibration=use_profile_calibration,
            profile_strength=profile_strength,
        )
        scenario_rows.append(
            _scenario_summary_row(
                "Suggested target",
                suggested_view,
                inflow=float(suggested["suggested_inflow_multiplier"]),
                service=float(suggested["suggested_service_multiplier"]),
                lanes=float(suggested["suggested_lane_multiplier"]),
            )
        )

    scenario_frame = _rank_scenarios(pd.DataFrame(scenario_rows))
    best_row = scenario_frame.iloc[0] if not scenario_frame.empty else pd.Series(dtype=object)
    return {
        "Window (days)": window_days,
        "Current @ +15": current_view.get("wait_15m"),
        "Current factor": current_metrics.get("median_factor"),
        "Current MAE": current_metrics.get("mae"),
        "Dominant cause": str(current_root.iloc[0]["Root cause"]) if not current_root.empty else "—",
        "Suggested @ +15": suggested_view.get("wait_15m") if suggested_view is not None else None,
        "Suggested factor": _backtest_metrics(hybrid_viz.training_values_frame(suggested_view)).get("median_factor") if suggested_view is not None else None,
        "Suggested MAE": _backtest_metrics(hybrid_viz.training_values_frame(suggested_view)).get("mae") if suggested_view is not None else None,
        "Best scenario": best_row.get("Scenario") if not best_row.empty else "Current sandbox",
        "Best @ +15": best_row.get("Wait @ +15 min") if not best_row.empty else current_view.get("wait_15m"),
        "Best factor": best_row.get("Median factor") if not best_row.empty else current_metrics.get("median_factor"),
        "Best MAE": best_row.get("MAE (min)") if not best_row.empty else current_metrics.get("mae"),
    }


def _validation_stability_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "verdict": "No validation windows",
            "suggested_win_rate": None,
            "avg_factor_gain": None,
            "avg_mae_gain": None,
            "summary": "No recent-window validation is available.",
        }

    work = frame.copy()
    for col in [
        "Current factor",
        "Current MAE",
        "Suggested factor",
        "Suggested MAE",
        "Best factor",
        "Best MAE",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    suggested_wins = (work["Best scenario"] == "Suggested target").sum()
    total = len(work)
    suggested_win_rate = (suggested_wins / total) * 100.0 if total > 0 else None

    factor_gain = (work["Current factor"] - work["Suggested factor"]).dropna()
    mae_gain = (work["Current MAE"] - work["Suggested MAE"]).dropna()
    avg_factor_gain = float(factor_gain.mean()) if not factor_gain.empty else None
    avg_mae_gain = float(mae_gain.mean()) if not mae_gain.empty else None

    if suggested_win_rate is not None and suggested_win_rate >= 75.0:
        verdict = "Stable recommendation"
        summary = "The suggested calibration wins most recent windows and looks repeatable."
    elif suggested_win_rate is not None and suggested_win_rate >= 40.0:
        verdict = "Mixed recommendation"
        summary = "The suggested calibration helps in some windows but is not yet consistently dominant."
    else:
        verdict = "Unstable recommendation"
        summary = "The suggested calibration is not consistently winning across recent windows yet."

    return {
        "verdict": verdict,
        "suggested_win_rate": suggested_win_rate,
        "avg_factor_gain": avg_factor_gain,
        "avg_mae_gain": avg_mae_gain,
        "summary": summary,
    }


@st.cache_data(show_spinner=False, ttl=300)
def _profile_mode_window_row(
    *,
    window_days: int,
    use_bootstrap: bool,
    data_since_iso: str | None,
    current_weight_pct: int,
    auto_shift_history: bool,
    recent_window_min: int,
    lane_strategy: str,
    manual_active_lanes: int | None,
    applied_inflow_multiplier: float,
    applied_service_multiplier: float,
    applied_lane_multiplier: float,
    profile_strength: float,
) -> dict[str, object]:
    base = _load_base_result(window_days, use_bootstrap, data_since_iso)
    profile_on_view = _build_strategy_view(
        base,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
        inflow_multiplier=applied_inflow_multiplier,
        service_multiplier=applied_service_multiplier,
        lane_multiplier=applied_lane_multiplier,
        use_profile_calibration=True,
        profile_strength=profile_strength,
    )
    profile_off_view = _build_strategy_view(
        base,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
        inflow_multiplier=applied_inflow_multiplier,
        service_multiplier=applied_service_multiplier,
        lane_multiplier=applied_lane_multiplier,
        use_profile_calibration=False,
        profile_strength=0.0,
    )

    on_metrics = _backtest_metrics(hybrid_viz.training_values_frame(profile_on_view))
    off_metrics = _backtest_metrics(hybrid_viz.training_values_frame(profile_off_view))
    on_factor = on_metrics.get("median_factor")
    off_factor = off_metrics.get("median_factor")
    on_mae = on_metrics.get("mae")
    off_mae = off_metrics.get("mae")

    on_score = (
        abs(float(on_factor) - 1.0) * 100.0 + float(on_mae) * 10.0
        if on_factor is not None and on_mae is not None
        else float("inf")
    )
    off_score = (
        abs(float(off_factor) - 1.0) * 100.0 + float(off_mae) * 10.0
        if off_factor is not None and off_mae is not None
        else float("inf")
    )
    best_mode = "Profile on" if on_score <= off_score else "Profile off"

    return {
        "Window (days)": window_days,
        "Profile on @ +15": profile_on_view.get("wait_15m"),
        "Profile on factor": on_factor,
        "Profile on MAE": on_mae,
        "Profile off @ +15": profile_off_view.get("wait_15m"),
        "Profile off factor": off_factor,
        "Profile off MAE": off_mae,
        "Best mode": best_mode,
    }


def _profile_mode_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "verdict": "No profile comparison",
            "profile_on_win_rate": None,
            "avg_mae_gain": None,
            "avg_factor_gain": None,
            "summary": "No profile-mode comparison is available.",
        }

    work = frame.copy()
    for col in ["Profile on factor", "Profile on MAE", "Profile off factor", "Profile off MAE"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    profile_on_wins = (work["Best mode"] == "Profile on").sum()
    total = len(work)
    profile_on_win_rate = (profile_on_wins / total) * 100.0 if total > 0 else None
    avg_mae_gain = float((work["Profile off MAE"] - work["Profile on MAE"]).dropna().mean()) if not work.empty else None
    avg_factor_gain = float((work["Profile off factor"] - work["Profile on factor"]).dropna().mean()) if not work.empty else None

    if profile_on_win_rate is not None and profile_on_win_rate >= 75.0:
        verdict = "Keep profile calibration on"
        summary = "The automatic profile layer is consistently helping across recent windows."
    elif profile_on_win_rate is not None and profile_on_win_rate >= 40.0:
        verdict = "Profile calibration is mixed"
        summary = "The automatic profile layer helps in some windows but is not reliably better everywhere."
    else:
        verdict = "Prefer profile calibration off"
        summary = "The automatic profile layer is not consistently improving recent-window validation."

    return {
        "verdict": verdict,
        "profile_on_win_rate": profile_on_win_rate,
        "avg_mae_gain": avg_mae_gain,
        "avg_factor_gain": avg_factor_gain,
        "summary": summary,
    }


def _deployment_readiness(stability: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "verdict": "Not ready",
            "checks_passed": 0,
            "checks_total": 3,
            "summary": "No validation data is available yet.",
        }

    suggested_win_rate = stability.get("suggested_win_rate")
    avg_mae_gain = stability.get("avg_mae_gain")
    avg_factor_gain = stability.get("avg_factor_gain")

    checks = [
        bool(suggested_win_rate is not None and float(suggested_win_rate) >= 60.0),
        bool(avg_mae_gain is not None and float(avg_mae_gain) > 0.0),
        bool(avg_factor_gain is not None and float(avg_factor_gain) > 0.0),
    ]
    passed = int(sum(checks))

    if passed == 3:
        verdict = "Ready to trial"
        summary = "The suggested calibration is winning often enough and improving both error and factor on average."
    elif passed == 2:
        verdict = "Promising"
        summary = "The suggested calibration is improving the model, but it is not fully consistent across windows yet."
    else:
        verdict = "Not ready"
        summary = "The suggested calibration still needs more stability before it should be treated as the default choice."

    return {
        "verdict": verdict,
        "checks_passed": passed,
        "checks_total": len(checks),
        "summary": summary,
    }


def _operating_recommendation(
    *,
    best_row: pd.Series | None,
    current_row: pd.Series | None,
    readiness: dict[str, object] | None,
    profile_mode_summary: dict[str, object] | None,
    current_profile_enabled: bool,
) -> dict[str, str]:
    best_scenario = str(best_row["Scenario"]) if best_row is not None and not best_row.empty else "Current sandbox"
    readiness_verdict = str((readiness or {}).get("verdict") or "Not ready")
    profile_verdict = str((profile_mode_summary or {}).get("verdict") or "No profile comparison")

    if profile_verdict == "Keep profile calibration on":
        profile_mode = "On"
    elif profile_verdict == "Prefer profile calibration off":
        profile_mode = "Off"
    else:
        profile_mode = "On" if current_profile_enabled else "Off"

    if readiness_verdict == "Ready to trial" and profile_verdict in {"Keep profile calibration on", "No profile comparison"}:
        confidence = "High"
    elif readiness_verdict in {"Ready to trial", "Promising"} or profile_verdict == "Profile calibration is mixed":
        confidence = "Medium"
    else:
        confidence = "Low"

    reason_parts: list[str] = []
    if current_row is not None and best_row is not None and not current_row.empty and not best_row.empty:
        best_factor = pd.to_numeric(pd.Series([best_row.get("Median factor")]), errors="coerce").iloc[0]
        current_factor = pd.to_numeric(pd.Series([current_row.get("Median factor")]), errors="coerce").iloc[0]
        best_mae = pd.to_numeric(pd.Series([best_row.get("MAE (min)")]), errors="coerce").iloc[0]
        current_mae = pd.to_numeric(pd.Series([current_row.get("MAE (min)")]), errors="coerce").iloc[0]
        if pd.notna(best_factor) and pd.notna(current_factor) and best_factor < current_factor:
            reason_parts.append("better factor alignment")
        if pd.notna(best_mae) and pd.notna(current_mae) and best_mae < current_mae:
            reason_parts.append("lower backtest MAE")
    if profile_verdict == "Keep profile calibration on":
        reason_parts.append("profile calibration is stable across windows")
    elif profile_verdict == "Prefer profile calibration off":
        reason_parts.append("profile calibration is not consistently helping")
    elif profile_verdict == "Profile calibration is mixed":
        reason_parts.append("profile calibration effect is mixed")

    rationale = ", ".join(reason_parts) if reason_parts else "current evidence is limited"

    if confidence == "High":
        action = "Use this as the current best operating setup in the separate hybrid app and keep validating on fresh REAL windows."
    elif confidence == "Medium":
        action = "Use this as a working candidate, but keep watching recent-window validation before treating it as the default."
    else:
        action = "Treat this as exploratory only and keep tuning before relying on it."

    return {
        "scenario": best_scenario,
        "profile_mode": profile_mode,
        "confidence": confidence,
        "rationale": rationale,
        "action": action,
    }


def _operating_recommendation_trace(
    *,
    best_row: pd.Series | None,
    current_row: pd.Series | None,
    readiness: dict[str, object] | None,
    profile_mode_summary: dict[str, object] | None,
    operating_recommendation: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if best_row is not None and not best_row.empty:
        rows.append(
            {
                "Signal": "Best scenario",
                "Value": str(best_row.get("Scenario")),
                "Why it matters": "Top-ranked candidate from the current calibration scenario board.",
            }
        )
        rows.append(
            {
                "Signal": "Best scenario factor",
                "Value": _fmt_num(pd.to_numeric(pd.Series([best_row.get("Median factor")]), errors="coerce").iloc[0], "x"),
                "Why it matters": "Closer to 1.00x means the hybrid estimate is better aligned with the snapshot-derived proxy.",
            }
        )
        rows.append(
            {
                "Signal": "Best scenario MAE",
                "Value": _fmt_num(pd.to_numeric(pd.Series([best_row.get("MAE (min)")]), errors="coerce").iloc[0], " min"),
                "Why it matters": "Lower MAE means less absolute backtest error during the training-period comparison.",
            }
        )

    if current_row is not None and best_row is not None and not current_row.empty and not best_row.empty:
        best_factor = pd.to_numeric(pd.Series([best_row.get("Median factor")]), errors="coerce").iloc[0]
        current_factor = pd.to_numeric(pd.Series([current_row.get("Median factor")]), errors="coerce").iloc[0]
        best_mae = pd.to_numeric(pd.Series([best_row.get("MAE (min)")]), errors="coerce").iloc[0]
        current_mae = pd.to_numeric(pd.Series([current_row.get("MAE (min)")]), errors="coerce").iloc[0]
        rows.append(
            {
                "Signal": "Best vs current factor delta",
                "Value": _fmt_num(best_factor - current_factor if pd.notna(best_factor) and pd.notna(current_factor) else None, "x"),
                "Why it matters": "Negative is better because it moves the factor closer to 1.00x than the current sandbox.",
            }
        )
        rows.append(
            {
                "Signal": "Best vs current MAE delta",
                "Value": _fmt_num(best_mae - current_mae if pd.notna(best_mae) and pd.notna(current_mae) else None, " min"),
                "Why it matters": "Negative is better because it reduces backtest error versus the current sandbox.",
            }
        )

    if readiness is not None:
        rows.append(
            {
                "Signal": "Recent-window readiness",
                "Value": str(readiness.get("verdict")),
                "Why it matters": "Summarizes whether the suggested calibration is stable enough across recent history slices.",
            }
        )
        rows.append(
            {
                "Signal": "Readiness checks",
                "Value": f"{readiness.get('checks_passed', 0)}/{readiness.get('checks_total', 0)}",
                "Why it matters": "Counts how many recent-window stability checks are currently passing.",
            }
        )

    if profile_mode_summary is not None:
        rows.append(
            {
                "Signal": "Profile mode verdict",
                "Value": str(profile_mode_summary.get("verdict")),
                "Why it matters": "Determines whether the automatic backtest-derived profile layer should stay on or off by default.",
            }
        )
        rows.append(
            {
                "Signal": "Profile-on win rate",
                "Value": _fmt_num(profile_mode_summary.get("profile_on_win_rate"), "%"),
                "Why it matters": "Higher means the profile-calibration layer is winning more recent validation windows.",
            }
        )

    rows.append(
        {
            "Signal": "Final recommendation",
            "Value": f"{operating_recommendation['scenario']} / profile {operating_recommendation['profile_mode']} / confidence {operating_recommendation['confidence']}",
            "Why it matters": "This is the combined operating choice synthesized from scenario ranking, window validation, and profile-mode validation.",
        }
    )

    return pd.DataFrame(rows)


def _snapshot_age_text(snapshot_ts: object) -> str:
    ts = pd.to_datetime(snapshot_ts, errors="coerce")
    if pd.isna(ts):
        return "unknown"
    delta = pd.Timestamp(datetime.now().astimezone()).tz_localize(None) - ts
    minutes = max(int(delta.total_seconds() // 60), 0)
    return f"{minutes} min old"


def _hybrid_open_flag(ts: pd.Timestamp | datetime | None) -> bool:
    if ts is None:
        return False
    stamp = pd.Timestamp(ts)
    if pd.isna(stamp):
        return False
    total_minutes = int(stamp.hour) * 60 + int(stamp.minute)
    return (9 * 60) <= total_minutes < (20 * 60)


def _boundary_debug_frame(anchor_ts: object) -> pd.DataFrame:
    anchor = pd.to_datetime(anchor_ts, errors="coerce")
    if pd.isna(anchor):
        anchor = pd.Timestamp.now()
    day = anchor.normalize()
    checkpoints = [
        ("08:55", day + pd.Timedelta(hours=8, minutes=55)),
        ("09:00", day + pd.Timedelta(hours=9, minutes=0)),
        ("19:57", day + pd.Timedelta(hours=19, minutes=57)),
        ("20:00", day + pd.Timedelta(hours=20, minutes=0)),
        ("20:15", day + pd.Timedelta(hours=20, minutes=15)),
    ]
    rows = []
    for label, ts in checkpoints:
        is_open_now = _hybrid_open_flag(ts)
        rows.append(
            {
                "Time": label,
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "Open?": "OPEN" if is_open_now else "CLOSED",
                "Forced lanes": "1" if is_open_now else "0",
                "Forced inflow": "modeled" if is_open_now else "0",
                "Forced wait": "modeled" if is_open_now else "0",
            }
        )
    return pd.DataFrame(rows).astype(str)


def _closed_hours_summary_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Component": "Active lanes now",
                "Current value": str(result.get("active_lanes_now", "—")),
                "Closed-hours rule": "Forced to 0 outside 09:00-20:00",
            },
            {
                "Component": "Waiting backlog now",
                "Current value": str(result.get("current_waiting_backlog", "—")),
                "Closed-hours rule": "Forced to 0 outside 09:00-20:00",
            },
            {
                "Component": "Recent checkout rate",
                "Current value": _fmt_num(result.get("recent_checkout_rate"), f" / {predict_viz.BUCKET_MINUTES} min"),
                "Closed-hours rule": "Forced to 0 outside 09:00-20:00",
            },
            {
                "Component": "Recent service median",
                "Current value": _fmt_num(result.get("service_median_min"), " min"),
                "Closed-hours rule": "Ignored outside 09:00-20:00",
            },
            {
                "Component": "Forecast inflow buckets",
                "Current value": "modeled" if result.get("anchor_is_open") else "0",
                "Closed-hours rule": "Forced to 0 outside 09:00-20:00",
            },
            {
                "Component": "Queue rollforward",
                "Current value": "modeled" if result.get("anchor_is_open") else "0",
                "Closed-hours rule": "Forced to 0 outside 09:00-20:00",
            },
            {
                "Component": "Wait @ +15 min",
                "Current value": _fmt_wait(result.get("wait_15m")),
                "Closed-hours rule": "Forced to 0/empty outside 09:00-20:00",
            },
            {
                "Component": "Wait @ +30 min",
                "Current value": _fmt_wait(result.get("wait_30m")),
                "Closed-hours rule": "Forced to 0/empty outside 09:00-20:00",
            },
        ]
    ).astype(str)


@st.cache_data(show_spinner=False, ttl=300)
def _build_strategy_view(
    base_result: dict,
    *,
    current_weight_pct: int,
    auto_shift_history: bool,
    recent_window_min: int,
    lane_strategy: str,
    manual_active_lanes: int | None = None,
    inflow_multiplier: float = 1.0,
    service_multiplier: float = 1.0,
    lane_multiplier: float = 1.0,
    use_profile_calibration: bool = True,
    profile_strength: float = 1.0,
) -> dict:
    build_fn = hybrid_wait.build_hybrid_wait_view
    params = inspect.signature(build_fn).parameters
    kwargs = {
        "current_weight_pct": current_weight_pct,
        "auto_shift_history": auto_shift_history,
        "recent_window_min": recent_window_min,
        "lane_strategy": lane_strategy,
        "manual_active_lanes": manual_active_lanes,
    }
    if "inflow_multiplier" in params:
        kwargs["inflow_multiplier"] = inflow_multiplier
    if "service_multiplier" in params:
        kwargs["service_multiplier"] = service_multiplier
    if "lane_multiplier" in params:
        kwargs["lane_multiplier"] = lane_multiplier
    if "use_profile_calibration" in params:
        kwargs["use_profile_calibration"] = use_profile_calibration
    if "profile_strength" in params:
        kwargs["profile_strength"] = profile_strength
    return build_fn(base_result, **kwargs)


def _strategy_driver_rows(strategy_views: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy_name in ["auto", "snapshot", "inferred", "manual"]:
        view = strategy_views.get(strategy_name)
        if not view:
            continue
        future_vals = hybrid_viz.future_values_frame(view)
        future_only = future_vals.iloc[1:].reset_index(drop=True) if len(future_vals) > 1 else future_vals.copy()
        row: dict[str, object] = {
            "Strategy": strategy_name,
            "Lane source": view.get("active_lane_source"),
            "Active lanes now": view.get("active_lanes_now"),
        }
        for label, idx in [("15", WAIT_15M_INDEX), ("30", WAIT_30M_INDEX)]:
            if len(future_only) > idx:
                point = future_only.iloc[idx]
                row[f"Wait @ +{label} min"] = point.get("wait_min")
                row[f"Lanes @ +{label}"] = point.get("lanes_used")
                row[f"Service @ +{label}"] = point.get("service_min_used")
                row[f"Inflow @ +{label}"] = point.get("blended_checkout_rate")
                row[f"Queue @ +{label}"] = point.get("queue_after")
            else:
                row[f"Wait @ +{label} min"] = None
                row[f"Lanes @ +{label}"] = None
                row[f"Service @ +{label}"] = None
                row[f"Inflow @ +{label}"] = None
                row[f"Queue @ +{label}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _driver_attribution_frame(result: dict) -> pd.DataFrame:
    future_vals = hybrid_viz.future_values_frame(result)
    future_only = future_vals.iloc[1:].reset_index(drop=True) if len(future_vals) > 1 else future_vals.copy()
    fallback_service = pd.to_numeric(pd.Series([result.get("service_fallback_min")]), errors="coerce").fillna(0.0).iloc[0]
    rows: list[dict[str, object]] = []
    for label, idx in [("15", WAIT_15M_INDEX), ("30", WAIT_30M_INDEX)]:
        if len(future_only) <= idx:
            rows.append(
                {
                    "Horizon": f"+{label} min",
                    "Wait (min)": None,
                    "Lanes": None,
                    "Service (min)": None,
                    "Service capacity": None,
                    "Blended inflow": None,
                    "Net queue change": None,
                    "Primary driver": "No forecast bucket",
                }
            )
            continue
        point = future_only.iloc[idx]
        inflow = pd.to_numeric(pd.Series([point.get("blended_checkout_rate")]), errors="coerce").fillna(0.0).iloc[0]
        capacity = pd.to_numeric(pd.Series([point.get("service_capacity")]), errors="coerce").fillna(0.0).iloc[0]
        net_change = pd.to_numeric(pd.Series([point.get("net_queue_change")]), errors="coerce").fillna(0.0).iloc[0]
        lanes = pd.to_numeric(pd.Series([point.get("lanes_used")]), errors="coerce").fillna(0.0).iloc[0]
        service = pd.to_numeric(pd.Series([point.get("service_min_used")]), errors="coerce").fillna(0.0).iloc[0]
        wait_min = pd.to_numeric(pd.Series([point.get("wait_min")]), errors="coerce").fillna(0.0).iloc[0]
        if lanes <= 0:
            primary_driver = "Closed hours"
        elif lanes <= 1 and wait_min > 2.0:
            primary_driver = "Limited lane count"
        elif net_change > 0.25:
            primary_driver = "Inflow exceeds service capacity"
        elif net_change < -0.25:
            primary_driver = "Service capacity clears queue"
        elif fallback_service > 0 and service > (fallback_service * 1.25):
            primary_driver = "Slow service time"
        else:
            primary_driver = "Balanced flow"
        rows.append(
            {
                "Horizon": f"+{label} min",
                "Wait (min)": point.get("wait_min"),
                "Lanes": lanes,
                "Service (min)": service,
                "Service capacity": capacity,
                "Blended inflow": inflow,
                "Net queue change": net_change,
                "Primary driver": primary_driver,
            }
        )
    return pd.DataFrame(rows)


st.title("Hybrid Wait Forecast")
st.caption(
    "Separate REAL-only dashboard for future wait prediction. This app does not replace or change the existing simulator/prediction dashboard."
)
st.caption("Open hours enforced in this hybrid app: 09:00-20:00. Outside those hours, lanes, queue, inflow, and wait are forced to 0.")
st.caption(
    f"Runtime paths: app={__file__} | hybrid_wait={getattr(hybrid_wait, '__file__', 'unknown')}"
)
status_box = st.empty()
run_started_at = perf_counter()
_phase_log("App run started", status_box=status_box)

supports_calibration = _hybrid_supports_calibration()

st.sidebar.header("Hybrid Controls")
st.sidebar.caption("Source is fixed to REAL for this dashboard.")
days = st.sidebar.slider("History (days)", 7, 90, 30)
use_bootstrap = st.sidebar.checkbox("Bootstrap sparse training windows", value=False)
use_data_since = st.sidebar.checkbox("Filter data since date", value=False)
default_since = date.today() - timedelta(days=min(days, 30))
data_since_date = st.sidebar.date_input("Data since", value=default_since, disabled=not use_data_since)
current_weight_pct = st.sidebar.slider("Current vs historical weight", 0, 100, 70, help="Higher values trust recent REAL operations more heavily.")
auto_shift_history = st.sidebar.checkbox("Shift more weight to history by +30 min", value=True)
recent_window_min = st.sidebar.slider("Recent signal window (min)", 15, 120, 45, step=15)
st.sidebar.markdown("**Calibration Sandbox**")
st.sidebar.caption("These levers affect only the separate hybrid app. The old dashboard and base pipeline stay unchanged.")
calibration_preset = st.sidebar.selectbox(
    "Calibration preset",
    [
        "Custom",
        "Neutral baseline",
        "Balanced correction",
        "Service trim",
        "Lane uplift",
        "Inflow trim",
    ],
    help="Safe preset multipliers for the separate hybrid app. Choose Custom to use the sliders directly.",
)
if not supports_calibration:
    st.sidebar.warning(
        "This running app session is using an older hybrid module signature, so the calibration multipliers are visible for reference only and won’t change the model until the module reloads."
    )
manual_calibration_enabled = supports_calibration and calibration_preset == "Custom"
inflow_multiplier = st.sidebar.slider("Future inflow multiplier", 0.5, 1.5, 1.0, 0.05, disabled=not manual_calibration_enabled)
service_multiplier = st.sidebar.slider("Service-time multiplier", 0.5, 1.5, 1.0, 0.05, disabled=not manual_calibration_enabled)
lane_multiplier = st.sidebar.slider("Lane multiplier", 0.5, 1.5, 1.0, 0.05, disabled=not manual_calibration_enabled)
safe_preset_values = {
    "Custom": (inflow_multiplier, service_multiplier, lane_multiplier),
    "Neutral baseline": (1.0, 1.0, 1.0),
    "Balanced correction": (0.95, 0.95, 1.05),
    "Service trim": (1.0, 0.90, 1.0),
    "Lane uplift": (1.0, 1.0, 1.10),
    "Inflow trim": (0.90, 1.0, 1.0),
}
applied_inflow_multiplier, applied_service_multiplier, applied_lane_multiplier = safe_preset_values[calibration_preset]
if calibration_preset != "Custom":
    st.sidebar.caption(
        f"Preset applied: inflow {applied_inflow_multiplier:.2f}x · service {applied_service_multiplier:.2f}x · lanes {applied_lane_multiplier:.2f}x"
    )

if st.session_state.get("_last_calibration_preset") != calibration_preset:
    st.session_state["run_deep_validation"] = False
    st.session_state["run_lane_strategy_comparison"] = False
    st.session_state["run_calibration_analysis"] = False
    st.session_state["_last_calibration_preset"] = calibration_preset

use_profile_calibration = st.sidebar.checkbox(
    "Use backtest profile calibration",
    value=True,
    help="When enabled, the separate hybrid app applies mild time-of-day service/lane/inflow corrections derived from the training backtest.",
)
run_deep_validation = st.sidebar.checkbox(
    "Run deep validation sections",
    value=bool(st.session_state.get("run_deep_validation", False)),
    key="run_deep_validation",
    help="Enables the heaviest scenario/window validation blocks. Turn this on only when you want to inspect calibration stability in detail.",
)
run_lane_strategy_comparison = st.sidebar.checkbox(
    "Run lane strategy comparison",
    value=bool(st.session_state.get("run_lane_strategy_comparison", False)),
    key="run_lane_strategy_comparison",
    help="Builds extra strategy views for auto/snapshot/inferred/manual comparison. Turn this on only when you want to inspect lane policy behavior.",
)
run_calibration_analysis = st.sidebar.checkbox(
    "Run calibration analysis sections",
    value=bool(st.session_state.get("run_calibration_analysis", False)),
    key="run_calibration_analysis",
    help="Builds neutral/profile-off/suggested/single-factor scenario comparisons. This is one of the heaviest rerun paths.",
)
render_extended_sections = st.sidebar.checkbox(
    "Render extended diagnostics sections",
    value=bool(st.session_state.get("render_extended_sections", False)),
    key="render_extended_sections",
    help="Shows the large backtest, root-cause, guidance, scenario, and operating recommendation sections. Leave this off for faster page loads.",
)
profile_strength_pct = st.sidebar.slider(
    "Profile calibration strength",
    0,
    100,
    100,
    5,
    disabled=not use_profile_calibration,
    help="Scales the automatic backtest-derived correction layer toward or away from 1.00x. 0% is effectively off, 100% is the full learned correction.",
)
profile_strength = profile_strength_pct / 100.0
lane_strategy = st.sidebar.selectbox(
    "Lane strategy",
    ["auto", "snapshot", "inferred", "manual"],
    help="Auto prefers a fresh snapshot and falls back to inferred lanes when needed.",
)

if st.sidebar.button("Refresh base data"):
    _load_base_result.clear()
    st.rerun()

data_since_iso = datetime.combine(data_since_date, datetime.min.time()).isoformat() if use_data_since else None

try:
    phase_started_at = perf_counter()
    _phase_log(
        f"Loading base result: days={days}, bootstrap={use_bootstrap}, preset={calibration_preset}, lane_strategy={lane_strategy}, "
        f"lane_compare={run_lane_strategy_comparison}, calibration_analysis={run_calibration_analysis}, deep_validation={run_deep_validation}",
        status_box=status_box,
    )
    base_result = _load_base_result(days=days, use_bootstrap=use_bootstrap, data_since_iso=data_since_iso)
    _phase_log("Base result loaded", started_at=phase_started_at, status_box=status_box)
except Exception as exc:
    st.error(f"Could not load the hybrid base result: {exc}")
    st.stop()

manual_active_lanes = None
if lane_strategy == "manual":
    cap = int(max(1, base_result.get("active_lanes", 1), 1))
    snapshot_cap = base_result.get("df_snapshots", pd.DataFrame())
    if not snapshot_cap.empty and "active_lanes" in snapshot_cap.columns:
        snap_max = pd.to_numeric(snapshot_cap["active_lanes"], errors="coerce").dropna()
        if not snap_max.empty:
            cap = max(cap, int(snap_max.max()))
    manual_active_lanes = st.sidebar.slider("Manual active lanes", 1, max(cap, 1), min(max(cap, 1), max(1, int(base_result.get("active_lanes", 1) or 1))))

phase_started_at = perf_counter()
_phase_log("Building main strategy view", status_box=status_box)
result = _build_strategy_view(
    base_result,
    current_weight_pct=current_weight_pct,
    auto_shift_history=auto_shift_history,
    recent_window_min=recent_window_min,
    lane_strategy=lane_strategy,
    manual_active_lanes=manual_active_lanes,
    inflow_multiplier=applied_inflow_multiplier,
    service_multiplier=applied_service_multiplier,
    lane_multiplier=applied_lane_multiplier,
    use_profile_calibration=use_profile_calibration,
    profile_strength=profile_strength,
)
_phase_log("Main strategy view ready", started_at=phase_started_at, status_box=status_box)

strategy_views = {}
if run_lane_strategy_comparison:
    phase_started_at = perf_counter()
    _phase_log("Building lane strategy comparison views", status_box=status_box)
    strategy_views = {
        "auto": _build_strategy_view(
            base_result,
            current_weight_pct=current_weight_pct,
            auto_shift_history=auto_shift_history,
            recent_window_min=recent_window_min,
            lane_strategy="auto",
            inflow_multiplier=applied_inflow_multiplier,
            service_multiplier=applied_service_multiplier,
            lane_multiplier=applied_lane_multiplier,
            use_profile_calibration=use_profile_calibration,
            profile_strength=profile_strength,
        ),
        "snapshot": _build_strategy_view(
            base_result,
            current_weight_pct=current_weight_pct,
            auto_shift_history=auto_shift_history,
            recent_window_min=recent_window_min,
            lane_strategy="snapshot",
            inflow_multiplier=applied_inflow_multiplier,
            service_multiplier=applied_service_multiplier,
            lane_multiplier=applied_lane_multiplier,
            use_profile_calibration=use_profile_calibration,
            profile_strength=profile_strength,
        ),
        "inferred": _build_strategy_view(
            base_result,
            current_weight_pct=current_weight_pct,
            auto_shift_history=auto_shift_history,
            recent_window_min=recent_window_min,
            lane_strategy="inferred",
            inflow_multiplier=applied_inflow_multiplier,
            service_multiplier=applied_service_multiplier,
            lane_multiplier=applied_lane_multiplier,
            use_profile_calibration=use_profile_calibration,
            profile_strength=profile_strength,
        ),
    }
    if lane_strategy == "manual" and manual_active_lanes is not None:
        strategy_views["manual"] = _build_strategy_view(
            base_result,
            current_weight_pct=current_weight_pct,
            auto_shift_history=auto_shift_history,
            recent_window_min=recent_window_min,
            lane_strategy="manual",
            manual_active_lanes=manual_active_lanes,
            inflow_multiplier=applied_inflow_multiplier,
            service_multiplier=applied_service_multiplier,
            lane_multiplier=applied_lane_multiplier,
            use_profile_calibration=use_profile_calibration,
            profile_strength=profile_strength,
        )
    _phase_log("Lane strategy comparison views ready", started_at=phase_started_at, status_box=status_box)
else:
    _phase_log("Lane strategy comparison skipped", status_box=status_box)

calibration_active = any(abs(val - 1.0) > 1e-9 for val in [applied_inflow_multiplier, applied_service_multiplier, applied_lane_multiplier])
neutral_result = None
if calibration_active and run_calibration_analysis:
    phase_started_at = perf_counter()
    _phase_log("Building neutral calibration comparison view", status_box=status_box)
    neutral_result = _build_strategy_view(
        base_result,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
        inflow_multiplier=1.0,
        service_multiplier=1.0,
        lane_multiplier=1.0,
        use_profile_calibration=use_profile_calibration,
        profile_strength=profile_strength,
    )
    _phase_log("Neutral calibration comparison view ready", started_at=phase_started_at, status_box=status_box)

profile_off_result = None
if use_profile_calibration and run_calibration_analysis:
    phase_started_at = perf_counter()
    _phase_log("Building profile-off comparison view", status_box=status_box)
    profile_off_result = _build_strategy_view(
        base_result,
        current_weight_pct=current_weight_pct,
        auto_shift_history=auto_shift_history,
        recent_window_min=recent_window_min,
        lane_strategy=lane_strategy,
        manual_active_lanes=manual_active_lanes,
        inflow_multiplier=applied_inflow_multiplier,
        service_multiplier=applied_service_multiplier,
        lane_multiplier=applied_lane_multiplier,
        use_profile_calibration=False,
        profile_strength=0.0,
    )
    _phase_log("Profile-off comparison view ready", started_at=phase_started_at, status_box=status_box)
elif not run_calibration_analysis:
    _phase_log("Calibration comparison views skipped", status_box=status_box)

if base_result.get("forecast", pd.DataFrame()).empty:
    st.info(
        "There are no future open forecast buckets available right now for the current horizon. "
        "The model inputs and backtest still render, but the +15 / +30 future wait cards will stay empty."
    )
if base_result.get("fallback_warning"):
    st.warning(
        "The shared Prophet pipeline could not build a future forecast for this run, so the hybrid dashboard is using a fallback REAL snapshot/service view. "
        f"Original error: {base_result['fallback_warning']}"
    )

anchor_state_open = bool(result.get("anchor_is_open"))
anchor_state_color = "#16a34a" if anchor_state_open else "#475569"
anchor_state_label = "OPEN NOW" if anchor_state_open else "CLOSED NOW"
st.markdown(
    f'<div style="background:{anchor_state_color};padding:8px 16px;border-radius:8px;'
    f'font-size:1.0em;font-weight:bold;color:white;display:inline-block">{anchor_state_label} - enforced window 09:00-20:00</div>',
    unsafe_allow_html=True,
)
if supports_calibration:
    st.caption("Calibration sandbox status: active. Multiplier controls are being applied inside the separate hybrid app.")
else:
    st.warning(
        "Calibration sandbox status: compatibility mode. This app session is still using an older imported `prediction.hybrid_wait` signature, so multiplier controls stay disabled until the module reloads."
    )
with st.expander("Closed-hours summary", expanded=not anchor_state_open):
    st.caption(
        "This panel shows which hybrid inputs and outputs are actively forced to zero or ignored outside the enforced 09:00-20:00 window."
    )
    st.dataframe(_closed_hours_summary_frame(result), width="stretch")

wait_15 = result.get("wait_15m")
if wait_15 is None:
    banner_color = "#475569"
    label = "PENDING"
    banner_text = "Waiting for a valid +15 min wait estimate"
else:
    banner_color = "#16a34a" if wait_15 < 2 else "#d97706" if wait_15 < 5 else "#dc2626"
    label = "OK" if wait_15 < 2 else "BUSY" if wait_15 < 5 else "ALERT"
    banner_text = f"Predicted wait in 15 min: {_fmt_wait(wait_15)}"
st.markdown(
    f'<div style="background:{banner_color};padding:10px 18px;border-radius:8px;'
    f'font-size:1.2em;font-weight:bold;color:white">{label} - {banner_text}</div>',
    unsafe_allow_html=True,
)

for warning in result.get("warnings", []):
    st.warning(warning)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Wait @ +15 min", _fmt_wait(result.get("wait_15m")))
k2.metric("Wait @ +30 min", _fmt_wait(result.get("wait_30m")))
k3.metric("Waiting backlog now", str(result.get("current_waiting_backlog", "—")))
k4.metric("Active lanes now", str(result.get("active_lanes_now", "—")))
k5.metric("Service median", _fmt_num(result.get("service_median_min"), " min"))
k6.metric("Recent checkout rate", _fmt_num(result.get("recent_checkout_rate"), f" / {predict_viz.BUCKET_MINUTES} min"))

st.subheader("Driver Attribution")
driver_attribution = _driver_attribution_frame(result)
if not driver_attribution.empty:
    dcols = st.columns(len(driver_attribution))
    for idx, (_, row) in enumerate(driver_attribution.iterrows()):
        with dcols[idx]:
            st.metric(row["Horizon"], _fmt_wait(row["Wait (min)"]))
            st.caption(f"driver: {row['Primary driver']}")
            st.caption(f"lanes={_fmt_num(row['Lanes'])} inflow={_fmt_num(row['Blended inflow'])}")
            st.caption(f"capacity={_fmt_num(row['Service capacity'])} net={_fmt_num(row['Net queue change'])}")
    with st.expander("Driver attribution details", expanded=False):
        driver_table = driver_attribution.copy()
        for col in ["Wait (min)", "Lanes", "Service (min)", "Service capacity", "Blended inflow", "Net queue change"]:
            driver_table[col] = pd.to_numeric(driver_table[col], errors="coerce").round(2)
        st.dataframe(driver_table, width="stretch")

st.subheader("Lane Strategy Comparison")
if run_lane_strategy_comparison:
    comparison_rows = []
    for strategy_name in ["auto", "snapshot", "inferred", "manual"]:
        view = strategy_views.get(strategy_name)
        if view is None:
            continue
        comparison_rows.append(
            {
                "Strategy": strategy_name,
                "Lane source": view.get("active_lane_source"),
                "Active lanes now": view.get("active_lanes_now"),
                "Wait @ +15 min": view.get("wait_15m"),
                "Wait @ +30 min": view.get("wait_30m"),
                "Inferred lanes": view.get("inferred_lanes"),
                "Snapshot lanes": view.get("snapshot_lanes"),
            }
        )

    strategy_frame = pd.DataFrame(comparison_rows)
    if not strategy_frame.empty:
        cols = st.columns(len(strategy_frame))
        for idx, (_, row) in enumerate(strategy_frame.iterrows()):
            with cols[idx]:
                st.metric(f"{row['Strategy']} lanes", str(row["Active lanes now"]))
                st.caption(f"source: {row['Lane source']}")
                st.caption(f"+15: {_fmt_wait(row['Wait @ +15 min'])}")
                st.caption(f"+30: {_fmt_wait(row['Wait @ +30 min'])}")
        strategy_table = strategy_frame.copy()
        strategy_table["Wait @ +15 min"] = pd.to_numeric(strategy_table["Wait @ +15 min"], errors="coerce").round(2)
        strategy_table["Wait @ +30 min"] = pd.to_numeric(strategy_table["Wait @ +30 min"], errors="coerce").round(2)
        st.plotly_chart(
            hybrid_viz.fig_lane_strategy_comparison(strategy_views),
            width="stretch",
            key="fig_lane_strategy_comparison",
        )
        st.dataframe(strategy_table, width="stretch")
        with st.expander("Lane strategy driver details", expanded=False):
            driver_frame = _strategy_driver_rows(strategy_views)
            if not driver_frame.empty:
                driver_table = driver_frame.copy()
                numeric_cols = [
                    "Wait @ +15 min",
                    "Lanes @ +15",
                    "Service @ +15",
                    "Inflow @ +15",
                    "Queue @ +15",
                    "Wait @ +30 min",
                    "Lanes @ +30",
                    "Service @ +30",
                    "Inflow @ +30",
                    "Queue @ +30",
                ]
                for col in numeric_cols:
                    if col in driver_table.columns:
                        driver_table[col] = pd.to_numeric(driver_table[col], errors="coerce").round(2)
                st.dataframe(driver_table, width="stretch")
            else:
                st.info("No lane strategy driver rows are available for the current horizon.")
else:
    st.info("Lane strategy comparison is paused for faster reruns. Enable `Run lane strategy comparison` in the sidebar to build those extra views.")

st.subheader("Strategy Comparison")
c_old1, c_old2, c_new1, c_new2 = st.columns(4)
base_wait_15 = base_result.get("wait_15m")
base_wait_30 = base_result.get("wait_30m")
hybrid_wait_15 = result.get("wait_15m")
hybrid_wait_30 = result.get("wait_30m")
c_old1.metric("Base pipeline @ +15", _fmt_wait(base_wait_15))
c_old2.metric("Base pipeline @ +30", _fmt_wait(base_wait_30))
c_new1.metric(
    "Hybrid @ +15",
    _fmt_wait(hybrid_wait_15),
    delta=_fmt_num((hybrid_wait_15 - base_wait_15) if hybrid_wait_15 is not None and base_wait_15 is not None else None, " min"),
)
c_new2.metric(
    "Hybrid @ +30",
    _fmt_wait(hybrid_wait_30),
    delta=_fmt_num((hybrid_wait_30 - base_wait_30) if hybrid_wait_30 is not None and base_wait_30 is not None else None, " min"),
)
st.caption(
    "The base pipeline is the existing arrival-to-wait path from the old dashboard. The hybrid path keeps that forecast as an ingredient, but anchors more heavily on current REAL queue and service behavior."
)

if run_calibration_analysis and calibration_active and neutral_result is not None:
    st.subheader("Calibration Impact")
    cc1, cc2, cc3, cc4 = st.columns(4)
    neutral_wait_15 = neutral_result.get("wait_15m")
    neutral_wait_30 = neutral_result.get("wait_30m")
    cc1.metric("Neutral @ +15", _fmt_wait(neutral_wait_15))
    cc2.metric(
        "Calibrated @ +15",
        _fmt_wait(result.get("wait_15m")),
        delta=_fmt_num(
            (result.get("wait_15m") - neutral_wait_15)
            if result.get("wait_15m") is not None and neutral_wait_15 is not None
            else None,
            " min",
        ),
    )
    cc3.metric("Neutral @ +30", _fmt_wait(neutral_wait_30))
    cc4.metric(
        "Calibrated @ +30",
        _fmt_wait(result.get("wait_30m")),
        delta=_fmt_num(
            (result.get("wait_30m") - neutral_wait_30)
            if result.get("wait_30m") is not None and neutral_wait_30 is not None
            else None,
            " min",
        ),
    )
    st.caption(
        "This compares the current hybrid sandbox levers against the same hybrid strategy with neutral multipliers at 1.00x."
    )

if run_calibration_analysis and profile_off_result is not None:
    st.subheader("Profile Calibration Impact")
    pc1, pc2, pc3, pc4 = st.columns(4)
    profile_off_training = hybrid_viz.training_values_frame(profile_off_result)
    profile_on_training = hybrid_viz.training_values_frame(result)
    profile_off_metrics = _backtest_metrics(profile_off_training)
    profile_on_metrics = _backtest_metrics(profile_on_training)
    pc1.metric("Profile off @ +15", _fmt_wait(profile_off_result.get("wait_15m")))
    pc2.metric(
        "Profile on @ +15",
        _fmt_wait(result.get("wait_15m")),
        delta=_fmt_num(
            (result.get("wait_15m") - profile_off_result.get("wait_15m"))
            if result.get("wait_15m") is not None and profile_off_result.get("wait_15m") is not None
            else None,
            " min",
        ),
    )
    pc3.metric("Profile off MAE", _fmt_num(profile_off_metrics.get("mae"), " min"))
    pc4.metric(
        "Profile on MAE",
        _fmt_num(profile_on_metrics.get("mae"), " min"),
        delta=_fmt_num(
            (profile_on_metrics.get("mae") - profile_off_metrics.get("mae"))
            if profile_on_metrics.get("mae") is not None and profile_off_metrics.get("mae") is not None
            else None,
            " min",
        ),
    )
    st.caption(
        "This isolates the effect of the automatic backtest-derived service/lane/inflow profile corrections on top of the current preset and lane strategy."
    )

with st.expander("Model inputs", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshot queue count", str(result.get("snapshot_queue_count", "—")))
    c2.metric("Snapshot lanes", str(result.get("snapshot_lanes", "—")))
    c3.metric("Inferred lanes", str(result.get("inferred_lanes", "—")))
    c4.metric("Lane source", str(result.get("active_lane_source", "—")))

    detail_parts = [
        f"Effective blend @ +15 min: **{result.get('effective_current_weight_15_pct', 0):.1f}% current / {result.get('effective_history_weight_15_pct', 0):.1f}% historical**",
        f"Effective blend @ +30 min: **{result.get('effective_current_weight_30_pct', 0):.1f}% current / {result.get('effective_history_weight_30_pct', 0):.1f}% historical**",
        f"Recent signal window: **{result.get('recent_window_min', recent_window_min)} min**",
        f"Current local time: **{pd.to_datetime(result.get('current_clock_ts')).strftime('%Y-%m-%d %H:%M') if result.get('current_clock_ts') is not None else '—'}**",
        f"Anchor snapshot: **{_snapshot_age_text(result.get('snapshot_ts'))}**",
        f"Anchor state: **{'OPEN' if result.get('anchor_is_open') else 'CLOSED'}**",
        f"Lane bounds: **1 to {result.get('max_lanes_cap', '—')}** from REAL data",
        "Lane rule: **recent open-hour service only, minimum 1 lane while open**",
        "Open-hours rule: **09:00-20:00; outside-hours values forced to 0**",
        f"Lane strategy: **{result.get('lane_strategy', lane_strategy)}**",
        f"Calibration support: **{'active' if supports_calibration else 'compatibility mode'}**",
        f"Calibration preset: **{calibration_preset}**",
        f"Backtest profile calibration enabled: **{'yes' if use_profile_calibration else 'no'}**",
        f"Backtest profile strength: **{int(profile_strength * 100)}%**",
        f"Calibration sandbox: **inflow {result.get('inflow_multiplier', 1.0):.2f}x · service {result.get('service_multiplier', 1.0):.2f}x · lanes {result.get('lane_multiplier', 1.0):.2f}x**",
    ]
    calibration_profiles = result.get("calibration_profiles", {}) or {}
    if result.get("use_profile_calibration"):
        detail_parts.append(
            "Backtest profile calibration: "
            f"**service {float(calibration_profiles.get('global_service_factor', 1.0)):.2f}x · "
            f"lanes {float(calibration_profiles.get('global_lane_factor', 1.0)):.2f}x · "
            f"inflow {float(calibration_profiles.get('global_inflow_factor', 1.0)):.2f}x**"
        )
        detail_parts.append(
            f"Profile pockets: **service {len(calibration_profiles.get('service_profile', {}))} · "
            f"lanes {len(calibration_profiles.get('lane_profile', {}))} · "
            f"inflow {len(calibration_profiles.get('inflow_profile', {}))}**"
        )
    if result.get("raw_inferred_lanes") is not None:
        detail_parts.append(f"Raw inferred lanes: **{result['raw_inferred_lanes']}**")
    if result.get("recent_service_events") is not None:
        detail_parts.append(f"Recent service events: **{result['recent_service_events']}**")
    if result.get("recent_distinct_lane_ids") is not None:
        detail_parts.append(f"Recent distinct lane IDs: **{result['recent_distinct_lane_ids']}**")
    if result.get("service_mean_min") is not None:
        detail_parts.append(f"Recent service mean: **{result['service_mean_min']:.2f} min**")
    if result.get("service_fallback_min") is not None:
        detail_parts.append(f"Fallback service: **{result['service_fallback_min']:.2f} min**")
    st.caption("  ·  ".join(detail_parts))

    st.plotly_chart(
        predict_viz.fig_active_lanes_history(base_result),
        width="stretch",
        key="hybrid_active_lanes_history",
    )
    with st.expander("Open-hours boundary debug", expanded=False):
        st.dataframe(_boundary_debug_frame(result.get("snapshot_ts")), width="stretch")

st.subheader("Hybrid Wait Forecast")
st.caption(
    "Observed wait proxy is shown up to now using REAL queue snapshots. The future line rolls the queue forward from the current REAL state while blending recent operations with historical/predicted inflow."
)
st.plotly_chart(
    hybrid_viz.fig_hybrid_wait_forecast(result),
    width="stretch",
    key="fig_hybrid_wait_forecast",
)
with st.expander("Future wait values", expanded=False):
    future_vals = hybrid_viz.future_values_frame(result)
    if not future_vals.empty:
        table = future_vals.copy()
        table["Time"] = pd.to_datetime(table["ds"]).dt.strftime("%Y-%m-%d %H:%M")
        table["Wait (min)"] = table["wait_min"].round(2)
        table["Queue after"] = table["queue_after"].round(2)
        table["Lanes used"] = table["lanes_used"].round(2)
        table["Service used (min)"] = table["service_min_used"].round(2)
        table["Current weight"] = (table["current_weight"] * 100).round(1)
        table["History weight"] = (table["history_weight"] * 100).round(1)
        table["Recent REAL rate"] = table["recent_real_checkout_rate"].round(2)
        table["Forecast checkout"] = table["forecast_checkout_pred"].round(2)
        table["Historical checkout"] = table["historical_checkout_rate"].round(2)
        table["Historical/predicted blend"] = table["historical_pred_checkout_rate"].round(2)
        table["Blended inflow used"] = table["blended_checkout_rate"].round(2)
        st.dataframe(
            table[
                [
                    "Time",
                    "Wait (min)",
                    "Queue after",
                    "Lanes used",
                    "Service used (min)",
                    "Current weight",
                    "History weight",
                    "Recent REAL rate",
                    "Forecast checkout",
                    "Historical checkout",
                    "Historical/predicted blend",
                    "Blended inflow used",
                ]
            ],
            width="stretch",
        )
    else:
        st.info("No future wait values available yet.")

st.subheader("Checkout Inflow Blend")
st.caption(
    "This chart shows how the new model mixes recent REAL checkout throughput with the historical/predicted inflow pattern before projecting backlog into the future."
)
st.plotly_chart(
    hybrid_viz.fig_hybrid_inflow_components(result),
    width="stretch",
    key="fig_hybrid_inflow_components",
)

if not render_extended_sections:
    st.info(
        "Extended diagnostics sections are paused for faster page loads. "
        "Enable `Render extended diagnostics sections` in the sidebar when you want the full backtest, guidance, and scenario analysis."
    )
    _phase_log("Extended diagnostics skipped", status_box=status_box)
    _phase_log("Render complete", started_at=run_started_at, status_box=status_box)
    status_box.success("Hybrid app render complete")
    st.stop()

st.subheader("Training-Period Wait Backtest")
st.caption(
    "This comparison stays on the same historical training window so you can inspect whether the hybrid wait estimate is still inflated. The observed line is a snapshot-derived wait proxy, not a direct completed-customer wait measurement."
)
training_vals = hybrid_viz.training_values_frame(result)
metrics = _backtest_metrics(training_vals)
driver_summary = _backtest_driver_summary(training_vals)
root_summary = _root_cause_summary(result)
suggested_calibration = _suggested_calibration(metrics, driver_summary, root_summary)
suggested_result = None
calibration_scenarios: list[tuple[str, float, float, float, dict | None]] = []
calibration_scenario_views: dict[str, dict] = {}

if run_calibration_analysis:
    suggested_is_non_neutral = any(
        abs(float(suggested_calibration[key]) - 1.0) > 1e-9
        for key in [
            "suggested_inflow_multiplier",
            "suggested_service_multiplier",
            "suggested_lane_multiplier",
        ]
    )
    if suggested_is_non_neutral:
        phase_started_at = perf_counter()
        _phase_log("Building suggested calibration preview view", status_box=status_box)
        suggested_result = _build_strategy_view(
            base_result,
            current_weight_pct=current_weight_pct,
            auto_shift_history=auto_shift_history,
            recent_window_min=recent_window_min,
            lane_strategy=lane_strategy,
            manual_active_lanes=manual_active_lanes,
            inflow_multiplier=float(suggested_calibration["suggested_inflow_multiplier"]),
            service_multiplier=float(suggested_calibration["suggested_service_multiplier"]),
            lane_multiplier=float(suggested_calibration["suggested_lane_multiplier"]),
            use_profile_calibration=use_profile_calibration,
            profile_strength=profile_strength,
        )
        _phase_log("Suggested calibration preview view ready", started_at=phase_started_at, status_box=status_box)
    calibration_scenarios = [
        ("Neutral baseline", 1.0, 1.0, 1.0, neutral_result if neutral_result is not None else result),
        ("Current sandbox", applied_inflow_multiplier, applied_service_multiplier, applied_lane_multiplier, result),
    ]
    if suggested_result is not None:
        calibration_scenarios.append(
            (
                "Suggested target",
                float(suggested_calibration["suggested_inflow_multiplier"]),
                float(suggested_calibration["suggested_service_multiplier"]),
                float(suggested_calibration["suggested_lane_multiplier"]),
                suggested_result,
            )
        )
    single_axis_specs = [
        ("Service-only correction", 1.0, float(suggested_calibration["suggested_service_multiplier"]), 1.0),
        ("Lane-only correction", 1.0, 1.0, float(suggested_calibration["suggested_lane_multiplier"])),
        ("Inflow-only correction", float(suggested_calibration["suggested_inflow_multiplier"]), 1.0, 1.0),
    ]
    for scenario_name, scenario_inflow, scenario_service, scenario_lanes in single_axis_specs:
        if abs(scenario_inflow - 1.0) < 1e-9 and abs(scenario_service - 1.0) < 1e-9 and abs(scenario_lanes - 1.0) < 1e-9:
            continue
        calibration_scenarios.append(
            (
                scenario_name,
                scenario_inflow,
                scenario_service,
                scenario_lanes,
                _build_strategy_view(
                    base_result,
                    current_weight_pct=current_weight_pct,
                    auto_shift_history=auto_shift_history,
                    recent_window_min=recent_window_min,
                    lane_strategy=lane_strategy,
                    manual_active_lanes=manual_active_lanes,
                    inflow_multiplier=scenario_inflow,
                    service_multiplier=scenario_service,
                    lane_multiplier=scenario_lanes,
                    use_profile_calibration=use_profile_calibration,
                    profile_strength=profile_strength,
                ),
            )
        )
    calibration_scenario_views = {
        scenario_name: scenario_view
        for scenario_name, _scenario_inflow, _scenario_service, _scenario_lanes, scenario_view in calibration_scenarios
        if scenario_view is not None
    }
m1, m2, m3 = st.columns(3)
m1.metric("Median factor", _fmt_num(metrics["median_factor"], "x"))
m2.metric("Mean abs error", _fmt_num(metrics["mae"], " min"))
m3.metric("P90 abs error", _fmt_num(metrics["p90_abs_error"], " min"))
st.plotly_chart(
    hybrid_viz.fig_hybrid_training_backtest(result),
    width="stretch",
    key="fig_hybrid_training_backtest",
)
st.subheader("Backtest Error Profile")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Overestimate buckets", _fmt_num(driver_summary["overestimate_share"], "%"))
e2.metric("Median service ratio", _fmt_num(driver_summary["median_service_ratio"], "x"))
e3.metric("Median lane ratio (est/obs)", _fmt_num(driver_summary["median_lane_ratio"], "x"))
e4.metric("Worst time of day", str(driver_summary["worst_time_label"] or "—"))
st.caption(
    "This profile groups the same training-period comparison by time of day, so we can see whether the inflation clusters around specific hours and whether service or lane assumptions are consistently above the snapshot-derived proxy."
)
st.plotly_chart(
    hybrid_viz.fig_backtest_factor_profile(result),
    width="stretch",
    key="fig_backtest_factor_profile",
)
st.subheader("Backtest Root Cause Mix")
if not root_summary.empty:
    root_cols = st.columns(min(len(root_summary), 4))
    for idx, (_, row) in enumerate(root_summary.head(4).iterrows()):
        with root_cols[idx]:
            st.metric(str(row["Root cause"]), f"{int(row['Buckets'])} buckets")
            st.caption(f"{float(row['Share %']):.1f}% of training backtest buckets")
    st.caption(
        "Each backtest bucket is classified as aligned, service-led, lane-led, combined service-plus-lanes, or other inflation. This is a heuristic view meant to show which assumption is most often responsible when the hybrid estimate runs hot."
    )
    st.plotly_chart(
        hybrid_viz.fig_backtest_root_cause_profile(result),
        width="stretch",
        key="fig_backtest_root_cause_profile",
    )
    with st.expander("Backtest root cause summary", expanded=False):
        st.dataframe(root_summary, width="stretch")
else:
    st.info("No root-cause mix is available for the current training backtest.")
st.subheader("Tuning Guidance")
guidance_rows = _tuning_guidance(result, driver_summary, root_summary)
if guidance_rows:
    for row in guidance_rows:
        st.markdown(
            f"**P{row['Priority']} · {row['Focus']}**  \n"
            f"Why: {row['Why']}  \n"
            f"Next: {row['Next move']}"
        )
    st.caption(
        "Backtest-based sandbox target: "
        f"inflow **{suggested_calibration['suggested_inflow_multiplier']:.2f}x**, "
        f"service **{suggested_calibration['suggested_service_multiplier']:.2f}x**, "
        f"lanes **{suggested_calibration['suggested_lane_multiplier']:.2f}x** "
        f"from dominant cause **{suggested_calibration['dominant_cause']}** "
        f"({suggested_calibration['reason']})."
    )
    with st.expander("Tuning guidance table", expanded=False):
        st.dataframe(pd.DataFrame(guidance_rows), width="stretch")
    with st.expander("Suggested calibration target", expanded=False):
        st.dataframe(pd.DataFrame([suggested_calibration]), width="stretch")
    if run_calibration_analysis and suggested_result is not None:
        st.subheader("Suggested Calibration Preview")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Current @ +15", _fmt_wait(result.get("wait_15m")))
        sc2.metric(
            "Suggested @ +15",
            _fmt_wait(suggested_result.get("wait_15m")),
            delta=_fmt_num(
                (suggested_result.get("wait_15m") - result.get("wait_15m"))
                if suggested_result.get("wait_15m") is not None and result.get("wait_15m") is not None
                else None,
                " min",
            ),
        )
        sc3.metric("Current @ +30", _fmt_wait(result.get("wait_30m")))
        sc4.metric(
            "Suggested @ +30",
            _fmt_wait(suggested_result.get("wait_30m")),
            delta=_fmt_num(
                (suggested_result.get("wait_30m") - result.get("wait_30m"))
                if suggested_result.get("wait_30m") is not None and result.get("wait_30m") is not None
                else None,
                " min",
            ),
        )
        st.caption(
            "This preview applies the backtest-based target multipliers with the same lane strategy and weight settings, so you can compare the recommended calibration against your current sandbox state before changing the sliders."
        )
        with st.expander("Suggested calibration details", expanded=False):
            preview_table = pd.DataFrame(
                [
                    {
                        "State": "Current sandbox",
                        "Inflow multiplier": applied_inflow_multiplier,
                        "Service multiplier": applied_service_multiplier,
                        "Lane multiplier": applied_lane_multiplier,
                        "Wait @ +15 min": result.get("wait_15m"),
                        "Wait @ +30 min": result.get("wait_30m"),
                    },
                    {
                        "State": "Suggested target",
                        "Inflow multiplier": suggested_calibration["suggested_inflow_multiplier"],
                        "Service multiplier": suggested_calibration["suggested_service_multiplier"],
                        "Lane multiplier": suggested_calibration["suggested_lane_multiplier"],
                        "Wait @ +15 min": suggested_result.get("wait_15m"),
                        "Wait @ +30 min": suggested_result.get("wait_30m"),
                    },
                ]
            )
            for col in [
                "Inflow multiplier",
                "Service multiplier",
                "Lane multiplier",
                "Wait @ +15 min",
                "Wait @ +30 min",
            ]:
                preview_table[col] = pd.to_numeric(preview_table[col], errors="coerce").round(2)
            st.dataframe(preview_table, width="stretch")
    elif not run_calibration_analysis:
        st.info("Suggested calibration preview is paused for faster reruns. Enable `Run calibration analysis sections` in the sidebar to build it.")
st.subheader("Calibration Scenario Comparison")
if not run_calibration_analysis:
    st.info("Calibration scenario comparison is paused for faster reruns. Enable `Run calibration analysis sections` in the sidebar to build neutral/profile-off/suggested scenario boards.")
scenario_rows: list[dict[str, object]] = []
best_row: pd.Series | None = None
current_row: pd.Series | None = None
if run_calibration_analysis:
    for scenario_name, scenario_inflow, scenario_service, scenario_lanes, scenario_view in calibration_scenarios:
        if scenario_view is None:
            continue
        scenario_rows.append(
            _scenario_summary_row(
                scenario_name,
                scenario_view,
                inflow=scenario_inflow,
                service=scenario_service,
                lanes=scenario_lanes,
            )
        )
scenario_frame = pd.DataFrame(scenario_rows)
if run_calibration_analysis and not scenario_frame.empty:
    sortable = _rank_scenarios(scenario_frame)
    delta_frame = _scenario_delta_frame(sortable)
    best_row = sortable.iloc[0]
    current_candidates = sortable[sortable["Scenario"] == "Current sandbox"]
    current_row = current_candidates.iloc[0] if not current_candidates.empty else None
    st.subheader("Recommended Scenario")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Best scenario", str(best_row["Scenario"]))
    rc2.metric("Best @ +15", _fmt_wait(best_row["Wait @ +15 min"]))
    rc3.metric("Best factor", _fmt_num(best_row["Median factor"], "x"))
    rc4.metric("Best MAE", _fmt_num(best_row["MAE (min)"], " min"))
    if current_row is not None:
        st.caption(
            "Best scenario versus current sandbox: "
            f"+15 **{_fmt_num(pd.to_numeric(pd.Series([best_row['Wait @ +15 min']]), errors='coerce').iloc[0] - pd.to_numeric(pd.Series([current_row['Wait @ +15 min']]), errors='coerce').iloc[0], ' min')}**, "
            f"factor **{_fmt_num(pd.to_numeric(pd.Series([best_row['Median factor']]), errors='coerce').iloc[0] - pd.to_numeric(pd.Series([current_row['Median factor']]), errors='coerce').iloc[0], 'x')}**, "
            f"MAE **{_fmt_num(pd.to_numeric(pd.Series([best_row['MAE (min)']]), errors='coerce').iloc[0] - pd.to_numeric(pd.Series([current_row['MAE (min)']]), errors='coerce').iloc[0], ' min')}**."
        )
        st.caption(f"Why it wins: {_best_scenario_reason(best_row, current_row)}.")
    top_cols = st.columns(min(len(sortable), 4))
    for idx, (_, row) in enumerate(sortable.head(4).iterrows()):
        with top_cols[idx]:
            st.metric(str(row["Scenario"]), _fmt_wait(row["Wait @ +15 min"]))
            st.caption(f"+30: {_fmt_wait(row['Wait @ +30 min'])}")
            st.caption(f"factor={_fmt_num(row['Median factor'], 'x')} mae={_fmt_num(row['MAE (min)'], ' min')}")
    st.caption(
        "These scenarios use the same hybrid strategy and data window, but vary only the calibration sandbox multipliers so you can compare neutral, current, suggested, and single-factor corrections. Ranking prioritizes closeness to a 1.00x median factor, then lower MAE, then lower +15 wait, with extra penalties for suspicious underestimation and unrealistically low +15 waits when backlog is still present."
    )
    ordered_views = {
        str(row["Scenario"]): calibration_scenario_views.get(str(row["Scenario"]))
        for _, row in sortable.head(5).iterrows()
        if calibration_scenario_views.get(str(row["Scenario"])) is not None
    }
    if ordered_views:
        st.plotly_chart(
            hybrid_viz.fig_calibration_scenario_comparison(ordered_views),
            width="stretch",
            key="fig_calibration_scenario_comparison",
        )
    display_frame = sortable.copy()
    for col in [
        "Inflow x",
        "Service x",
        "Lanes x",
        "Wait @ +15 min",
        "Wait @ +30 min",
        "Median factor",
        "MAE (min)",
        "Current backlog",
        "Active lanes now",
        "factor_distance",
        "underestimate_penalty",
        "backlog_floor_penalty",
        "score",
    ]:
        display_frame[col] = pd.to_numeric(display_frame[col], errors="coerce").round(2)
    st.dataframe(display_frame, width="stretch")
    if not delta_frame.empty:
        with st.expander("Scenario deltas vs current sandbox", expanded=False):
            delta_display = delta_frame.copy()
            for col in [
                "Inflow x",
                "Service x",
                "Lanes x",
                "Wait @ +15 min",
                "Wait @ +30 min",
                "Median factor",
                "MAE (min)",
                "Delta +15 (min)",
                "Delta +30 (min)",
                "Delta factor",
                "Delta MAE (min)",
                "Delta score",
            ]:
                if col in delta_display.columns:
                    delta_display[col] = pd.to_numeric(delta_display[col], errors="coerce").round(2)
            st.dataframe(
                delta_display[
                    [
                        "Scenario",
                        "Inflow x",
                        "Service x",
                        "Lanes x",
                        "Wait @ +15 min",
                        "Wait @ +30 min",
                        "Median factor",
                        "MAE (min)",
                        "Delta +15 (min)",
                        "Delta +30 (min)",
                        "Delta factor",
                        "Delta MAE (min)",
                        "Delta score",
                    ]
                ],
                width="stretch",
            )
st.subheader("Recent Window Validation")
validation_windows = sorted({7, 14, 30, int(days)})
stability: dict[str, object] | None = None
readiness: dict[str, object] | None = None
profile_mode_summary: dict[str, object] | None = None

if run_deep_validation:
    phase_started_at = perf_counter()
    _phase_log("Running recent-window validation", status_box=status_box)
    validation_rows: list[dict[str, object]] = []
    for window_days in validation_windows:
        validation_rows.append(
            _validation_window_row(
                window_days=window_days,
                use_bootstrap=use_bootstrap,
                data_since_iso=data_since_iso,
                current_weight_pct=current_weight_pct,
                auto_shift_history=auto_shift_history,
                recent_window_min=recent_window_min,
                lane_strategy=lane_strategy,
                manual_active_lanes=manual_active_lanes,
                applied_inflow_multiplier=applied_inflow_multiplier,
                applied_service_multiplier=applied_service_multiplier,
                applied_lane_multiplier=applied_lane_multiplier,
                use_profile_calibration=use_profile_calibration,
                profile_strength=profile_strength,
            )
        )
    validation_frame = pd.DataFrame(validation_rows)
    if not validation_frame.empty:
        stability = _validation_stability_summary(validation_frame)
        readiness = _deployment_readiness(stability, validation_frame)
        vv1, vv2, vv3 = st.columns(3)
        best_counts = validation_frame["Best scenario"].value_counts()
        vv1.metric("Windows tested", str(len(validation_frame)))
        vv2.metric("Suggested wins", str(int(best_counts.get("Suggested target", 0))))
        vv3.metric("Current wins", str(int(best_counts.get("Current sandbox", 0))))
        sv1, sv2, sv3 = st.columns(3)
        sv1.metric("Validation verdict", str(stability["verdict"]))
        sv2.metric("Suggested win rate", _fmt_num(stability["suggested_win_rate"], "%"))
        sv3.metric("Avg MAE gain", _fmt_num(stability["avg_mae_gain"], " min"))
        st.caption(
            "This re-runs the hybrid setup across several recent REAL history windows to check whether the recommendation is stable or only local to one training slice."
        )
        st.caption(stability["summary"])
        rv1, rv2 = st.columns(2)
        rv1.metric("Deployment readiness", str(readiness["verdict"]))
        rv2.metric("Readiness checks", f"{readiness['checks_passed']}/{readiness['checks_total']}")
        st.caption(readiness["summary"])
        validation_frame["Delta factor (current-suggested)"] = (
            pd.to_numeric(validation_frame["Current factor"], errors="coerce")
            - pd.to_numeric(validation_frame["Suggested factor"], errors="coerce")
        )
        validation_frame["Delta MAE (current-suggested)"] = (
            pd.to_numeric(validation_frame["Current MAE"], errors="coerce")
            - pd.to_numeric(validation_frame["Suggested MAE"], errors="coerce")
        )
        st.plotly_chart(
            hybrid_viz.fig_recent_window_validation(validation_frame),
            width="stretch",
            key="fig_recent_window_validation",
        )
        for col in [
            "Current @ +15",
            "Current factor",
            "Current MAE",
            "Suggested @ +15",
            "Suggested factor",
            "Suggested MAE",
            "Best @ +15",
            "Best factor",
            "Best MAE",
            "Delta factor (current-suggested)",
            "Delta MAE (current-suggested)",
        ]:
            validation_frame[col] = pd.to_numeric(validation_frame[col], errors="coerce").round(2)
        st.dataframe(validation_frame, width="stretch")

    if supports_calibration:
        profile_phase_started_at = perf_counter()
        _phase_log("Running profile-mode validation", status_box=status_box)
        st.subheader("Profile Mode Validation")
        profile_mode_rows: list[dict[str, object]] = []
        for window_days in validation_windows:
            profile_mode_rows.append(
                _profile_mode_window_row(
                    window_days=window_days,
                    use_bootstrap=use_bootstrap,
                    data_since_iso=data_since_iso,
                    current_weight_pct=current_weight_pct,
                    auto_shift_history=auto_shift_history,
                    recent_window_min=recent_window_min,
                    lane_strategy=lane_strategy,
                    manual_active_lanes=manual_active_lanes,
                    applied_inflow_multiplier=applied_inflow_multiplier,
                    applied_service_multiplier=applied_service_multiplier,
                    applied_lane_multiplier=applied_lane_multiplier,
                    profile_strength=profile_strength,
                )
            )
        profile_mode_frame = pd.DataFrame(profile_mode_rows)
        if not profile_mode_frame.empty:
            profile_mode_summary = _profile_mode_summary(profile_mode_frame)
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Profile verdict", str(profile_mode_summary["verdict"]))
            pm2.metric("Profile-on win rate", _fmt_num(profile_mode_summary["profile_on_win_rate"], "%"))
            pm3.metric("Avg MAE gain", _fmt_num(profile_mode_summary["avg_mae_gain"], " min"))
            st.caption(profile_mode_summary["summary"])
            for col in [
                "Profile on @ +15",
                "Profile on factor",
                "Profile on MAE",
                "Profile off @ +15",
                "Profile off factor",
                "Profile off MAE",
            ]:
                profile_mode_frame[col] = pd.to_numeric(profile_mode_frame[col], errors="coerce").round(2)
            st.dataframe(profile_mode_frame, width="stretch")
        _phase_log("Profile-mode validation ready", started_at=profile_phase_started_at, status_box=status_box)
    _phase_log("Recent-window validation ready", started_at=phase_started_at, status_box=status_box)
else:
    _phase_log("Deep validation skipped", status_box=status_box)
    st.info(
        "Deep validation is paused for faster preset changes. Enable `Run deep validation sections` in the sidebar when you want the multi-window stability and profile-mode checks."
    )

operating_recommendation = _operating_recommendation(
    best_row=best_row,
    current_row=current_row,
    readiness=readiness,
    profile_mode_summary=profile_mode_summary,
    current_profile_enabled=use_profile_calibration,
)
st.subheader("Operating Recommendation")
or1, or2, or3 = st.columns(3)
or1.metric("Recommended scenario", operating_recommendation["scenario"])
or2.metric("Profile calibration", operating_recommendation["profile_mode"])
or3.metric("Confidence", operating_recommendation["confidence"])
st.caption(f"Why: {operating_recommendation['rationale']}.")
st.caption(f"Next: {operating_recommendation['action']}")
with st.expander("Operating recommendation audit trail", expanded=False):
    audit_frame = _operating_recommendation_trace(
        best_row=best_row,
        current_row=current_row,
        readiness=readiness,
        profile_mode_summary=profile_mode_summary,
        operating_recommendation=operating_recommendation,
    )
    st.dataframe(audit_frame, width="stretch")
with st.expander("Training-period wait values", expanded=False):
    if not training_vals.empty:
        table = training_vals.copy()
        table["Time"] = pd.to_datetime(table["ds"]).dt.strftime("%Y-%m-%d %H:%M")
        table["Observed wait proxy (min)"] = table["observed_wait_min"].round(2)
        table["Hybrid estimated wait (min)"] = table["estimated_wait_min"].round(2)
        table["Wait delta (min)"] = table["wait_delta"].round(2)
        table["Waiting queue"] = table["waiting_queue"].round(2)
        table["Observed service (min)"] = table["observed_service_min"].round(2)
        table["Estimated service (min)"] = table["estimated_service_min"].round(2)
        table["Observed lanes"] = table["active_lanes"].round(2)
        table["Estimated lanes"] = table["estimated_lanes"].round(2)
        table["Service ratio"] = table["service_ratio"].round(2)
        table["Lane ratio"] = table["lane_ratio"].round(2)
        table["Factor"] = table["factor"].round(2)
        table["Queue"] = table["queue_count"].round(2)
        st.dataframe(
            table[
                [
                    "Time",
                    "Observed wait proxy (min)",
                    "Hybrid estimated wait (min)",
                    "Wait delta (min)",
                    "Factor",
                    "Queue",
                    "Waiting queue",
                    "Observed service (min)",
                    "Estimated service (min)",
                    "Observed lanes",
                    "Estimated lanes",
                    "Service ratio",
                    "Lane ratio",
                ]
            ],
            width="stretch",
        )
    else:
        st.info("No historical backtest values available.")

st.subheader("Demand Context")
st.caption(
    "This is the existing REAL entrance forecast reused as a future-pressure ingredient. In the hybrid app it informs future inflow, but it is no longer the only foundation of the wait prediction."
)
if base_result.get("forecast", pd.DataFrame()).empty:
    st.info("Base Prophet demand context is unavailable for this run.")
else:
    st.plotly_chart(
        predict_viz.fig_prophet_forecast(base_result),
        width="stretch",
        key="fig_hybrid_base_forecast",
    )

_phase_log("Render complete", started_at=run_started_at, status_box=status_box)
status_box.success("Hybrid app render complete")
