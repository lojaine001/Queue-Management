import re
from pathlib import Path

p = Path("Queue-Management/Queue-Management-System-v2-main/Queue-Management-System-v2-main/ensemble_predict.py")
content = p.read_text(encoding="utf-8")

# 1. Replace Unicode characters across the whole file
unicode_map = {
    "\u2014": "--",
    "\u2192": "->",
    "\u2713": "[OK]",
    "\u2717": "[FAIL]",
    "\xd7": "x",
    "\u2500": "-",
}
for k, v in unicode_map.items():
    content = content.replace(k, v)

# 2. Update Section 3 & 3b (Outlier cap and smoothing with BURST_BYPASS)
old_section = """    # -- 3. Outlier cap -------------------------------------------------------
    non_zero = df.loc[df["y"] > 0, "y"]
    if not non_zero.empty:
        cap_value = max(float(non_zero.quantile(0.99)), 30.0)
        df["y"] = df["y"].clip(upper=cap_value)
        _log(f"[Data] Outlier cap applied at y<={cap_value:.2f} from {len(non_zero)} non-zero buckets")
    else:
        _log("[Data] No non-zero buckets before closed-hour fill")

    # -- 3b. 15-min rolling median smooth (5 x 3-min buckets) -----------------
    df["y"] = (
        df["y"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .round(2)
    )
    _log("[Data] Applied 15-min rolling median smooth to training data")"""

new_section = """    # -- 3. Outlier cap (configurable) --------------------------------------
    OUTLIER_CAP = os.getenv("OUTLIER_CAP", "50")
    OUTLIER_QUANTILE = float(os.getenv("OUTLIER_QUANTILE", "0.99"))
    OUTLIER_BURST_THRESHOLD = float(os.getenv("OUTLIER_BURST_THRESHOLD", "40"))
    BURST_BYPASS = os.getenv("BURST_BYPASS", "true").strip().lower() in {"1", "true", "yes", "on"}

    non_zero = df.loc[df["y"] > 0, "y"]
    burst_days = set()
    if not non_zero.empty:
        try:
            if OUTLIER_CAP.strip().lower() in {"", "none", "0", "false"}:
                cap_value = None
            else:
                cap_env = float(OUTLIER_CAP)
                cap_value = max(float(non_zero.quantile(OUTLIER_QUANTILE)), cap_env)
        except Exception:
            cap_value = None

        df["date_only"] = pd.to_datetime(df["ds"]).dt.date
        burst_days = set(df.loc[df["y"] >= OUTLIER_BURST_THRESHOLD, "date_only"].unique())
        if cap_value is not None:
            if burst_days and BURST_BYPASS:
                _log(f"[Data] Burst days detected: {len(burst_days)} -- bypassing cap for those days")
                mask_no_burst = ~df["date_only"].isin(burst_days)
                df.loc[mask_no_burst, "y"] = df.loc[mask_no_burst, "y"].clip(upper=cap_value)
                _log(f"[Data] Outlier cap applied at y<={cap_value:.2f} excluding burst days")
            else:
                df["y"] = df["y"].clip(upper=cap_value)
                _log(f"[Data] Outlier cap applied at y<={cap_value:.2f} from {len(non_zero)} non-zero buckets")
        else:
            _log("[Data] Outlier cap disabled by environment")
    else:
        _log("[Data] No non-zero buckets before closed-hour fill")

    # -- 3b. 15-min rolling median smooth (5 x 3-min buckets) -----------------
    SMOOTH_WINDOW = int(os.getenv("SMOOTH_WINDOW", "5"))
    if SMOOTH_WINDOW > 1:
        smoothed = (
            df["y"]
            .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
            .median()
            .round(2)
        )
        if BURST_BYPASS and "date_only" in df.columns and burst_days:
            mask_no_burst = ~df["date_only"].isin(burst_days)
            df.loc[mask_no_burst, "y"] = smoothed.loc[mask_no_burst]
            _log(f"[Data] Applied rolling median smooth (window={SMOOTH_WINDOW}) bypassing burst days ({len(burst_days)} days)")
        elif BURST_BYPASS and (df["y"] >= OUTLIER_BURST_THRESHOLD).any():
            burst_mask = df["y"] >= OUTLIER_BURST_THRESHOLD
            df.loc[~burst_mask, "y"] = smoothed.loc[~burst_mask]
            _log(f"[Data] Applied rolling median smooth (window={SMOOTH_WINDOW}) bypassing individual burst buckets")
        else:
            df["y"] = smoothed
            _log(f"[Data] Applied rolling median smooth to training data (window={SMOOTH_WINDOW})")
    else:
        _log(f"[Data] Rolling median smoothing skipped (SMOOTH_WINDOW={SMOOTH_WINDOW})")"""

if old_section in content:
    content = content.replace(old_section, new_section)
    print("Replaced outlier/smoothing section successfully!")
else:
    print("Could not find exact old section; will check regex")

p.write_text(content, encoding="utf-8")
print("Saved updated ensemble_predict.py")
