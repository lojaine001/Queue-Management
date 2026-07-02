"""
IQMS Data Analysis — generates all charts as PNG files.
Run: python generate_analysis.py
Output: analysis_output/ folder with all charts.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.titlesize": 13})

BASE = os.path.dirname(__file__)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {name}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading CSVs...")
preds = pd.read_csv(os.path.join(BASE, "queue_predictions.csv"))
preds["predicted_at"]   = pd.to_datetime(preds["predicted_at"],   utc=True, errors="coerce")
preds["prediction_for"] = pd.to_datetime(preds["prediction_for"], utc=True, errors="coerce")
svc   = pd.read_csv(os.path.join(BASE, "service_events.csv"))
svc["timestamp"] = pd.to_datetime(svc["timestamp"], utc=True, errors="coerce")
ent   = pd.read_csv(os.path.join(BASE, "entrance_events.csv"))
ent["timestamp"] = pd.to_datetime(ent["timestamp"], utc=True, errors="coerce")

# Filter to real camera data only
svc_real  = svc[~svc["camera_id"].str.startswith("SIM", na=False)].copy()
ent_real  = ent[~ent["camera_id"].str.startswith("SIM", na=False)].copy()
preds_ens = preds[preds["source"] == "ensemble"].copy()

print(f"  queue_predictions: {len(preds):,} rows")
print(f"  service_events:    {len(svc_real):,} real rows")
print(f"  entrance_events:   {len(ent_real):,} real rows\n")

# ─────────────────────────────────────────────────────────────────────────────
# 01 — Service event dwell time distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Generating charts...")
dwell = svc_real["total_dwell_sec"].dropna()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Service Event Dwell Time Distribution", fontweight="bold")

axes[0].hist(dwell.clip(upper=300), bins=80, color="#4e9af1", edgecolor="none")
axes[0].set_xlabel("Dwell time (seconds)")
axes[0].set_ylabel("Count")
axes[0].set_title("Full range (clipped at 300s)")
axes[0].axvline(dwell.median(), color="orange", lw=2, label=f"Median {dwell.median():.1f}s")
axes[0].axvline(dwell.mean(),   color="red",    lw=2, linestyle="--", label=f"Mean {dwell.mean():.1f}s")
axes[0].legend()

axes[1].hist(np.log1p(dwell), bins=80, color="#a0d8a0", edgecolor="none")
axes[1].set_xlabel("log(1 + dwell_sec)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log scale — reveals shape")

save(fig, "01_dwell_time_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 02 — Equipment type breakdown (service events)
# ─────────────────────────────────────────────────────────────────────────────
equip_svc = svc_real["equipment_type"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Equipment Type — Service Events", fontweight="bold")

colors = ["#4e9af1", "#f1914e", "#a0d8a0"]
axes[0].bar(equip_svc.index, equip_svc.values, color=colors[:len(equip_svc)])
axes[0].set_ylabel("Count")
axes[0].set_title("Count by equipment type")
for i, v in enumerate(equip_svc.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=9)

axes[1].pie(equip_svc.values, labels=equip_svc.index, autopct="%1.1f%%",
            colors=colors[:len(equip_svc)], startangle=140)
axes[1].set_title("Share")

save(fig, "02_equipment_service_events.png")

# ─────────────────────────────────────────────────────────────────────────────
# 03 — Dwell time by equipment type (boxplot)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Dwell Time by Equipment Type (seconds)", fontweight="bold")
svc_box = svc_real[svc_real["total_dwell_sec"] < 200]
sns.boxplot(data=svc_box, x="equipment_type", y="total_dwell_sec",
            palette=["#4e9af1","#f1914e","#a0d8a0"], ax=ax)
ax.set_xlabel("Equipment type")
ax.set_ylabel("Dwell time (seconds)")
ax.set_title("Clipped at 200s for readability")
save(fig, "03_dwell_by_equipment.png")

# ─────────────────────────────────────────────────────────────────────────────
# 04 — Lane performance
# ─────────────────────────────────────────────────────────────────────────────
lane_stats = (svc_real.groupby("lane_id")["total_dwell_sec"]
              .agg(["count", "mean", "median"])
              .reset_index()
              .dropna(subset=["lane_id"]))
lane_stats["lane_id"] = lane_stats["lane_id"].astype(str)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Lane Performance", fontweight="bold")

axes[0].bar(lane_stats["lane_id"], lane_stats["count"], color="#4e9af1")
axes[0].set_xlabel("Lane")
axes[0].set_ylabel("Number of events")
axes[0].set_title("Event count per lane")
for i, (_, row) in enumerate(lane_stats.iterrows()):
    axes[0].text(i, row["count"] + 100, f"{int(row['count']):,}", ha="center", fontsize=9)

x = np.arange(len(lane_stats))
w = 0.35
axes[1].bar(x - w/2, lane_stats["mean"],   w, label="Mean",   color="#f1914e")
axes[1].bar(x + w/2, lane_stats["median"], w, label="Median", color="#a0d8a0")
axes[1].set_xticks(x)
axes[1].set_xticklabels(lane_stats["lane_id"])
axes[1].set_xlabel("Lane")
axes[1].set_ylabel("Dwell time (seconds)")
axes[1].set_title("Mean vs median dwell per lane")
axes[1].legend()
save(fig, "04_lane_performance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 05 — Daily event volume over time
# ─────────────────────────────────────────────────────────────────────────────
svc_real["date"] = svc_real["timestamp"].dt.date
daily = svc_real.groupby("date").size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(daily["date"], daily["count"], alpha=0.4, color="#4e9af1")
ax.plot(daily["date"], daily["count"], color="#4e9af1", lw=1.5)
ax.set_xlabel("Date")
ax.set_ylabel("Service events per day")
ax.set_title("Daily Service Volume Over Time", fontweight="bold")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
fig.autofmt_xdate()
save(fig, "05_daily_volume.png")

# ─────────────────────────────────────────────────────────────────────────────
# 06 — Hourly traffic pattern
# ─────────────────────────────────────────────────────────────────────────────
svc_real["hour"] = svc_real["timestamp"].dt.hour
ent_real["hour"] = ent_real["timestamp"].dt.hour

hourly_svc = svc_real.groupby("hour").size()
hourly_ent = ent_real[ent_real["camera_id"].str.contains("exit", case=False, na=False)].groupby("hour").size()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hourly Traffic Patterns", fontweight="bold")

axes[0].bar(hourly_svc.index, hourly_svc.values, color="#4e9af1", edgecolor="none")
axes[0].set_xlabel("Hour of day")
axes[0].set_ylabel("Total events")
axes[0].set_title("Service events by hour")
axes[0].set_xticks(range(24))

axes[1].bar(hourly_ent.index, hourly_ent.values, color="#f1914e", edgecolor="none")
axes[1].set_xlabel("Hour of day")
axes[1].set_ylabel("Total detections")
axes[1].set_title("Entrance detections by hour")
axes[1].set_xticks(range(24))
save(fig, "06_hourly_patterns.png")

# ─────────────────────────────────────────────────────────────────────────────
# 07 — Day-of-week patterns
# ─────────────────────────────────────────────────────────────────────────────
svc_real["dow"] = svc_real["timestamp"].dt.day_name()
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_counts = svc_real.groupby("dow").size().reindex(dow_order)
dow_mean   = svc_real.groupby("dow")["total_dwell_sec"].mean().reindex(dow_order)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Day-of-Week Patterns", fontweight="bold")

axes[0].bar(dow_order, dow_counts.values, color="#a0d8a0")
axes[0].set_xticklabels(dow_order, rotation=30, ha="right")
axes[0].set_ylabel("Total service events")
axes[0].set_title("Volume by day of week")

axes[1].bar(dow_order, dow_mean.values, color="#c97dd4")
axes[1].set_xticklabels(dow_order, rotation=30, ha="right")
axes[1].set_ylabel("Mean dwell time (seconds)")
axes[1].set_title("Avg dwell by day of week")
save(fig, "07_day_of_week.png")

# ─────────────────────────────────────────────────────────────────────────────
# 08 — Model components over time (7-day sample)
# ─────────────────────────────────────────────────────────────────────────────
sample = preds_ens.dropna(subset=["ensemble_yhat","prophet_yhat","lstm_yhat","xgb_yhat"])
sample = sample.sort_values("prediction_for").tail(1000)

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(sample["prediction_for"], sample["prophet_yhat"],  label="Prophet",  alpha=0.7, lw=1.2, color="#f1914e")
ax.plot(sample["prediction_for"], sample["lstm_yhat"],     label="LSTM",     alpha=0.7, lw=1.2, color="#a0d8a0")
ax.plot(sample["prediction_for"], sample["xgb_yhat"],      label="XGBoost",  alpha=0.7, lw=1.2, color="#c97dd4")
ax.plot(sample["prediction_for"], sample["ensemble_yhat"], label="Ensemble", lw=2.0,    color="#4e9af1")
ax.set_xlabel("Time")
ax.set_ylabel("Predicted arrivals / wait (model units)")
ax.set_title("Model Components Over Time (last 1000 predictions)", fontweight="bold")
ax.legend()
fig.autofmt_xdate()
save(fig, "08_model_components_over_time.png")

# ─────────────────────────────────────────────────────────────────────────────
# 09 — Model component correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
corr_cols = ["prophet_yhat","lstm_yhat","xgb_yhat","ensemble_yhat"]
corr_data = preds_ens[corr_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_data, annot=True, fmt=".3f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Model Component Correlations", fontweight="bold")
save(fig, "09_model_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10 — Status distribution
# ─────────────────────────────────────────────────────────────────────────────
status_counts = preds_ens["status"].value_counts()
status_colors = {"OK": "#a0d8a0", "BUSY": "#f1c14e", "ALERT": "#f1504e"}
colors_list = [status_colors.get(s, "#888") for s in status_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Prediction Status Distribution", fontweight="bold")

axes[0].bar(status_counts.index, status_counts.values, color=colors_list)
axes[0].set_ylabel("Count")
axes[0].set_title("Count by status")
for i, v in enumerate(status_counts.values):
    axes[0].text(i, v + 10, f"{v:,}\n({v/len(preds_ens)*100:.1f}%)", ha="center", fontsize=9)

axes[1].pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
            colors=colors_list, startangle=90)
axes[1].set_title("Share")
save(fig, "10_status_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11 — Forecast vs actual (est_wait_minutes) by status
# ─────────────────────────────────────────────────────────────────────────────
cmp = preds_ens.dropna(subset=["est_wait_minutes","ensemble_yhat","status"]).copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Forecast vs Actual Wait — by Status", fontweight="bold")

for ax, status in zip(axes, ["OK","BUSY","ALERT"]):
    sub = cmp[cmp["status"] == status]
    if len(sub) == 0:
        ax.set_title(f"{status} (no data)")
        continue
    ax.scatter(sub["ensemble_yhat"], sub["est_wait_minutes"],
               alpha=0.4, s=15, color=status_colors.get(status, "#888"))
    lim = max(sub["ensemble_yhat"].max(), sub["est_wait_minutes"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect forecast")
    ax.set_xlabel("Predicted (ensemble_yhat)")
    ax.set_ylabel("Actual (est_wait_minutes)")
    ax.set_title(f"Status: {status}  (n={len(sub)})")
    ax.legend(fontsize=8)

save(fig, "11_forecast_vs_actual_by_status.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12 — Correction factor distribution (ALERT events only)
# ─────────────────────────────────────────────────────────────────────────────
alert = cmp[(cmp["status"] == "ALERT") & (cmp["ensemble_yhat"] > 0)].copy()
alert["ratio"] = alert["est_wait_minutes"] / alert["ensemble_yhat"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(alert["ratio"], bins=20, color="#f1504e", edgecolor="white", alpha=0.8)
ax.axvline(alert["ratio"].median(), color="black",  lw=2, linestyle="--",
           label=f"Median = {alert['ratio'].median():.2f}x")
ax.axvline(alert["ratio"].mean(),   color="orange", lw=2,
           label=f"Mean = {alert['ratio'].mean():.2f}x")
ax.set_xlabel("Correction factor (actual / predicted)")
ax.set_ylabel("Count")
ax.set_title("Correction Factor Distribution — ALERT Events Only", fontweight="bold")
ax.legend()
save(fig, "12_correction_factor_alert.png")

# ─────────────────────────────────────────────────────────────────────────────
# 13 — Model bias by status (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
bias = cmp.groupby("status").agg(
    mean_actual=("est_wait_minutes","mean"),
    mean_predicted=("ensemble_yhat","mean"),
).reset_index()

x  = np.arange(len(bias))
w  = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, bias["mean_actual"],    w, label="Actual (est_wait_min)", color="#4e9af1")
ax.bar(x + w/2, bias["mean_predicted"], w, label="Predicted (ensemble)",  color="#f1914e")
ax.set_xticks(x)
ax.set_xticklabels(bias["status"])
ax.set_ylabel("Minutes")
ax.set_title("Model Bias by Status — Actual vs Predicted", fontweight="bold")
ax.legend()
for i, row in bias.iterrows():
    ax.text(i - w/2, row["mean_actual"]    + 0.3, f"{row['mean_actual']:.1f}",    ha="center", fontsize=9)
    ax.text(i + w/2, row["mean_predicted"] + 0.3, f"{row['mean_predicted']:.1f}", ha="center", fontsize=9)
save(fig, "13_model_bias_by_status.png")

# ─────────────────────────────────────────────────────────────────────────────
# 14 — Hourly ensemble_yhat average
# ─────────────────────────────────────────────────────────────────────────────
preds_ens["hour"] = preds_ens["prediction_for"].dt.hour
hourly_pred = preds_ens.groupby("hour")["ensemble_yhat"].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(hourly_pred.index, hourly_pred.values, color="#4e9af1")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean ensemble_yhat")
ax.set_title("Average Predicted Wait by Hour of Day", fontweight="bold")
ax.set_xticks(range(24))
save(fig, "14_hourly_predictions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 15 — Equipment type in entrance events
# ─────────────────────────────────────────────────────────────────────────────
equip_ent = ent_real["equipment_type"].value_counts().head(6)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Equipment Type — Entrance Events", fontweight="bold")

axes[0].bar(equip_ent.index, equip_ent.values, color=["#888","#4e9af1","#a0d8a0"])
axes[0].set_ylabel("Count")
axes[0].set_title("Count by type")
for i, v in enumerate(equip_ent.values):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=9)

axes[1].pie(equip_ent.values, labels=equip_ent.index, autopct="%1.1f%%",
            colors=["#888","#4e9af1","#a0d8a0"], startangle=140)
axes[1].set_title("Share")
save(fig, "15_equipment_entrance_events.png")

# ─────────────────────────────────────────────────────────────────────────────
# 16 — Queue depth over time (active_head_tracks_in_lane)
# ─────────────────────────────────────────────────────────────────────────────
depth = ent_real["active_head_tracks_in_lane"].value_counts().sort_index().head(8)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(depth.index.astype(str), depth.values, color="#c97dd4")
ax.set_xlabel("Active tracks in lane")
ax.set_ylabel("Count")
ax.set_title("Queue Depth Distribution (active_head_tracks_in_lane)", fontweight="bold")
for i, v in enumerate(depth.values):
    ax.text(i, v + 500, f"{v:,}\n({v/len(ent_real)*100:.1f}%)", ha="center", fontsize=9)
save(fig, "16_queue_depth_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 17 — Has bag vs no bag
# ─────────────────────────────────────────────────────────────────────────────
bag_counts = ent_real["has_bag"].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["No bag" if not k else "Has bag" for k in bag_counts.index]
ax.pie(bag_counts.values, labels=labels, autopct="%1.1f%%",
       colors=["#a0d8a0","#f1914e"], startangle=140)
ax.set_title("Has Bag Distribution — Entrance Events", fontweight="bold")
save(fig, "17_has_bag.png")

# ─────────────────────────────────────────────────────────────────────────────
# 18 — Dwell time percentiles summary table (visual)
# ─────────────────────────────────────────────────────────────────────────────
percentiles = [10, 25, 50, 75, 90, 95, 99]
pct_values  = [np.percentile(dwell, p) for p in percentiles]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([f"p{p}" for p in percentiles], pct_values, color="#4e9af1", edgecolor="white")
ax.set_xlabel("Dwell time (seconds)")
ax.set_title("Dwell Time Percentiles — Service Events (real camera)", fontweight="bold")
for i, v in enumerate(pct_values):
    ax.text(v + 1, i, f"{v:.1f}s  ({v/60:.2f} min)", va="center", fontsize=9)
save(fig, "18_dwell_percentiles.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nDone. All charts saved to: {OUT_DIR}")
print(f"Total charts generated: 18")
