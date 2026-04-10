import psycopg2
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Connect to PostgreSQL ──
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="iqms",
    user="postgres",
    password="0000"
)

print("[Prophet] Connected to PostgreSQL ✓")

# ── Load real data ──
query = """
    SELECT 
        date_trunc('minute', timestamp) as minute,
        COUNT(*) as entry_count
    FROM entrance_events
    GROUP BY date_trunc('minute', timestamp)
    ORDER BY minute
"""
df_real = pd.read_sql(query, conn)
conn.close()

print(f"[Prophet] Loaded {len(df_real)} minutes of real data from DB")

# ── Generate simulated historical data (7 days) ──
# This gives Prophet enough history to learn daily/weekly patterns
print("[Prophet] Generating 7 days of simulated historical data...")

sim_rows = []
start_date = datetime.now() - timedelta(days=7)

for day in range(7):
    current_day = start_date + timedelta(days=day)
    is_weekend = current_day.weekday() >= 5

    for hour in range(8, 22):  # store open 8am to 10pm
        for minute in range(0, 60, 5):  # 5-minute buckets
            t = current_day.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Simulate realistic traffic patterns
            if 8 <= hour < 10:
                base = 3 if is_weekend else 2
            elif 10 <= hour < 13:
                base = 8 if is_weekend else 5
            elif 13 <= hour < 15:
                base = 6 if is_weekend else 7   # lunch peak
            elif 15 <= hour < 18:
                base = 5 if is_weekend else 4
            elif 18 <= hour < 20:
                base = 4 if is_weekend else 8   # after-work peak
            else:
                base = 2

            count = max(0, int(np.random.poisson(base)))
            sim_rows.append({'ds': t, 'y': count})

df_sim = pd.DataFrame(sim_rows)

# ── Merge real data on top of simulated ──
df_real_renamed = df_real.rename(columns={'minute': 'ds', 'entry_count': 'y'})
df_real_renamed['ds'] = pd.to_datetime(df_real_renamed['ds']).dt.tz_localize(None)

df_combined = pd.concat([df_sim, df_real_renamed], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset='ds').sort_values('ds').reset_index(drop=True)

print(f"[Prophet] Total training rows: {len(df_combined)}")

# ── Train Prophet ──
print("[Prophet] Training model...")
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    interval_width=0.80
)
model.fit(df_combined)
print("[Prophet] Model trained ✓")

# ── Predict next 60 minutes ──
future = model.make_future_dataframe(periods=12, freq='5min')
forecast = model.predict(future)

# ── Show predictions for next 60 minutes ──
now = pd.Timestamp.now() - pd.Timedelta(hours=2)
upcoming = forecast[forecast['ds'] > now][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(12)
upcoming['yhat'] = upcoming['yhat'].clip(lower=0).round(1)
upcoming['yhat_lower'] = upcoming['yhat_lower'].clip(lower=0).round(1)
upcoming['yhat_upper'] = upcoming['yhat_upper'].clip(lower=0).round(1)

print("\n" + "="*60)
print("  PREDICTED CUSTOMER ENTRIES — NEXT 60 MINUTES")
print("="*60)
print(f"  {'Time':<12} {'Predicted':>10} {'Min':>8} {'Max':>8}")
print("-"*60)
for _, row in upcoming.iterrows():
    time_str = row['ds'].strftime('%H:%M')
    print(f"  {time_str:<12} {row['yhat']:>10} {row['yhat_lower']:>8} {row['yhat_upper']:>8}")
print("="*60)

# ── Queue wait time estimation ──
print("\n  QUEUE WAIT TIME ESTIMATES")
print("="*60)
print(f"  {'Time':<12} {'Entries':>10} {'Est. Wait':>12}")
print("-"*60)
for _, row in upcoming.iterrows():
    time_str = row['ds'].strftime('%H:%M')
    entries = max(0, row['yhat'])
    # Simple model: each customer takes ~3 min, 2 lanes open
    wait = round((entries * 3) / 2, 1)
    status = "🟢 OK" if wait < 5 else "🟡 BUSY" if wait < 10 else "🔴 ALERT"
    print(f"  {time_str:<12} {entries:>10} {wait:>8} min  {status}")
print("="*60)
print("\n[Prophet] Done.")