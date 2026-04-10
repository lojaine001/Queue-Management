import psycopg2
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ── Config ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "iqms",
    "user":     "postgres",
    "password": "0000"
}

W_PROPHET     = 0.40   # Prophet weight in ensemble
W_LSTM        = 0.30   # LSTM weight
W_XGB         = 0.30   # XGBoost weight
SEQUENCE_LEN  = 12     # 12 x 5 min = 60 min lookback for LSTM
FORECAST_STEPS = 12    # predict next 60 minutes (12 x 5 min)

# ── 1. Load real data from DB ────────────────────────────────────────────────
conn = psycopg2.connect(**DB_CONFIG)
print("[Ensemble] Connected to PostgreSQL ✓")

df_real = pd.read_sql("""
    SELECT
        date_trunc('minute', timestamp) AS minute,
        COUNT(*)                        AS entry_count
    FROM entrance_events
    GROUP BY date_trunc('minute', timestamp)
    ORDER BY minute
""", conn)
conn.close()
print(f"[Ensemble] Loaded {len(df_real)} minutes of real data from DB")

# ── 2. Simulated historical data (7 days) ───────────────────────────────────
print("[Ensemble] Generating 7-day simulated history...")
sim_rows = []
start_date = datetime.now() - timedelta(days=7)

for day in range(7):
    current_day = start_date + timedelta(days=day)
    is_weekend = current_day.weekday() >= 5

    for hour in range(8, 22):
        for minute in range(0, 60, 5):
            t = current_day.replace(hour=hour, minute=minute, second=0, microsecond=0)

            if   8 <= hour < 10: base = 3 if is_weekend else 2
            elif 10 <= hour < 13: base = 8 if is_weekend else 5
            elif 13 <= hour < 15: base = 6 if is_weekend else 7
            elif 15 <= hour < 18: base = 5 if is_weekend else 4
            elif 18 <= hour < 20: base = 4 if is_weekend else 8
            else:                 base = 2

            sim_rows.append({'ds': t, 'y': max(0, int(np.random.poisson(base)))})

df_sim = pd.DataFrame(sim_rows)

# ── 3. Merge real + simulated ────────────────────────────────────────────────
df_real_r = df_real.rename(columns={'minute': 'ds', 'entry_count': 'y'})
df_real_r['ds'] = pd.to_datetime(df_real_r['ds']).dt.tz_localize(None)

df = pd.concat([df_sim, df_real_r], ignore_index=True)
df = df.drop_duplicates(subset='ds').sort_values('ds').reset_index(drop=True)
print(f"[Ensemble] Total training rows: {len(df)}")

now = pd.Timestamp.now() - pd.Timedelta(hours=2)   # timezone adjustment (same as prophet_predict.py)

# ────────────────────────────────────────────────────────────────────────────
# MODEL 1 — PROPHET
# ────────────────────────────────────────────────────────────────────────────
print("\n[Prophet] Training...")
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    interval_width=0.80
)
prophet_model.fit(df)

# Generate future timestamps directly from now (avoids empty results when
# training data ends in the past and make_future_dataframe doesn't reach now)
future_timestamps = [
    now + pd.Timedelta(minutes=5 * (i + 1)) for i in range(FORECAST_STEPS)
]
future_df  = pd.DataFrame({'ds': future_timestamps})
forecast   = prophet_model.predict(future_df)
prophet_preds = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)
prophet_vals  = prophet_preds['yhat'].clip(lower=0).values
print("[Prophet] Done ✓")

# ────────────────────────────────────────────────────────────────────────────
# MODEL 2 — LSTM  (TensorFlow / Keras)
# ────────────────────────────────────────────────────────────────────────────
print("\n[LSTM] Preparing sequences...")

scaler   = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(df['y'].values.reshape(-1, 1))

def make_sequences(data, seq_len):
    X, y_out = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y_out.append(data[i + seq_len])
    return np.array(X), np.array(y_out)

X_lstm, y_lstm = make_sequences(y_scaled, SEQUENCE_LEN)
print(f"[LSTM] Training on {len(X_lstm)} sequences (epochs=20)...")

lstm_model = Sequential([
    LSTM(64, input_shape=(SEQUENCE_LEN, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32, verbose=0)
print("[LSTM] Done ✓")

# Recursive forecast: feed each prediction back as next input
last_seq = y_scaled[-SEQUENCE_LEN:].copy()
lstm_scaled_preds = []
for _ in range(FORECAST_STEPS):
    inp      = last_seq.reshape(1, SEQUENCE_LEN, 1)
    next_val = lstm_model.predict(inp, verbose=0)[0][0]
    lstm_scaled_preds.append(next_val)
    last_seq = np.append(last_seq[1:], [[next_val]], axis=0)

lstm_vals = scaler.inverse_transform(
    np.array(lstm_scaled_preds).reshape(-1, 1)
).flatten().clip(min=0)

# ────────────────────────────────────────────────────────────────────────────
# MODEL 3 — XGBOOST
# ────────────────────────────────────────────────────────────────────────────
print("\n[XGBoost] Building features...")

FEATURE_COLS = [
    'hour', 'minute_of_hour', 'day_of_week', 'is_weekend',
    'lag_1', 'lag_2', 'lag_3', 'lag_12', 'rolling_mean_6'
]

def build_features(data_df):
    d = data_df.copy()
    d['hour']           = d['ds'].dt.hour
    d['minute_of_hour'] = d['ds'].dt.minute
    d['day_of_week']    = d['ds'].dt.dayofweek
    d['is_weekend']     = (d['day_of_week'] >= 5).astype(int)
    d['lag_1']          = d['y'].shift(1)
    d['lag_2']          = d['y'].shift(2)
    d['lag_3']          = d['y'].shift(3)
    d['lag_12']         = d['y'].shift(12)       # 1 hour ago
    d['rolling_mean_6'] = d['y'].shift(1).rolling(6).mean()  # 30-min avg
    return d.dropna()

df_feat = build_features(df)
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(df_feat[FEATURE_COLS], df_feat['y'])
print("[XGBoost] Done ✓")

# Recursive forecast using future_timestamps aligned with Prophet
recent_y = df['y'].tolist()
xgb_vals = []

for ts in future_timestamps:
    ts_dt = pd.Timestamp(ts)
    row = {
        'hour':           ts_dt.hour,
        'minute_of_hour': ts_dt.minute,
        'day_of_week':    ts_dt.dayofweek,
        'is_weekend':     int(ts_dt.dayofweek >= 5),
        'lag_1':          recent_y[-1],
        'lag_2':          recent_y[-2],
        'lag_3':          recent_y[-3],
        'lag_12':         recent_y[-12] if len(recent_y) >= 12 else recent_y[0],
        'rolling_mean_6': np.mean(recent_y[-6:]) if len(recent_y) >= 6 else np.mean(recent_y),
    }
    pred = float(xgb_model.predict(pd.DataFrame([row]))[0])
    pred = max(0.0, pred)
    xgb_vals.append(pred)
    recent_y.append(pred)

xgb_vals = np.array(xgb_vals)

# ────────────────────────────────────────────────────────────────────────────
# ENSEMBLE  (weighted average)
# ────────────────────────────────────────────────────────────────────────────
print("\n[Ensemble] Combining predictions (Prophet 40% / LSTM 30% / XGBoost 30%)...")
ensemble_vals = (
    W_PROPHET * prophet_vals +
    W_LSTM    * lstm_vals    +
    W_XGB     * xgb_vals
).clip(min=0).round(1)

# ── Console output ───────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ENSEMBLE PREDICTED CUSTOMER ENTRIES — NEXT 60 MINUTES")
print("=" * 75)
print(f"  {'Time':<10} {'Prophet':>10} {'LSTM':>10} {'XGBoost':>10} {'Ensemble':>10}")
print("-" * 75)
for i, row in prophet_preds.iterrows():
    print(
        f"  {row['ds'].strftime('%H:%M'):<10}"
        f" {prophet_vals[i]:>10.1f}"
        f" {lstm_vals[i]:>10.1f}"
        f" {xgb_vals[i]:>10.1f}"
        f" {ensemble_vals[i]:>10.1f}"
    )
print("=" * 75)

print("\n  QUEUE WAIT TIME ESTIMATES  (Ensemble)")
print("=" * 75)
print(f"  {'Time':<10} {'Entries':>10} {'Est. Wait':>12}   Status")
print("-" * 75)
for i, row in prophet_preds.iterrows():
    entries = ensemble_vals[i]
    wait    = round((entries * 3) / 2, 1)          # 3 min/customer, 2 lanes
    status  = "OK" if wait < 5 else "BUSY" if wait < 10 else "ALERT"
    print(f"  {row['ds'].strftime('%H:%M'):<10} {entries:>10} {wait:>8} min   {status}")
print("=" * 75)

# ── Save to PostgreSQL ───────────────────────────────────────────────────────
print("\n[DB] Saving predictions to queue_predictions...")

conn2 = psycopg2.connect(**DB_CONFIG)
cur   = conn2.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS queue_predictions (
        id                SERIAL PRIMARY KEY,
        predicted_at      TIMESTAMP   NOT NULL,
        prediction_for    TIMESTAMP   NOT NULL,
        prophet_yhat      NUMERIC(8,2),
        lstm_yhat         NUMERIC(8,2),
        xgb_yhat          NUMERIC(8,2),
        ensemble_yhat     NUMERIC(8,2),
        est_wait_minutes  NUMERIC(8,2),
        status            VARCHAR(10)
    )
""")

predicted_at = datetime.now()
for i, row in prophet_preds.iterrows():
    entries = float(ensemble_vals[i])
    wait    = round((entries * 3) / 2, 2)
    status  = "OK" if wait < 5 else "BUSY" if wait < 10 else "ALERT"
    cur.execute("""
        INSERT INTO queue_predictions
            (predicted_at, prediction_for, prophet_yhat, lstm_yhat,
             xgb_yhat, ensemble_yhat, est_wait_minutes, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        predicted_at,
        row['ds'].to_pydatetime(),
        round(float(prophet_vals[i]), 2),
        round(float(lstm_vals[i]),    2),
        round(float(xgb_vals[i]),     2),
        round(entries,                2),
        wait,
        status
    ))

conn2.commit()
cur.close()
conn2.close()
print(f"[DB] Saved {FORECAST_STEPS} rows to queue_predictions ✓")
print("\n[Ensemble] All done.")
