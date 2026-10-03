import pickle
import pandas as pd
from pathlib import Path

model_path = Path("Queue-Management/Queue-Management-System-v2-main/Queue-Management-System-v2-main/models/prophet_queue_3m.pkl")

print(f"Loading Prophet model from {model_path}...")
with open(model_path, "rb") as f:
    m = pickle.load(f)

# Prophet training was on local wall-clock time
# Generate 15-min intervals from 08:00 to 20:00 for today
future = pd.date_range("2026-09-27 08:00:00", "2026-09-27 20:00:00", freq="15min")
df_future = pd.DataFrame({"ds": future})
fc = m.predict(df_future)
print("\nProphet prediction for 2026-09-27:")
for _, r in fc[["ds", "yhat"]].iterrows():
    print(f"{r['ds'].strftime('%H:%M')}: yhat={r['yhat']:.2f}")
