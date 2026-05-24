import pandas as pd
from prediction import pipeline as pred_pipeline
from prediction import hybrid_wait

print("=" * 60)
print("1. Loading base prediction...")
result = pred_pipeline.run_prediction_pipeline(source='REAL', days=30)
print("[OK] Forecast: {} rows".format(len(result.get('forecast', []))))

print("\n2. Building hybrid wait view...")
view = hybrid_wait.build_hybrid_wait_view(
    result,
    current_weight_pct=70,
    auto_shift_history=True,
    recent_window_min=45,
    lane_strategy="auto",
)

print("[OK] View built successfully")
print("Hybrid wait curve: {} rows".format(len(view.get('hybrid_wait_curve', []))))
print("Wait 15m: {}".format(view.get('wait_15m')))
print("Wait 30m: {}".format(view.get('wait_30m')))

print("\n3. Checking hybrid_wait_viz functions...")
from simulator import hybrid_wait_viz

wait_agg = hybrid_wait_viz.future_wait_display_frame(view)
print("future_wait_display_frame: {} rows".format(len(wait_agg)))

raw_waits = hybrid_wait_viz.future_values_frame(view)
print("future_values_frame: {} rows".format(len(raw_waits)))

print("\n" + "=" * 60)
if len(wait_agg) == 0 and len(raw_waits) == 0:
    print("ISSUE: Hybrid wait curve exists but visualization frames are empty!")
elif len(wait_agg) > 0 and len(raw_waits) > 0:
    print("OK: All data appears to be flowing correctly")
else:
    print("WARN: Mixed results - check above")
