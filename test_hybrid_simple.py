import pandas as pd
from prediction import pipeline as pred_pipeline
from prediction import hybrid_wait

result = pred_pipeline.run_prediction_pipeline(source='REAL', days=30)
print("Forecast rows:", len(result.get('forecast', [])))
print("Future open timestamps needed")

view = hybrid_wait.build_hybrid_wait_view(result)
hybrid_curve = view.get('hybrid_wait_curve', pd.DataFrame())

print("\nHybrid wait curve: {} rows".format(len(hybrid_curve)))
if not hybrid_curve.empty:
    print("Columns:", list(hybrid_curve.columns))
    print("First row:\n", hybrid_curve.iloc[0])
    print("Last row:\n", hybrid_curve.iloc[-1])
else:
    print("ERROR: hybrid_wait_curve is empty!")
