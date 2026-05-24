import sys
import pandas as pd
from prediction import pipeline as pred_pipeline

try:
    result = pred_pipeline.run_prediction_pipeline(source='REAL', days=30)
    print('✓ Pipeline succeeded')
    print(f'  Forecast rows: {len(result.get("forecast", []))}')
    print(f'  Wait 15m: {result.get("wait_15m")}')
    print(f'  Wait 30m: {result.get("wait_30m")}')
except ValueError as e:
    print(f'✗ ValueError: {e}')
except Exception as e:
    print(f'✗ {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
