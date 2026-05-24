import pandas as pd
from prediction import pipeline as pred_pipeline
from prediction import hybrid_wait
from simulator import hybrid_wait_viz

result = pred_pipeline.run_prediction_pipeline(source='REAL', days=30)
view = hybrid_wait.build_hybrid_wait_view(result)
fig = hybrid_wait_viz.fig_hybrid_wait_forecast(view)
print('traces:', len(fig.data))
for i, trace in enumerate(fig.data):
    print(i, trace.name, len(trace.x), len(trace.y))
print('layout x axis type', fig.layout.xaxis.type if 'xaxis' in fig.layout else 'missing')
print('layout y axis type', fig.layout.yaxis.type if 'yaxis' in fig.layout else 'missing')
print('first x', fig.data[0].x[0] if len(fig.data) else 'none')
print('first y', fig.data[0].y[0] if len(fig.data) else 'none')
