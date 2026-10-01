"""Run dashboard LSTM work outside Streamlit's shared Keras state."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys


def forecast_dwell(values, seq_len, steps, epochs, floor, cap, timeout=90):
    payload = dict(values=values, seq_len=seq_len, steps=steps,
                   epochs=epochs, floor=floor, cap=cap)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"},
    )
    try:
        stdout, stderr = process.communicate(json.dumps(payload), timeout=timeout)
    except subprocess.TimeoutExpired:
        # Windows virtualenv launchers spawn a base interpreter as a child.
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            process.kill()
        process.communicate()
        raise TimeoutError("Dwell forecast exceeded its time limit") from None
    if process.returncode:
        raise RuntimeError(f"LSTM worker failed: {stderr[-2000:]}")
    predictions = json.loads(stdout)
    if (not isinstance(predictions, list) or len(predictions) != steps
            or not all(isinstance(v, (int, float)) and math.isfinite(v)
                       and floor <= v <= cap for v in predictions)):
        raise ValueError("LSTM worker returned invalid predictions")
    return predictions


def _train_and_predict(payload):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    values = np.asarray(payload["values"], dtype=float)
    seq_len = int(payload["seq_len"])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    X = np.array([scaled[i:i + seq_len] for i in range(len(scaled) - seq_len)])
    X = X.reshape(-1, seq_len, 1)
    y = scaled[seq_len:]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, 1)),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=int(payload["epochs"]), batch_size=min(16, len(X)), verbose=0)
    history = values[-seq_len:].copy()
    predictions = []
    for _ in range(int(payload["steps"])):
        sequence = scaler.transform(history.reshape(-1, 1)).reshape(1, seq_len, 1)
        scaled_value = float(model(sequence, training=False).numpy()[0][0])
        value = float(scaler.inverse_transform([[scaled_value]])[0][0])
        value = float(np.clip(value, payload["floor"], payload["cap"]))
        predictions.append(value)
        history = np.append(history[1:], value)
    return predictions


if __name__ == "__main__":
    from contextlib import redirect_stdout

    request = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        response = _train_and_predict(request)
    json.dump(response, sys.stdout, allow_nan=False)
