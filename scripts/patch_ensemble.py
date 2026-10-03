import sys
from pathlib import Path

p = Path("Queue-Management/Queue-Management-System-v2-main/Queue-Management-System-v2-main/ensemble_predict.py")
with open(p, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("\u2500", "-")
lines = text.splitlines(keepends=True)

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if "# -- MODEL 2 -- LSTM" in line or "# -- MODEL 2 - LSTM" in line:
        new_lines.append(line)
        new_lines.append("    if tf is None:\n")
        new_lines.append("        _log('[LSTM] TensorFlow not installed, falling back to Prophet baseline for LSTM component')\n")
        new_lines.append("        lstm_vals = prophet_vals.copy()\n")
        new_lines.append("    else:\n")
        i += 1
        while i < len(lines) and ("if tf is None:" in lines[i] or "lstm_vals = prophet_vals.copy()" in lines[i] or "else:" in lines[i] or "_log('[LSTM]" in lines[i] or '_log("[LSTM]' in lines[i]):
            i += 1
        while i < len(lines) and not ("# -- MODEL 3" in lines[i]):
            new_lines.append("    " + lines[i])
            i += 1
        continue
    elif "# -- MODEL 3 -- XGBOOST" in line or "# -- MODEL 3 - XGBOOST" in line:
        new_lines.append(line)
        new_lines.append("    if xgb is None:\n")
        new_lines.append("        _log('[XGBoost] XGBoost not installed, falling back to Prophet baseline for XGBoost component')\n")
        new_lines.append("        xgb_vals = prophet_vals.copy()\n")
        new_lines.append("    else:\n")
        i += 1
        while i < len(lines) and ("if xgb is None:" in lines[i] or "xgb_vals = prophet_vals.copy()" in lines[i] or "else:" in lines[i] or "_log('[XGBoost]" in lines[i]):
            i += 1
        while i < len(lines) and not ("# -- ENSEMBLE" in lines[i]):
            new_lines.append("    " + lines[i])
            i += 1
        continue
    else:
        new_lines.append(line)
        i += 1

with open(p, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Successfully patched ensemble_predict.py")
