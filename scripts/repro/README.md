# Failure Reproduction Scripts

This directory contains standalone reproduction scripts that demonstrate the root causes of the intermittent failures observed in the coding agent, scheduler, and Streamlit dashboard.

---

## 1. File Contention & Lock Collision (`repro_write_json.py`)

- **Root Cause:** On Windows, `os.replace(tmp, path)` fails with `PermissionError: [WinError 5] Access is denied` if another process holds an open file handle without `FILE_SHARE_DELETE`. In high-frequency scheduler polling loops (saving state every 1 second), read/write collisions happen frequently.
- **How to Run:**
  ```powershell
  python scripts/repro/repro_write_json.py
  ```
- **Expected Failure:**
  `PermissionError: [WinError 5] Access is denied: ... -> ...`

---

## 2. Unicode Console Encoding Crash (`repro_unicode.py`)

- **Root Cause:** Log and CLI print statements containing non-ASCII glyphs (e.g. `→`, `✓`, `✗`, `—`) trigger `UnicodeEncodeError: 'charmap' codec can't encode character` when standard output runs under Windows `cp1252` encoding.
- **How to Run:**
  ```powershell
  python scripts/repro/repro_unicode.py
  ```
- **Expected Failure:**
  `UnicodeEncodeError: cp1252 cannot encode '\u2192'` / `'\u2713'`

---

## 3. Missing Package Import Crash (`repro_missing_imports.py`)

- **Root Cause:** `dashboard.py` unconditionally imports `xgboost as xgb` at the module level. In environments where `xgboost` is not installed (e.g., system Python), Streamlit crashes immediately before rendering any dashboard UI.
- **How to Run:**
  ```powershell
  python scripts/repro/repro_missing_imports.py
  ```
- **Expected Failure:**
  `ModuleNotFoundError: No module named 'xgboost'`
