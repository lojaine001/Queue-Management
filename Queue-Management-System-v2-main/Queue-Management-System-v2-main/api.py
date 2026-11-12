"""
Entry point kept for `uvicorn api:app` (start_all.ps1). The API itself lives in
the iqms_api package next to this file; see iqms_api/main.py for its layout.
"""
import sys
from pathlib import Path

_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:  # also when loaded by file path (the unit tests do this)
    sys.path.insert(0, _here)

from iqms_api.main import app  # noqa: E402,F401
