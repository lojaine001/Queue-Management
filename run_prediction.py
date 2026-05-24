from __future__ import annotations

import os
import sys


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from prediction.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
