from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from prediction.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["quick", *sys.argv[1:]]))
