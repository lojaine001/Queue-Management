"""Download pre-trained fisheye person detection weights into Fisheye-Detector/models/.

Supports:
1. RAPiD overhead fisheye checkpoints from GitHub Releases (duanzhiihao/RAPiD v0.1):
   - pL1_MWHB1024_Mar11_4000.ckpt (default, COCO + MW-R + HABBOF @ 1024x1024)
   - pL1_MWCP1024_Apr14_3000.ckpt (COCO + MW-R + CEPDOF @ 1024x1024)
   - pL1_HBCP1024_Apr14_3000.ckpt (COCO + HABBOF + CEPDOF @ 1024x1024)
   - pL1_MWHB608_Mar11_4500.ckpt  (COCO + MW-R + HABBOF @ 608x608)
   - pL1_MWCP608_Apr14_5500.ckpt  (COCO + MW-R + CEPDOF @ 608x608)
   - pL1_HBCP608_Apr14_6000.ckpt  (COCO + HABBOF + CEPDOF @ 608x608)

2. UT_AICITY2025_Track4 Fisheye8K + VisDrone checkpoints from Google Drive:
   - yolor_d6_best_checkpoint.pt  (1rrySwLYitocJ0hSzz5hiFXyNjJaVhPob)
   - yolov10_x_best_checkpoint.pt (1pfKQURsycm4MfgQNfhmI65BFXJLhZYc5)
   - yolov13_l_best_checkpoint.pt (1rQeOer-aL5bzZBzmhjNPbfwr-hYmXxua)

Usage:
    python Fisheye-Detector/download_models.py
    python Fisheye-Detector/download_models.py --rapid-all
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.request
from pathlib import Path
import requests

MODELS_DIR = Path(__file__).resolve().parent / "models"

RAPID_RELEASE_BASE = "https://github.com/duanzhiihao/RAPiD/releases/download/v0.1"
RAPID_CHECKPOINTS = {
    "pL1_MWHB1024_Mar11_4000.ckpt": f"{RAPID_RELEASE_BASE}/pL1_MWHB1024_Mar11_4000.ckpt",
    "pL1_MWCP1024_Apr14_3000.ckpt": f"{RAPID_RELEASE_BASE}/pL1_MWCP1024_Apr14_3000.ckpt",
    "pL1_HBCP1024_Apr14_3000.ckpt": f"{RAPID_RELEASE_BASE}/pL1_HBCP1024_Apr14_3000.ckpt",
    "pL1_MWHB608_Mar11_4500.ckpt": f"{RAPID_RELEASE_BASE}/pL1_MWHB608_Mar11_4500.ckpt",
    "pL1_MWCP608_Apr14_5500.ckpt": f"{RAPID_RELEASE_BASE}/pL1_MWCP608_Apr14_5500.ckpt",
    "pL1_HBCP608_Apr14_6000.ckpt": f"{RAPID_RELEASE_BASE}/pL1_HBCP608_Apr14_6000.ckpt",
}

AICITY_GDRIVE_CHECKPOINTS = {
    "yolor_d6_best_checkpoint.pt": "1rrySwLYitocJ0hSzz5hiFXyNjJaVhPob",
    "yolov10_x_best_checkpoint.pt": "1pfKQURsycm4MfgQNfhmI65BFXJLhZYc5",
    "yolov13_l_best_checkpoint.pt": "1rQeOer-aL5bzZBzmhjNPbfwr-hYmXxua",
}


def download_url(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / (1024*1024):.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {dest.name} <- {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100.0 / total
                    print(f"\r  {downloaded / (1024*1024):6.1f} / {total / (1024*1024):6.1f} MB ({pct:5.1f}%)", end="", flush=True)
        print()
    tmp.replace(dest)
    print(f"[done] {dest} ({dest.stat().st_size / (1024*1024):.1f} MB)")


def download_gdrive(file_id: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / (1024*1024):.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore
        gdown.download(id=file_id, output=str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            return
    except ImportError:
        pass

    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    resp = session.get(base_url, params={"id": file_id}, stream=True, timeout=60)
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        html = resp.text
        resp.close()
        # Extract confirm token / uuid from Google Drive large-file warning page
        m_uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
        m_confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html)
        params = {"id": file_id, "export": "download", "confirm": m_confirm.group(1) if m_confirm else "t"}
        if m_uuid:
            params["uuid"] = m_uuid.group(1)
        download_endpoint = "https://drive.usercontent.google.com/download"
        resp = session.get(download_endpoint, params=params, stream=True, timeout=60)

    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = 0
    print(f"[download] {dest.name} (Google Drive id={file_id})")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100.0 / total
                print(f"\r  {downloaded / (1024*1024):6.1f} / {total / (1024*1024):6.1f} MB ({pct:5.1f}%)", end="", flush=True)
    print()
    resp.close()
    if tmp.stat().st_size < 100_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Google Drive returned an HTML page instead of binary weights for {dest.name} (id={file_id}).")
    tmp.replace(dest)
    print(f"[done] {dest} ({dest.stat().st_size / (1024*1024):.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(MODELS_DIR), help="Target directory (default: Fisheye-Detector/models)")
    ap.add_argument("--rapid-only", action="store_true", help="Download only RAPiD weights")
    ap.add_argument("--rapid-all", action="store_true", help="Download all 6 RAPiD checkpoints (1024 and 608)")
    ap.add_argument("--aicity-only", action="store_true", help="Download only UT_AICITY2025_Track4 weights")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.aicity_only:
        rapid_keys = list(RAPID_CHECKPOINTS.keys()) if args.rapid_all else ["pL1_MWHB1024_Mar11_4000.ckpt"]
        for name in rapid_keys:
            download_url(RAPID_CHECKPOINTS[name], out_dir / name)

    if not args.rapid_only:
        for name, file_id in AICITY_GDRIVE_CHECKPOINTS.items():
            download_gdrive(file_id, out_dir / name)


if __name__ == "__main__":
    main()
