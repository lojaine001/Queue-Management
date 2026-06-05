"""
test_owlv2_raw.py — Diagnostic: run OWLv2 on overhead snapshots with zero filtering.

No track-matching, no IoU gating — just raw detections above a floor threshold.
This tells us whether OWLv2 can see personal bags at all from the overhead angle.

Usage:
    python test_owlv2_raw.py
    python test_owlv2_raw.py --thresh 0.05 --frames 5
    python test_owlv2_raw.py --image path/to/specific.jpg
"""
from __future__ import annotations

import argparse
import os
import glob

import cv2
import numpy as np
import torch


# ── Query groups — mirrors owlv2_detector.py but we can edit freely here ──────
QUERY_GROUPS: dict[str, list[str]] = {
    "trolley": [
        "shopping trolley", "supermarket trolley", "shopping cart",
        "grocery cart", "metal trolley",
    ],
    "store_basket": [
        "shopping basket", "hand basket", "wire basket",
        "plastic basket", "carry basket",
    ],
    "personal_bag": [
        # existing
        "handbag", "backpack", "shoulder bag", "tote bag",
        "crossbody bag", "canvas bag", "leather bag", "purse",
        "satchel", "cloth bag", "personal bag", "reusable bag",
        # overhead / queue context
        "bag carried by person", "grocery bag", "fabric bag",
        "bag on floor", "bag beside person",
        "wheeled shopping bag", "rolling shopping bag",
        "personal wheeled grocery trolley",
        "fabric bag on frame", "granny cart",
        "bag from above", "bag seen from above",
        "shopping bag from above",
    ],
}

ALL_QUERIES: list[str] = []
QUERY_TO_LABEL: dict[int, str] = {}
for _label, _queries in QUERY_GROUPS.items():
    for _q in _queries:
        QUERY_TO_LABEL[len(ALL_QUERIES)] = _label
        ALL_QUERIES.append(_q)

MODEL_ID = "google/owlv2-base-patch16-ensemble"

LABEL_COLOURS = {
    "trolley":      (0,   200, 255),
    "store_basket": (50,  255, 50),
    "personal_bag": (200, 100, 255),
}


def load_model(device: str):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    print(f"Loading OWLv2 ({MODEL_ID}) on {device} ...")
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model     = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
    model.to(device).eval()
    print("Model ready.")
    return processor, model


def run_on_frame(
    frame_bgr: np.ndarray,
    processor,
    model,
    device: str,
    thresh: float,
) -> list[tuple[int, int, int, int, str, float, str]]:
    """Return list of (x1,y1,x2,y2, label, score, query_text)."""
    from PIL import Image as PILImage

    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    inputs = processor(text=[ALL_QUERIES], images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        score_threshold=thresh,
    )[0]

    fh, fw = frame_bgr.shape[:2]
    detections = []
    for box, score, label_idx in zip(
        results["boxes"].cpu().numpy(),
        results["scores"].cpu().numpy(),
        results["labels"].cpu().numpy(),
    ):
        x1 = int(max(0,  box[0]))
        y1 = int(max(0,  box[1]))
        x2 = int(min(fw, box[2]))
        y2 = int(min(fh, box[3]))
        label     = QUERY_TO_LABEL.get(int(label_idx), "unknown")
        query_txt = ALL_QUERIES[int(label_idx)]
        detections.append((x1, y1, x2, y2, label, float(score), query_txt))

    return sorted(detections, key=lambda d: d[5], reverse=True)


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list,
    show_all_labels: bool = True,
) -> np.ndarray:
    out = frame_bgr.copy()
    for x1, y1, x2, y2, label, score, query_txt in detections:
        colour = LABEL_COLOURS.get(label, (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{label} {score:.2f}"
        if show_all_labels:
            text += f"  [{query_txt}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (20, 20, 20), -1)
        cv2.putText(out, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh",  type=float, default=0.05,
                        help="Raw detection threshold (default 0.05 — very low, shows everything)")
    parser.add_argument("--frames",  type=int,   default=3,
                        help="How many snapshots to process (default 3)")
    parser.add_argument("--image",   type=str,   default="",
                        help="Path to a specific image (overrides --frames)")
    parser.add_argument("--out_dir", type=str,   default="owlv2_test_output",
                        help="Folder to save annotated images")
    parser.add_argument("--device",  type=str,   default="",
                        help="cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor, model = load_model(device)

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect images to process
    if args.image:
        images = [args.image]
    else:
        pattern = os.path.join("debug_snapshots", "debug_*.jpg")
        all_snaps = sorted(glob.glob(pattern),
                           key=os.path.getmtime, reverse=True)
        images = all_snaps[:args.frames]
        if not images:
            print("No debug snapshots found. Run with --image path/to/frame.jpg")
            return

    print(f"\nRunning on {len(images)} image(s) with thresh={args.thresh}\n")
    print(f"Total queries: {len(ALL_QUERIES)}")
    print()

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [SKIP] Could not read {img_path}")
            continue

        dets = run_on_frame(frame, processor, model, device, args.thresh)

        print(f"=== {os.path.basename(img_path)} — {len(dets)} detection(s) ===")
        if dets:
            for x1, y1, x2, y2, label, score, query_txt in dets:
                print(f"  {label:15s}  score={score:.3f}  query='{query_txt}'  "
                      f"box=({x1},{y1})-({x2},{y2})")
        else:
            print("  (no detections above threshold)")
        print()

        annotated = draw_detections(frame, dets)
        out_name  = f"result_{os.path.basename(img_path)}"
        out_path  = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, annotated)
        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
