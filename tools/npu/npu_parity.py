"""Check that a detector gives the same boxes on NPU as on CPU.

    Head-Detector\\.venv\\Scripts\\python.exe tools\\npu\\npu_parity.py
    Head-Detector\\.venv\\Scripts\\python.exe tools\\npu\\npu_parity.py --model yolov9t --device GPU

Runs every frame through the model on CPU (FP32, the reference = today's
behaviour) and on the target device, matches boxes by IoU and reports
matched / missing / extra boxes and score drift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
HD_DIR = ROOT / "Head-Detector"
QM_DIR = ROOT / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main"


# --- Head detector (Pipeline A): raw BGR 0-255 in, [N,7] post-NMS boxes out ---
def head_prep(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, (640, 480))  # main.py resizes every frame to 640x480
    return frame.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def head_post(out: np.ndarray, score_th: float) -> np.ndarray:
    """-> [K, 6] (classid, score, x1, y1, x2, y2)"""
    out = out.reshape(-1, 7)
    out = out[out[:, 2] > score_th]
    return out[:, 1:7]


# --- yolov9t (Pipeline B): RGB 0-1 640x640 in, [1,84,8400] raw out ---
def coco_prep(frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640)) / 255.0
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def coco_post(out: np.ndarray, score_th: float, iou_th: float = 0.4) -> np.ndarray:
    """Same decode as utils/yolov9.py postprocess, in 640x640 coords."""
    pred = np.squeeze(out).T
    scores = pred[:, 4:].max(axis=1)
    keep = scores > score_th
    pred, scores = pred[keep], scores[keep]
    cls = pred[:, 4:].argmax(axis=1)
    xywh = pred[:, :4]
    idx = cv2.dnn.NMSBoxes(xywh.astype(np.int32).tolist(), scores.tolist(), score_th, iou_th)
    idx = np.array(idx).reshape(-1)
    xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)
    return np.column_stack([cls[idx], scores[idx], xyxy[idx]]) if len(idx) else np.zeros((0, 6))


MODELS = {
    "head": dict(path=HD_DIR / "models" / "yolov9_s_discrete_headpose_post_0100_1x3x480x640.onnx",
                 shape=[1, 3, 480, 640], prep=head_prep, post=head_post, score_th=0.15),
    "yolov9t": dict(path=QM_DIR / "models" / "yolov9t.onnx",
                    shape=None, prep=coco_prep, post=coco_post, score_th=0.4),
}


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(ref: np.ndarray, tgt: np.ndarray, iou_th: float = 0.5):
    """Greedy same-class IoU matching -> (matched pairs, missing, extra)."""
    pairs, used = [], set()
    if len(ref) and len(tgt):
        m = iou(ref[:, 2:6], tgt[:, 2:6])
        for i in np.argsort(-ref[:, 1]):
            for j in np.argsort(-m[i]):
                if m[i, j] < iou_th:
                    break
                if j not in used and ref[i, 0] == tgt[j, 0]:
                    pairs.append((i, j)); used.add(j); break
    return pairs, len(ref) - len(pairs), len(tgt) - len(pairs)


def compile_for(core: ov.Core, cfg: dict, device: str, precision: str | None):
    model = core.read_model(str(cfg["path"]))
    if cfg["shape"]:
        model.reshape({model.input(0).get_any_name(): cfg["shape"]})
    props = {"INFERENCE_PRECISION_HINT": precision} if precision else {}
    return core.compile_model(model, device, props)


def run_head_pipeline(device: str, frames: list[Path], precision: str) -> None:
    """End-to-end through Head-Detector's own YOLOv9 wrapper + ORT, as main.py runs it:
    original model on CPU EP (reference) vs *_nonms_* model on the OpenVINO EP target device."""
    import sys
    sys.path.insert(0, str(HD_DIR))
    from utils.yolo import YOLOv9

    def boxes_of(model, frame):
        return np.array([[b.classid, b.score, b.x1, b.y1, b.x2, b.y2] for b in model(frame, True)]).reshape(-1, 6)

    ref = YOLOv9(model_path=str(MODELS["head"]["path"]), obj_class_score_th=0.35, attr_class_score_th=0.70,
                 providers=["CPUExecutionProvider"])
    ov_opts = {"device_type": device, "precision": precision, "cache_dir": str(HD_DIR / "ov_cache")}
    tgt = YOLOv9(model_path=str(HD_DIR / "models" / "yolov9_s_discrete_headpose_nonms_1x3x480x640.onnx"),
                 obj_class_score_th=0.35, attr_class_score_th=0.70,
                 providers=[("OpenVINOExecutionProvider", ov_opts), "CPUExecutionProvider"])

    tot_ref = tot_match = tot_miss = tot_extra = 0
    score_diffs, worst, times = [], [], []
    import time
    for f in frames:
        frame = cv2.resize(cv2.imread(str(f)), (640, 480))
        r = boxes_of(ref, frame)
        t = time.perf_counter(); g = boxes_of(tgt, frame); times.append((time.perf_counter() - t) * 1000)
        pairs, miss, extra = match(r, g)
        tot_ref += len(r); tot_match += len(pairs); tot_miss += miss; tot_extra += extra
        score_diffs += [abs(r[i, 1] - g[j, 1]) for i, j in pairs]
        if miss or extra:
            worst.append(f"{f.name}: cpu={len(r)} {device}={len(g)} missing={miss} extra={extra}")

    print(f"head-pipeline device={device}/{precision} frames={len(frames)} target ms/frame mean={np.mean(times[3:]):.1f}")
    print(f"CPU boxes={tot_ref} matched={tot_match} missing_on_{device}={tot_miss} extra_on_{device}={tot_extra}")
    if score_diffs:
        print(f"score |diff| mean={np.mean(score_diffs):.4f} max={np.max(score_diffs):.4f}")
    for line in worst[:15]:
        print("  " + line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=[*MODELS, "head-pipeline"], default="head-pipeline")
    p.add_argument("--precision", default="FP16", help="head-pipeline only: OpenVINO EP precision")
    p.add_argument("--device", default="NPU")
    p.add_argument("--frames", type=Path, default=HD_DIR / "debug_snapshots")
    p.add_argument("--score-th", type=float, default=None, help="default: the pipeline's own threshold")
    args = p.parse_args()

    if args.model == "head-pipeline":
        frames = sorted(args.frames.glob("*.jpg")) + sorted(args.frames.glob("*.png"))
        run_head_pipeline(args.device, frames, args.precision)
        return

    cfg = MODELS[args.model]
    score_th = args.score_th if args.score_th is not None else cfg["score_th"]
    core = ov.Core()
    ref_model = compile_for(core, cfg, "CPU", "f32")
    tgt_model = compile_for(core, cfg, args.device, None)

    frames = sorted(args.frames.glob("*.jpg")) + sorted(args.frames.glob("*.png"))
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")

    tot_ref = tot_match = tot_miss = tot_extra = 0
    score_diffs, worst = [], []
    for f in frames:
        x = cfg["prep"](cv2.imread(str(f)))
        ref = cfg["post"](ref_model(x)[0], score_th)
        tgt = cfg["post"](tgt_model(x)[0], score_th)
        pairs, miss, extra = match(ref, tgt)
        tot_ref += len(ref); tot_match += len(pairs); tot_miss += miss; tot_extra += extra
        score_diffs += [abs(ref[i, 1] - tgt[j, 1]) for i, j in pairs]
        if miss or extra:
            worst.append(f"{f.name}: cpu={len(ref)} {args.device}={len(tgt)} missing={miss} extra={extra}")

    print(f"model={args.model} device={args.device} frames={len(frames)} score_th={score_th}")
    print(f"CPU boxes={tot_ref} matched={tot_match} missing_on_{args.device}={tot_miss} extra_on_{args.device}={tot_extra}")
    if score_diffs:
        print(f"score |diff| mean={np.mean(score_diffs):.4f} max={np.max(score_diffs):.4f}")
    for line in worst[:15]:
        print("  " + line)


if __name__ == "__main__":
    main()
