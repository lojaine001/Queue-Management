"""Export Fisheye-Detector checkpoints (.ckpt, .pt) to ONNX (.onnx) for OpenVINO / CUDA / CPU.

Supported exports:
1. RAPiD (.ckpt -> .onnx):
   Uses `Fisheye-Detector/third_party/RAPiD/models/rapid_export.py` to export a static
   `[1, 3, H, W]` ONNX graph outputting `[1, A, 6]` (`cx, cy, w, h, angle_deg, score`).
2. YOLOv10 / YOLOv13 (.pt -> .onnx):
   Uses `ultralytics.YOLO(weights).export(format='onnx', imgsz=...)`.

Usage:
    python Fisheye-Detector/export_onnx.py --weights Fisheye-Detector/models/pL1_MWHB1024_Mar11_4000.ckpt --img-size 1024
    python Fisheye-Detector/export_onnx.py --weights Fisheye-Detector/models/yolov13_l_best_checkpoint.pt --img-size 1280
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

FISHEYE_DIR = Path(__file__).resolve().parent
RAPID_DIR = FISHEYE_DIR / "third_party" / "RAPiD"


def export_rapid_to_onnx(weights_path: Path, out_path: Path, img_size: int = 1024, opset: int = 13) -> Path:
    import torch
    import onnx

    if "pycocotools" not in sys.modules:
        pc = types.ModuleType("pycocotools")
        pc.mask = types.ModuleType("pycocotools.mask")  # type: ignore[attr-defined]
        sys.modules["pycocotools"] = pc
        sys.modules["pycocotools.mask"] = pc.mask  # type: ignore[attr-defined]

    if str(RAPID_DIR) not in sys.path:
        sys.path.insert(0, str(RAPID_DIR))

    from models.rapid_export import RAPiD  # type: ignore

    device = torch.device("cpu")
    model = RAPiD(input_hw=(img_size, img_size))
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state)
    model.eval().to(device)

    dummy_input = torch.rand(1, 3, img_size, img_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] Exporting RAPiD {weights_path.name} ({img_size}x{img_size}) -> {out_path} ...")
    export_kwargs = dict(
        verbose=False,
        opset_version=opset,
        input_names=["images"],
        output_names=["detections"],
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        **export_kwargs,
    )
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print(f"[done] Exported and verified ONNX model: {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def export_yolor_to_onnx(weights_path: Path, out_path: Path, img_size: int = 1280, opset: int = 12) -> Path:
    import inspect
    import torch
    import torch.nn as nn
    import onnx

    yolor_dir = FISHEYE_DIR / "third_party" / "YoloR"
    if str(yolor_dir) not in sys.path:
        sys.path.insert(0, str(yolor_dir))

    import models  # type: ignore
    from models.experimental import attempt_load  # type: ignore
    from utils.activations import Hardswish  # type: ignore
    from utils.general import check_img_size  # type: ignore

    device = torch.device("cpu")
    model = attempt_load(str(weights_path), map_location=device)
    gs = int(max(model.stride))
    sz = check_img_size(img_size, gs)
    dummy_input = torch.zeros(1, 3, sz, sz, device=device)

    for _, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()
    model.model[-1].export = False  # Keep Detect inference concatenation [1, A, 5+C]
    _ = model(dummy_input)

    class _YoloRInferenceWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            return self.base(x)[0]

    wrapped = _YoloRInferenceWrapper(model).eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] Exporting YOLOR {weights_path.name} ({sz}x{sz}) -> {out_path} ...")
    export_kwargs = dict(
        verbose=False,
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(wrapped, dummy_input, str(out_path), **export_kwargs)
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print(f"[done] Exported and verified YOLOR ONNX model: {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def export_ultralytics_to_onnx(weights_path: Path, out_path: Path, img_size: int = 1280, opset: int = 13) -> Path:
    y13_dir = FISHEYE_DIR / "third_party" / "YoloV13"
    if y13_dir.is_dir() and str(y13_dir) not in sys.path:
        sys.path.insert(0, str(y13_dir))
    from ultralytics import YOLO  # type: ignore
    import shutil

    model = YOLO(str(weights_path))
    exported = Path(model.export(format="onnx", imgsz=img_size, opset=opset, simplify=False))
    if exported.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(out_path))
    print(f"[done] Exported Ultralytics ONNX model: {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="Input checkpoint (.ckpt or .pt)")
    ap.add_argument("--output", help="Output .onnx path (default: alongside input with .onnx suffix)")
    ap.add_argument("--img-size", type=int, default=0, help="Input resolution (default: 1024 for RAPiD, 1280 for YOLO)")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"Weights file not found: {weights_path}")

    is_rapid = weights_path.suffix.lower() == ".ckpt" or "rapid" in weights_path.name.lower() or "pl1_" in weights_path.name.lower()
    is_yolor = "yolor" in weights_path.name.lower()
    img_size = args.img_size or (1024 if is_rapid else 1280)
    if args.output:
        out_path = Path(args.output)
    else:
        stem = weights_path.stem
        prefix = "rapid_" if is_rapid and not stem.lower().startswith("rapid") else ""
        out_path = weights_path.with_name(f"{prefix}{stem}.onnx")

    if is_rapid:
        export_rapid_to_onnx(weights_path, out_path, img_size=img_size, opset=args.opset)
    elif is_yolor:
        export_yolor_to_onnx(weights_path, out_path, img_size=img_size, opset=min(args.opset, 12))
    else:
        export_ultralytics_to_onnx(weights_path, out_path, img_size=img_size, opset=args.opset)


if __name__ == "__main__":
    main()
