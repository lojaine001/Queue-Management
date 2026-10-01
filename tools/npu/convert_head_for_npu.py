"""Make the Head-Detector YOLOv9 model NPU-compatible.

    Head-Detector\\.venv\\Scripts\\python.exe tools\\npu\\convert_head_for_npu.py

The original model has a dynamic input (1x3xHxW) and a NonMaxSuppression
op with a dynamic-length output. The NPU needs static shapes and crashes
(ZE_RESULT_ERROR_DEVICE_LOST) on the in-graph NMS, so this script:

1. fixes the input to 1x3x480x640 (main.py resizes every frame to 640x480);
2. cuts the graph right before NMS, keeping the raw outputs
   `x1y1x2y2` [1,6300,4] and `main01_scores` [1,9,6300].

NMS then runs on CPU in utils/yolo.py (`nms_raw_outputs`) with the same
parameters the graph used (per class, IoU 0.4, score 0.25, 20 per class).
Works for any custom-trained model with the same layout, not just this one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference
from onnx.utils import Extractor

HD_MODELS = Path(__file__).resolve().parents[2] / "Head-Detector" / "models"
SRC = HD_MODELS / "yolov9_s_discrete_headpose_post_0100_1x3x480x640.onnx"
DST = HD_MODELS / "yolov9_s_discrete_headpose_nonms_1x3x480x640.onnx"
RAW_OUTPUTS = ["x1y1x2y2", "main01_scores"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    args = p.parse_args()

    model = onnx.load(str(args.src))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[2].ClearField("dim_param"); dims[2].dim_value = args.height
    dims[3].ClearField("dim_param"); dims[3].dim_value = args.width
    model = shape_inference.infer_shapes(model)

    sub = Extractor(model).extract_model([model.graph.input[0].name], RAW_OUTPUTS)
    sub.opset_import.extend([o for o in model.opset_import if o.domain not in {x.domain for x in sub.opset_import}])
    # Source is IR 9 but only uses opset 13; cap to what the installed onnx can check.
    sub.ir_version = min(model.ir_version, onnx.IR_VERSION)
    onnx.checker.check_model(sub)
    onnx.save(sub, str(args.dst))

    ops = {n.op_type for n in sub.graph.node}
    assert "NonMaxSuppression" not in ops
    shape = lambda vi: [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
    print(f"saved {args.dst}")
    print("  inputs :", [(i.name, shape(i)) for i in sub.graph.input])
    print("  outputs:", [(o.name, shape(o)) for o in sub.graph.output])


if __name__ == "__main__":
    main()
