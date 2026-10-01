"""Benchmark the camera-pipeline detectors on CPU / GPU / NPU with OpenVINO.

Run with the Head-Detector venv (it has the `openvino` package):

    Head-Detector\\.venv\\Scripts\\python.exe tools\\npu\\npu_bench.py
    Head-Detector\\.venv\\Scripts\\python.exe tools\\npu\\npu_bench.py --models path\\to\\model.xml --devices NPU

For each (model, device) pair it reports: compile OK/FAIL, compile time,
mean and p95 latency. Models with dynamic input dims are reshaped to the
static shape given by --shape-override (NPU needs static shapes).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
HD_MODELS = ROOT / "Head-Detector" / "models"
QM_MODELS = ROOT / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main" / "models"

DEFAULT_MODELS = [
    HD_MODELS / "yolov9_s_discrete_headpose_post_0100_1x3x480x640.onnx",
    QM_MODELS / "yolov9t.onnx",
]

# Static shape used when a model input has dynamic dims (NCHW).
DEFAULT_STATIC_SHAPE = [1, 3, 480, 640]


def _static_shape(model: ov.Model, fallback: list[int]) -> list[int] | None:
    pshape = model.input(0).get_partial_shape()
    if pshape.is_static:
        return None
    return fallback


def bench(core: ov.Core, model_path: Path, device: str, runs: int, static_shape: list[int]) -> dict:
    result = {"model": model_path.name, "device": device}
    model = core.read_model(str(model_path))
    shape = _static_shape(model, static_shape)
    if shape is not None:
        model.reshape({model.input(0).get_any_name(): shape})
        result["reshaped"] = shape

    t0 = time.perf_counter()
    try:
        compiled = core.compile_model(model, device)
    except Exception as exc:  # unsupported op, driver issue, ...
        result["status"] = "FAIL"
        result["error"] = str(exc).splitlines()[0][:200]
        return result
    result["compile_s"] = time.perf_counter() - t0

    inp = compiled.input(0)
    dummy = np.random.rand(*inp.get_shape()).astype(np.float32)
    req = compiled.create_infer_request()
    for _ in range(5):  # warm-up
        req.infer({0: dummy})

    times = []
    for _ in range(runs):
        t = time.perf_counter()
        req.infer({0: dummy})
        times.append((time.perf_counter() - t) * 1000)
    result["status"] = "OK"
    result["mean_ms"] = float(np.mean(times))
    result["p95_ms"] = float(np.percentile(times, 95))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--devices", nargs="*", default=None, help="default: every available device")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--shape-override", nargs=4, type=int, default=DEFAULT_STATIC_SHAPE)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="OpenVINO model cache; omit to measure cold compile times")
    args = parser.parse_args()

    core = ov.Core()
    if args.cache_dir:
        core.set_property({"CACHE_DIR": str(args.cache_dir)})
    devices = args.devices or core.available_devices
    print(f"OpenVINO {ov.__version__} | devices: {core.available_devices}")
    for dev in devices:
        try:
            print(f"  {dev}: {core.get_property(dev, 'FULL_DEVICE_NAME')}")
        except Exception:
            pass

    print(f"\n{'model':<58} {'device':<6} {'status':<6} {'compile s':>9} {'mean ms':>8} {'p95 ms':>8}")
    for model_path in args.models:
        for dev in devices:
            r = bench(core, model_path, dev, args.runs, args.shape_override)
            if r["status"] == "OK":
                print(f"{r['model']:<58} {dev:<6} {'OK':<6} {r['compile_s']:>9.1f} {r['mean_ms']:>8.1f} {r['p95_ms']:>8.1f}")
            else:
                print(f"{r['model']:<58} {dev:<6} {'FAIL':<6}  {r['error']}")


if __name__ == "__main__":
    main()
