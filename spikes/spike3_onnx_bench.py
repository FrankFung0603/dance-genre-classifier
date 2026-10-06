"""Spike 3b: ONNX Runtime 對 16 幀輸入推理基準（CPU EP vs DML GPU EP）。

Windows 上 onnxruntime-directml 同時提供 CPUExecutionProvider 與
DmlExecutionProvider（不需 CUDA toolkit）。若 DML 不支援某些 op，
記錄錯誤作為已知限制（Docker 內改用 CUDA EP）。

Usage:
    python spike3_onnx_bench.py [--model ../models/r2plus1d_18_k400.onnx]
                                [--warmup 3] [--runs 10]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


def bench_session(model_path: str, provider: str, warmup: int, runs: int) -> dict:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=so, providers=[provider])
    input_name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape  # e.g. [1,3,16,224,224] or with strings
    x = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: x})

    times = []
    out_shape = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000)
        out_shape = list(out[0].shape)

    return {
        "provider": provider,
        "input_shape": shape,
        "output_shape": out_shape,
        "warmup_runs": warmup,
        "bench_runs": runs,
        "mean_ms": round(float(np.mean(times)), 1),
        "min_ms": round(float(np.min(times)), 1),
        "max_ms": round(float(np.max(times)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE.parent / "models" / "r2plus1d_18_k400.onnx"))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model} (run spike3_onnx_export.py first)")

    results = {"model": args.model, "ort_version": ort.__version__, "devices": []}
    for provider in ["CPUExecutionProvider", "DmlExecutionProvider"]:
        try:
            results["devices"].append(bench_session(args.model, provider, args.warmup, args.runs))
        except Exception as e:  # noqa: BLE001 — spike: 記錄失敗也是結果
            results["devices"].append({"provider": provider, "error": f"{type(e).__name__}: {e}"[:300]})

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
