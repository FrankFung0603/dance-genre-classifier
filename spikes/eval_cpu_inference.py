"""CPU EP 完整測試集評測 + 延遲基準（面試交付：CPU result）。

兩種模式：
  1) --onnx PATH：直接評測指定 ONNX（預設用 v3 單輸出部署檔，權重=線上部署模型）
  2) --pt PATH --save-onnx PATH：載入 .pt checkpoint → 匯出 FP32+FP16 ONNX（同 export_eval.py
     的匯出路徑，但寫到指定檔名、不覆蓋部署檔）→ 評測 FP16

評測路徑與 export_eval.py 逐字相同（live ffmpeg decode + 同前處理），
差異僅 provider：CPUExecutionProvider（export_eval.py 用 DML）→ 數字可直接對比。

Usage:
    python spikes/eval_cpu_inference.py --onnx models/dance_r2plus1d_fp16_v3.onnx --out docs/cpu_inference_metrics.json
    python spikes/eval_cpu_inference.py --onnx models/dance_r2plus1d_v3.onnx --out docs/cpu_fp32_latency.json --skip-eval
    python spikes/eval_cpu_inference.py --pt models/dance_finetuned_cpu.pt --save-onnx models/dance_r2plus1d_cpu_fp16.onnx --out docs/cpu_training_eval.json
"""
import argparse
import csv
import datetime
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "training"))
from export_eval import CLASSES, NUM_FRAMES, CROP_SIZE, decode_window, np_softmax, preprocess  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def export_onnx_from_pt(pt_path: Path, out_fp32: Path, out_fp16: Path) -> None:
    """與 export_eval.py 步驟 1–2 完全相同的匯出（opset 18 單檔 FP32 → FP16），只改輸出檔名。"""
    import onnx
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = torchvision.models.video.r2plus1d_18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE)
    torch.onnx.export(model, dummy, str(out_fp32), input_names=["frames"],
                      output_names=["logits"], opset_version=18, do_constant_folding=True)
    ext = out_fp32.with_suffix(out_fp32.suffix + ".data")
    if ext.exists():
        proto = onnx.load(str(out_fp32))
        onnx.save(proto, str(out_fp32))
        ext.unlink()
    from onnxconverter_common import float16 as f16
    fp16_proto = f16.convert_float_to_float16(onnx.load(str(out_fp32)), keep_io_types=True)
    onnx.save(fp16_proto, str(out_fp16))
    print(f"exported: {out_fp32.name} ({out_fp32.stat().st_size/1e6:.1f} MB) -> "
          f"{out_fp16.name} ({out_fp16.stat().st_size/1e6:.1f} MB)", flush=True)


def bench(sess, n_warm=3, n=20) -> dict:
    x = np.random.randn(1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE).astype(np.float32)
    for _ in range(n_warm):
        sess.run(None, {"frames": x})
    ts = []
    for _ in range(n):
        t0 = time.time()
        sess.run(None, {"frames": x})
        ts.append((time.time() - t0) * 1000)
    return {"mean_ms": round(float(np.mean(ts)), 1), "min_ms": round(float(np.min(ts)), 1),
            "max_ms": round(float(np.max(ts)), 1), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--pt", default=None, help="checkpoint → 匯出 ONNX（--save-onnx）再評測")
    ap.add_argument("--save-onnx", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--bench-n", type=int, default=20)
    ap.add_argument("--skip-eval", action="store_true", help="只跑延遲基準，不跑測試集")
    args = ap.parse_args()

    if args.pt:
        assert args.save_onnx, "--pt 需要 --save-onnx"
        save = Path(args.save_onnx)
        export_onnx_from_pt(ROOT / args.pt,
                            save.with_name(save.stem.replace("_fp16", "") + save.suffix), save)
        onnx_path = save
    else:
        assert args.onnx, "需要 --onnx 或 --pt"
        onnx_path = ROOT / args.onnx

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=[args.provider])
    print(f"session: {onnx_path.name} | providers={sess.get_providers()} | "
          f"inputs={[i.name for i in sess.get_inputs()]} outputs={[o.name for o in sess.get_outputs()]}",
          flush=True)
    out_name = sess.get_outputs()[0].name

    result = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "onnx_file": onnx_path.name,
        "provider": sess.get_providers()[0],
        "ort_version": ort.__version__,
        "torch_threads": torch.get_num_threads(),
        "os_cpu_count": multiprocessing.cpu_count(),
        "note": "export_eval.py 同前處理（live ffmpeg decode），僅 provider 不同，數字可直接與官方 DML 結果對比",
    }

    # ---------- 延遲基準（同 export_eval.py 的 bench：隨機輸入，3 warm + n timed） ----------
    result["latency_bench"] = bench(sess, n=args.bench_n)
    print(f"latency ({args.provider}): {result['latency_bench']['mean_ms']} ms/clip "
          f"(min {result['latency_bench']['min_ms']})", flush=True)

    if args.skip_eval:
        (ROOT / args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"-> {args.out}")
        return

    # ---------- 完整測試集（與 export_eval.py 相同：逐窗 decode + 推理） ----------
    manifest = {f"{r['video_id']}@{r['t_start']}": r
                for r in csv.DictReader(open(ROOT / "data" / "windows_manifest.csv", encoding="utf-8"))}
    test_rows = [manifest[k] for k in json.loads((ROOT / "data" / "splits.json").read_text())["test"]]
    ys, ps, infer_ms = [], [], []
    t_start = time.time()
    for i, r in enumerate(test_rows):
        x = preprocess(decode_window(r["video_path"], float(r["t_start"])))
        t0 = time.time()
        logits = sess.run([out_name], {"frames": x})[0]
        infer_ms.append((time.time() - t0) * 1000)
        ps.append(int(np_softmax(logits)[0].argmax()))
        ys.append(CLASSES.index(r["genre"]))
        if (i + 1) % 50 == 0:
            el = time.time() - t_start
            print(f"  eval {i + 1}/{len(test_rows)} | {el:.0f}s elapsed | "
                  f"eta {el / (i + 1) * (len(test_rows) - i - 1):.0f}s", flush=True)

    macro_f1 = f1_score(ys, ps, average="macro")
    result.update({
        "n_test": len(ys),
        "macro_f1": round(float(macro_f1), 4),
        "accuracy": round(float(accuracy_score(ys, ps)), 4),
        "classification_report": classification_report(ys, ps, target_names=CLASSES,
                                                       digits=3, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(ys, ps).tolist(),
        "inference_ms_mean_on_test": round(float(np.mean(infer_ms)), 1),
        "inference_ms_p50_on_test": round(float(np.percentile(infer_ms, 50)), 1),
        "testset_total_seconds": round(time.time() - t_start, 1),
    })
    (ROOT / args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== TEST ({args.provider}, {onnx_path.name}, n={len(ys)}) ===")
    print(classification_report(ys, ps, target_names=CLASSES, digits=3, zero_division=0))
    print(f"macro F1 = {macro_f1:.4f} | inference mean {result['inference_ms_mean_on_test']} ms/clip")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
