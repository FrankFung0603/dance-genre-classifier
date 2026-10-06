"""匯出 ONNX（opset 18 單檔）+ FP16 轉換 + 精度 parity + 完整測試集指標 + 延遲基準。

⚠️ 注意（2026-10-06 起）：線上部署檔 `dance_r2plus1d_fp16.onnx` 必須是**雙輸出**
（logits + features，供即時 CAM 歸因）。本腳本會把它覆蓋成**單輸出**版——
重新匯出部署模型請改用 `training/export_cam.py`；本腳本僅用於完整指標評測
（評測只用 logits，單輸出不影響指標數字），跑完記得再用 export_cam.py 復原部署檔。

輸出：
    models/dance_r2plus1d.onnx       FP32 單檔
    models/dance_r2plus1d_fp16.onnx  FP16（⚠️ 單輸出，覆蓋部署檔，見上）
    docs/phase1_metrics.json         macro F1 / per-class P/R/F1
    docs/phase1_confusion_matrix.png
    stdout: parity、延遲數字

Usage:
    python export_eval.py [--parity-n 32]
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
CLASSES = ["ballet", "breakdance", "contemporary", "hiphop", "salsa", "unknown"]
NUM_FRAMES, DECODE_SIZE, CROP_SIZE, FPS = 16, 256, 224, 8
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.221459, 0.216989], dtype=np.float32)


def find_ffmpeg() -> str:
    return glob.glob(str(ROOT / "tools" / "ffmpeg*" / "bin" / "ffmpeg.exe"))[0]


def decode_window(video_path: str, t_start: float) -> np.ndarray:
    cmd = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{t_start:.3f}", "-i", video_path,
        "-frames:v", str(NUM_FRAMES),
        "-vf", f"fps={FPS},scale={DECODE_SIZE}:{DECODE_SIZE}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    need = NUM_FRAMES * DECODE_SIZE * DECODE_SIZE * 3
    if len(out) < need:
        raise ValueError(f"short decode {len(out)}/{need}")
    return np.frombuffer(out[:need], dtype=np.uint8).reshape(NUM_FRAMES, DECODE_SIZE, DECODE_SIZE, 3)


def preprocess(frames: np.ndarray) -> np.ndarray:
    """uint8 THWC → normalized float32 (1,3,16,224,224)，center crop 224。"""
    x = frames.astype(np.float32) / 255.0
    off = (DECODE_SIZE - CROP_SIZE) // 2
    x = x[:, off:off + CROP_SIZE, off:off + CROP_SIZE, :]
    x = (x - MEAN) / STD
    return x.transpose(3, 0, 1, 2)[None]


def make_session(path: str, provider: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=[provider, "CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-n", type=int, default=32)
    args = ap.parse_args()

    # ---------- 1) 匯出 FP32 ONNX（opset 18，單檔 inline 權重） ----------
    ckpt = torch.load(ROOT / "models" / "dance_finetuned.pt", map_location="cpu", weights_only=False)
    model = torchvision.models.video.r2plus1d_18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_fp32 = ROOT / "models" / "dance_r2plus1d.onnx"
    dummy = torch.randn(1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE)
    # static batch-1：DML EP 不支援 dynamic-batch Reshape；部署模式本就是單 clip 1Hz
    torch.onnx.export(model, dummy, str(onnx_fp32), input_names=["frames"],
                      output_names=["logits"], opset_version=18, do_constant_folding=True)
    # 新版 exporter 可能寫外部檔 → 載入後重存為單檔
    ext = onnx_fp32.with_suffix(onnx_fp32.suffix + ".data")
    if ext.exists():
        proto = onnx.load(str(onnx_fp32))  # 自動讀入外部權重
        onnx.save(proto, str(onnx_fp32))
        ext.unlink()
    print(f"FP32 ONNX: {onnx_fp32.stat().st_size / 1e6:.1f} MB (single file)")

    # ---------- 2) FP16 ----------
    from onnxconverter_common import float16 as f16
    fp32_proto = onnx.load(str(onnx_fp32))
    fp16_proto = f16.convert_float_to_float16(fp32_proto, keep_io_types=True)
    onnx_fp16 = ROOT / "models" / "dance_r2plus1d_fp16.onnx"
    onnx.save(fp16_proto, str(onnx_fp16))
    print(f"FP16 ONNX: {onnx_fp16.stat().st_size / 1e6:.1f} MB")

    # ---------- 3) Parity（FP32 vs FP16，DML） ----------
    manifest = {f"{r['video_id']}@{r['t_start']}": r
                for r in csv.DictReader(open(DATA / "windows_manifest.csv", encoding="utf-8"))}
    test_rows = [manifest[k] for k in json.loads((DATA / "splits.json").read_text())["test"]]
    sample = test_rows[:: max(1, len(test_rows) // args.parity_n)][: args.parity_n]

    s32 = make_session(str(onnx_fp32), "DmlExecutionProvider")
    s16 = make_session(str(onnx_fp16), "DmlExecutionProvider")
    agree, diffs = 0, []
    for r in sample:
        x = preprocess(decode_window(r["video_path"], float(r["t_start"])))
        p32 = np_softmax(s32.run(None, {"frames": x})[0])[0]
        p16 = np_softmax(s16.run(None, {"frames": x})[0])[0]
        agree += int(p32.argmax() == p16.argmax())
        diffs.append(float(np.abs(p32 - p16).max()))
    print(f"parity: top1 agreement {agree}/{len(sample)} | max prob diff mean {np.mean(diffs):.4f}")

    # ---------- 4) 測試集指標（FP16 ORT = 部署等價路徑；static batch-1 逐窗推理） ----------
    ys, ps = [], []
    for i, r in enumerate(test_rows):
        x = preprocess(decode_window(r["video_path"], float(r["t_start"])))
        logits = s16.run(None, {"frames": x})[0]
        ps.append(int(np_softmax(logits)[0].argmax()))
        ys.append(CLASSES.index(r["genre"]))
        if (i + 1) % 50 == 0:
            print(f"  eval {i + 1}/{len(test_rows)}", flush=True)
    report = classification_report(ys, ps, target_names=CLASSES, digits=3, zero_division=0)
    macro_f1 = f1_score(ys, ps, average="macro")
    print(f"\n=== TEST (FP16 ORT, n={len(ys)}) ===")
    print(report)
    print(f"macro F1 = {macro_f1:.4f}")

    cm = confusion_matrix(ys, ps).tolist()
    (ROOT / "docs" / "phase1_metrics.json").write_text(json.dumps({
        "model": "R(2+1)D-18 finetuned (K400 init)",
        "classes": CLASSES,
        "n_test": len(ys),
        "macro_f1": round(float(macro_f1), 4),
        "classification_report": classification_report(ys, ps, target_names=CLASSES,
                                                       digits=3, zero_division=0, output_dict=True),
        "confusion_matrix": cm,
        "parity_top1_agreement": f"{agree}/{len(sample)}",
    }, indent=2, ensure_ascii=False))

    # ---------- 5) Confusion matrix 圖 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.array(cm), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for r_ in range(len(CLASSES)):
        for c_ in range(len(CLASSES)):
            ax.text(c_, r_, cm[r_][c_], ha="center", va="center",
                    color="white" if cm[r_][c_] > np.max(cm) / 2 else "black", fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion Matrix (test n={len(ys)}, macro F1={macro_f1:.3f})")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "phase1_confusion_matrix.png", dpi=150)
    print(f"-> docs/phase1_metrics.json, docs/phase1_confusion_matrix.png")

    # ---------- 6) 延遲基準 ----------
    def bench(sess, n_warm=3, n=20) -> float:
        x = np.random.randn(1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE).astype(np.float32)
        for _ in range(n_warm):
            sess.run(None, {"frames": x})
        ts = []
        for _ in range(n):
            t0 = time.time()
            sess.run(None, {"frames": x})
            ts.append((time.time() - t0) * 1000)
        return float(np.mean(ts))

    print(f"\nlatency FP16 (DML): {bench(s16):.1f} ms/clip | FP32: {bench(s32):.1f} ms/clip")


def np_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
