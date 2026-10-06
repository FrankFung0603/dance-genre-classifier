"""匯出雙輸出（logits + 最後卷積特徵圖）FP16 ONNX + 分類層權重，供即時 CAM 歸因。

產出：
    models/dance_r2plus1d_fp16.onnx   部署檔替換（雙輸出；舊單輸出版仍在 fp16_v3.onnx）
    models/fc_weights.npy             (6,512) float32，行序 = CLASSES
    stdout：parity（新舊 logits top1 一致性）+ 特徵圖形狀 + CAM sanity

CAM 原理：GAP+FC 結構下 Grad-CAM 權重恰等於 FC 權重 →
    cam_c = ReLU( Σ_k fc_w[c,k] · feat_k )   （免反傳播）

Usage: .venv/Scripts/python.exe training/export_cam.py [--parity-n 16]
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

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
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error",
           "-ss", f"{t_start:.3f}", "-i", video_path,
           "-frames:v", str(NUM_FRAMES),
           "-vf", f"fps={FPS},scale={DECODE_SIZE}:{DECODE_SIZE}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    need = NUM_FRAMES * DECODE_SIZE * DECODE_SIZE * 3
    if len(out) < need:
        raise ValueError(f"short decode {len(out)}/{need}")
    return np.frombuffer(out[:need], dtype=np.uint8).reshape(
        NUM_FRAMES, DECODE_SIZE, DECODE_SIZE, 3)


def preprocess(frames: np.ndarray) -> np.ndarray:
    x = frames.astype(np.float32) / 255.0
    off = (DECODE_SIZE - CROP_SIZE) // 2
    x = x[:, off:off + CROP_SIZE, off:off + CROP_SIZE, :]
    x = (x - MEAN) / STD
    return x.transpose(3, 0, 1, 2)[None]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-n", type=int, default=16)
    args = ap.parse_args()

    # ---------- 1) 載入 v3 權重，包成雙輸出 ----------
    ckpt = torch.load(ROOT / "models" / "dance_finetuned_v3.pt",
                      map_location="cpu", weights_only=False)
    assert list(ckpt["classes"]) == CLASSES, "class 順序與 CLASSES 不一致"
    model = torchvision.models.video.r2plus1d_18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class WithFeatures(torch.nn.Module):
        """forward → (logits, 最後卷積特徵圖 B,C,T,H,W)。"""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            x = self.m.stem(x)
            x = self.m.layer1(x)
            x = self.m.layer2(x)
            x = self.m.layer3(x)
            x = self.m.layer4(x)
            feats = x
            z = torch.flatten(self.m.avgpool(x), 1)
            return self.m.fc(z), feats

    wrapped = WithFeatures(model).eval()
    dummy = torch.randn(1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE)
    with torch.no_grad():
        l0, f0 = wrapped(dummy)
    print(f"torch sanity: logits {tuple(l0.shape)}, features {tuple(f0.shape)}")

    # ---------- 2) FP32 雙輸出 ONNX（opset 18）→ FP16 ----------
    onnx_fp32 = ROOT / "models" / "dance_r2plus1d_cam_fp32_tmp.onnx"
    torch.onnx.export(wrapped, dummy, str(onnx_fp32),
                      input_names=["frames"],
                      output_names=["logits", "features"],
                      opset_version=18, do_constant_folding=True)
    ext = onnx_fp32.with_suffix(onnx_fp32.suffix + ".data")
    if ext.exists():
        proto = onnx.load(str(onnx_fp32))
        onnx.save(proto, str(onnx_fp32))
        ext.unlink()

    import re
    import warnings

    from onnxconverter_common import float16 as f16
    proto32 = onnx.load(str(onnx_fp32))
    onnx_fp16 = ROOT / "models" / "dance_r2plus1d_fp16.onnx"  # 部署檔（舊版仍在 fp16_v3.onnx）
    # fp16 轉換偶發個別節點型別不一致（如 mean）→ 把該節點留在 FP32，迭代修到 ORT 能載入
    block_list: list[str] = []
    for attempt in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proto16 = f16.convert_float_to_float16(
                proto32, keep_io_types=True, node_block_list=list(block_list))
        onnx.save(proto16, str(onnx_fp16))
        try:
            ort.InferenceSession(str(onnx_fp16), providers=["CPUExecutionProvider"])
            break
        except Exception as e:  # noqa: BLE001
            m = re.search(r"of node \(([^)]+)\)", str(e))
            if not m or m.group(1) in block_list:
                raise
            block_list.append(m.group(1))
            print(f"fp16 block-list += {m.group(1)} (attempt {attempt + 1})")
    if block_list:
        print(f"fp16 node_block_list = {block_list}")
    onnx_fp32.unlink()
    print(f"FP16 dual-output: {onnx_fp16.stat().st_size / 1e6:.1f} MB -> {onnx_fp16.name}")

    # ---------- 3) 分類層權重 ----------
    fc_w = ckpt["state_dict"]["fc.weight"].detach().cpu().numpy().astype(np.float32)
    np.save(ROOT / "models" / "fc_weights.npy", fc_w)
    print(f"fc_weights.npy: {fc_w.shape} (row order = {CLASSES})")

    # ---------- 4) Parity：新雙輸出 vs 舊單輸出（fp16_v3.onnx） ----------
    manifest = {f"{r['video_id']}@{r['t_start']}": r
                for r in csv.DictReader(open(DATA / "windows_manifest.csv", encoding="utf-8"))}
    test_rows = [manifest[k] for k in json.loads((DATA / "splits.json").read_text())["test"]]
    sample = test_rows[:: max(1, len(test_rows) // args.parity_n)][: args.parity_n]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s_old = ort.InferenceSession(str(ROOT / "models" / "dance_r2plus1d_fp16_v3.onnx"),
                                 sess_options=so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    s_new = ort.InferenceSession(str(onnx_fp16),
                                 sess_options=so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    in_name = s_new.get_inputs()[0].name
    out_names = [o.name for o in s_new.get_outputs()]
    print(f"new model outputs: {out_names}")
    assert out_names == ["logits", "features"], "輸出順序必須 logits 在前"

    agree = 0
    diffs = []
    feats_shape = None
    for r in sample:
        x = preprocess(decode_window(r["video_path"], float(r["t_start"])))
        p_old = softmax(s_old.run(None, {"frames": x})[0][0])
        outs = s_new.run(None, {in_name: x})
        l_new, feats = outs[0], outs[1]
        p_new = softmax(np.asarray(l_new)[0].astype(np.float32))
        feats_shape = tuple(np.asarray(feats).shape)
        agree += int(p_old.argmax() == p_new.argmax())
        diffs.append(float(np.abs(p_old - p_new).max()))
    print(f"parity: top1 {agree}/{len(sample)} | max prob diff mean {np.mean(diffs):.4f}")
    print(f"features shape: {feats_shape}")

    # ---------- 5) CAM sanity：一個窗口、預測類，印 14×14 歸一化熱圖 ----------
    r = sample[0]
    x = preprocess(decode_window(r["video_path"], float(r["t_start"])))
    outs = s_new.run(None, {in_name: x})
    probs = softmax(np.asarray(outs[0])[0].astype(np.float32))
    c = int(probs.argmax())
    feats = np.asarray(outs[1], dtype=np.float32)[0]          # (C,T,H,W)
    w = fc_w[c]                                               # (C,)
    cam = np.maximum(np.einsum("k,kthw->thw", w, feats), 0.0)  # (T,H,W)
    cam2d = cam.mean(axis=0)                                  # (H,W)
    cam2d = (cam2d - cam2d.min()) / (cam2d.max() - cam2d.min() + 1e-8)
    print(f"CAM sanity [{r['video_id']}@{r['t_start']}] pred={CLASSES[c]} "
          f"({probs[c]:.2f}) shape={cam2d.shape} min={cam2d.min():.2f} max={cam2d.max():.2f}")
    for row in cam2d[::-1]:
        print("  " + " ".join(f"{v:.2f}" for v in row))


if __name__ == "__main__":
    main()
