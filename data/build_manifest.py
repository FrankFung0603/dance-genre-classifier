"""對每支原始影片解碼一次（8fps, 256×256, rgb24, 不落盤），切 2s 窗（stride 可調），
用 K400 預訓練 R(2+1)D-18（ONNX, DML）打分，產出 windows manifest CSV。

打分欄位用途（filter_split.py 消費）：
    dance_prob: K400 舞蹈相關類別機率總和
    top1_label / top1_prob: K400 top-1（人工抽查用）

前提：
    - models/r2plus1d_18_k400.onnx（+ .data）
    - data/k400_categories.json（由 dump_categories 產生）

Usage:
    python build_manifest.py [--stride 2] [--batch 8] [--append]
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw"
FPS = 8
WIN_FRAMES = 16
FRAME_SIZE = 256
FRAME_BYTES = FRAME_SIZE * FRAME_SIZE * 3

# K400 官方歸一化常數（R(2+1)D-18 預訓練所用）
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.221459, 0.216989], dtype=np.float32)

# 舞蹈訊號關鍵字：包含舞蹈類型本身 + 高度舞蹈化的動作
DANCE_KEYWORDS = ("danc", "ballet", "breakdanc", "salsa", "zumba", "hula", "cheerleading")


def find_ffmpeg() -> str:
    hits = glob.glob(str(ROOT / "tools" / "ffmpeg*" / "bin" / "ffmpeg.exe"))
    if not hits:
        sys.exit("ERROR: ffmpeg.exe not found")
    return hits[0]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def stream_windows(path: str, stride_s: float):
    """逐幀讀取，yield (t_start, window_uint8[16,256,256,3])。記憶體只保留最後 16 幀。"""
    cmd = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-vf", f"fps={FPS},scale={FRAME_SIZE}:{FRAME_SIZE}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    step = max(1, int(stride_s * FPS))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=10**7)
    buf: deque = deque(maxlen=WIN_FRAMES)
    frame_idx = 0
    next_start = 0  # 下一個窗口的起始幀號
    try:
        while True:
            chunk = proc.stdout.read(FRAME_BYTES)
            if not chunk or len(chunk) < FRAME_BYTES:
                break
            buf.append(np.frombuffer(chunk, dtype=np.uint8).reshape(FRAME_SIZE, FRAME_SIZE, 3))
            if frame_idx == next_start + WIN_FRAMES - 1 and len(buf) == WIN_FRAMES:
                yield (frame_idx - WIN_FRAMES + 1) / FPS, np.stack(buf)
                next_start += step
            frame_idx += 1
    finally:
        proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--append", action="store_true",
                    help="增量模式：跳過 manifest 已有的 video_id，只處理新影片（append 寫入）")
    args = ap.parse_args()

    cats = json.loads((HERE / "k400_categories.json").read_text())
    dance_idx = np.array([i for i, c in enumerate(cats) if any(k in c.lower() for k in DANCE_KEYWORDS)])
    print(f"K400 dance-related classes: {len(dance_idx)} -> {[cats[i] for i in dance_idx][:15]} ...")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ROOT / "models" / "r2plus1d_18_k400.onnx"),
                                sess_options=so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    out_csv = HERE / "windows_manifest.csv"
    done_ids: set[str] = set()
    if args.append and out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            done_ids = {r["video_id"] for r in csv.DictReader(f)}
        print(f"append mode: {len(done_ids)} videos already in manifest")

    videos = sorted(glob.glob(str(RAW / "*" / "*.mp4"))
                    + glob.glob(str(RAW / "*" / "*.webm"))
                    + glob.glob(str(RAW / "*" / "*.mkv")))
    if args.append:
        videos = [v for v in videos if Path(v).stem not in done_ids]
    print(f"videos: {len(videos)} (to process)")
    if not videos:
        print("nothing to do")
        return

    n_windows = 0
    append_mode = args.append and done_ids
    with open(out_csv, "a" if append_mode else "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not append_mode:
            w.writerow(["genre", "video_id", "video_path", "t_start", "t_end",
                        "dance_prob", "top1_label", "top1_prob"])
        for vp in videos:
            genre = Path(vp).parent.name
            vid = Path(vp).stem
            batch_t, batch_w = [], []

            def flush() -> None:
                nonlocal n_windows
                if not batch_w:
                    return
                # 打分模型為 static batch-1（DML EP 不支援 dynamic-batch Reshape）→ 逐窗推理
                for i in range(len(batch_w)):
                    x = batch_w[i].astype(np.float32)             # (16,256,256,3)
                    x = (x / 255.0 - MEAN) / STD
                    x = torch.from_numpy(x.transpose(3, 0, 1, 2))  # (3,16,256,256)
                    # 打分模型以 224² 匯出 → 空間縮放（時間維不變）
                    x = torch.nn.functional.interpolate(
                        x.unsqueeze(0), size=(WIN_FRAMES, 224, 224),
                        mode="trilinear", align_corners=False,
                    ).numpy()
                    logits = sess.run(None, {input_name: x})[0]
                    probs = softmax(logits)[0]
                    dp = float(probs[dance_idx].sum())
                    top1 = int(probs.argmax())
                    w.writerow([genre, vid, vp, round(batch_t[i], 3), round(batch_t[i] + 2.0, 3),
                                round(dp, 4), cats[top1], round(float(probs[top1]), 4)])
                n_windows += len(batch_w)
                batch_t.clear()
                batch_w.clear()

            try:
                for t_start, win in stream_windows(vp, args.stride):
                    batch_t.append(t_start)
                    batch_w.append(win)
                    if len(batch_w) >= args.batch:
                        flush()
                flush()
            except Exception as e:  # noqa: BLE001 — 單片失敗不中斷整體
                print(f"  WARN {vid}: {e}", flush=True)
            print(f"  done {genre}/{vid} (total windows: {n_windows})", flush=True)

    print(f"\nmanifest -> {out_csv} ({n_windows} windows)")


if __name__ == "__main__":
    main()
