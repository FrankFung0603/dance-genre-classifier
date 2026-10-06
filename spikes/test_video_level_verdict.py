"""一次性分析：contemporary 測試影片「整支看完」的主判斷是否正確。

模擬前端總結卡的投票規則（每秒一票、主判取非 unknown 票最多者）：
對 test 集每支 contemporary 來源影片逐窗推理（前處理/模型與 export_eval.py 完全一致），
按影片聚合票數，輸出每支影片的票面分佈與主判。

Usage: .venv/Scripts/python.exe spikes/test_video_level_verdict.py
"""
import csv
import glob
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["ballet", "breakdance", "contemporary", "hiphop", "salsa", "unknown"]
NUM_FRAMES, DECODE_SIZE, CROP_SIZE, FPS = 16, 256, 224, 8
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.221459, 0.216989], dtype=np.float32)
MODEL = ROOT / "models" / "dance_r2plus1d_fp16_v3.onnx"


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
    return np.frombuffer(out[:need], dtype=np.uint8).reshape(NUM_FRAMES, DECODE_SIZE, DECODE_SIZE, 3)


def preprocess(frames: np.ndarray) -> np.ndarray:
    x = frames.astype(np.float32) / 255.0
    off = (DECODE_SIZE - CROP_SIZE) // 2
    x = x[:, off:off + CROP_SIZE, off:off + CROP_SIZE, :]
    x = (x - MEAN) / STD
    return x.transpose(3, 0, 1, 2)[None]


def np_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    provs = ort.get_available_providers()
    prov = "DmlExecutionProvider" if "DmlExecutionProvider" in provs else "CPUExecutionProvider"
    print(f"providers: {provs} -> using {prov}", flush=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(MODEL), sess_options=so, providers=[prov, "CPUExecutionProvider"])

    test_keys = set(json.loads((DATA / "splits.json").read_text())["test"])
    rows = [r for r in csv.DictReader(open(DATA / "windows_manifest.csv", encoding="utf-8"))
            if f"{r['video_id']}@{r['t_start']}" in test_keys and r["genre"] == "contemporary"]
    by_video: dict[str, list] = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)

    print(f"\ncontemporary test: {len(rows)} windows / {len(by_video)} videos\n", flush=True)
    total_ok = 0
    for vid, vrows in sorted(by_video.items()):
        votes: Counter = Counter()
        for r in vrows:
            x = preprocess(decode_window(r["video_path"], float(r["t_start"])))
            logits = sess.run(None, {"frames": x})[0]
            votes[CLASSES[int(np_softmax(logits)[0].argmax())]] += 1
        n = sum(votes.values())
        non_unknown = {g: c for g, c in votes.items() if g != "unknown"}
        top, topn = (max(non_unknown.items(), key=lambda kv: kv[1]) if non_unknown else ("unknown", votes["unknown"]))
        ok = top == "contemporary"
        total_ok += ok
        dist = ", ".join(f"{g}:{c}" for g, c in votes.most_common())
        print(f"{'OK  ' if ok else 'MISS'} {vid}  windows={n}  [{dist}]  -> 主判 {top} ({topn / n:.0%})", flush=True)
    print(f"\n影片級主判正確：{total_ok}/{len(by_video)}", flush=True)


if __name__ == "__main__":
    main()
