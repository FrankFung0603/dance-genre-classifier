"""Spike 5: MediaPipe Pose（1.0 Tasks API）後端單幀 landmarks + 耗時量測。

流程：從 direct URL 用 ffmpeg 抓 1 個真實幀（rawvideo rgb24 → numpy），
餵 PoseLandmarker（IMAGE running mode），分別量測 lite / full 模型的
首次呼叫（含初始化）與穩態耗時。輸出 landmarks 數量（應為 33）。

Usage:
    python spike5_pose.py [--runs 20]
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W = H = 224
MODELS = {
    "lite": ROOT / "models" / "pose_landmarker_lite.task",
    "full": ROOT / "models" / "pose_landmarker_full.task",
}


def find_ffmpeg() -> str:
    hits = glob.glob(str(ROOT / "tools" / "ffmpeg*" / "bin" / "ffmpeg.exe"))
    if not hits:
        sys.exit("ERROR: ffmpeg.exe not found under project/tools/")
    return hits[0]


def grab_frame(direct_url: str) -> np.ndarray:
    """從 direct URL 快速 seek 後抓一幀（不落盤）。"""
    cmd = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error",
        "-ss", "30", "-i", direct_url,
        "-frames:v", "1",
        "-vf", f"scale={W}:{H}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    if len(out) < W * H * 3:
        raise RuntimeError(f"ffmpeg returned {len(out)} bytes (stream failed)")
    return np.frombuffer(out[: W * H * 3], dtype=np.uint8).reshape(H, W, 3).copy()


def bench_pose(frame: np.ndarray, variant: str, runs: int) -> dict:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODELS[variant])),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    t0 = time.perf_counter()
    detector = vision.PoseLandmarker.create_from_options(options)
    init_s = time.perf_counter() - t0

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    first_call_ms = None
    times = []
    n_landmarks = 0
    detected = False
    for i in range(runs):
        t0 = time.perf_counter()
        result = detector.detect(mp_image)
        dt_ms = (time.perf_counter() - t0) * 1000
        if i == 0:
            first_call_ms = dt_ms
            if result.pose_landmarks:
                detected = True
                n_landmarks = len(result.pose_landmarks[0])
        times.append(dt_ms)
    detector.close()

    return {
        "model": f"pose_landmarker_{variant}",
        "init_s": round(init_s, 3),
        "first_call_ms": round(first_call_ms, 1),
        "steady_mean_ms": round(float(np.mean(times[1:])), 1) if len(times) > 1 else None,
        "landmarks": n_landmarks,
        "person_detected": detected,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--url-file", default=str(HERE / "spike1_direct_url.txt"))
    args = ap.parse_args()

    try:
        frame = grab_frame(Path(args.url_file).read_text().strip())
        source = "youtube-direct-url @30s"
    except Exception as e:  # noqa: BLE001 — 退化為合成幀，仍可量測推理耗時
        frame = np.full((H, W, 3), 128, dtype=np.uint8)
        source = f"synthetic (fallback reason: {e})"[:200]

    results = {"frame_source": source, "frame_shape": list(frame.shape), "runs": args.runs, "results": []}
    for variant in ("lite", "full"):
        results["results"].append(bench_pose(frame, variant, args.runs))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
