"""Spike 2: ffmpeg 從 direct URL pipe 出 rawvideo rgb24 224x224 幀流（不落盤），
量測首幀延遲與吞吐。

Usage:
    python spike2_ffmpeg.py [--duration 12] [--url-file spike1_direct_url.txt]

指標:
    - first_frame_s: ffmpeg 進程啟動 → 第一個完整幀（解析度 224x224x3 bytes）
    - frames / read_seconds: 讀取吞吐（fps）。> 8 表示讀得比即時快（buffer 可超前播放）
    - realtime_ratio: 吞吐 / 8 fps
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRAME_W = FRAME_H = 224
FRAME_BYTES = FRAME_W * FRAME_H * 3  # rgb24
TARGET_FPS = 8


def find_ffmpeg() -> str:
    hits = glob.glob(str(ROOT / "tools" / "ffmpeg*" / "bin" / "ffmpeg.exe"))
    if not hits:
        sys.exit("ERROR: ffmpeg.exe not found under project/tools/ (build env task first)")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=12.0, help="讀取量測秒數")
    ap.add_argument("--url-file", default=str(HERE / "spike1_direct_url.txt"))
    args = ap.parse_args()

    url = Path(args.url_file).read_text().strip()
    ffmpeg = find_ffmpeg()

    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", url,
        "-vf", f"fps={TARGET_FPS},scale={FRAME_W}:{FRAME_H}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    result: dict = {"ffmpeg": ffmpeg, "target_fps": TARGET_FPS}

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
    first_frame_s = None
    frames = 0
    truncated = 0
    try:
        while True:
            buf = proc.stdout.read(FRAME_BYTES)
            if not buf:
                break
            if len(buf) < FRAME_BYTES:
                truncated += 1
                break
            frames += 1
            if first_frame_s is None:
                first_frame_s = time.perf_counter() - t0
            if time.perf_counter() - t0 > args.duration:
                break
    finally:
        elapsed = time.perf_counter() - t0
        proc.kill()
        err = proc.stderr.read().decode(errors="replace").strip()

    result.update(
        first_frame_s=round(first_frame_s, 3) if first_frame_s else None,
        frames=frames,
        read_seconds=round(elapsed, 3),
        throughput_fps=round(frames / elapsed, 2) if elapsed > 0 else None,
        realtime_ratio=round((frames / elapsed) / TARGET_FPS, 2) if elapsed > 0 else None,
        truncated_frames=truncated,
        ffmpeg_stderr=err[:500] or None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
