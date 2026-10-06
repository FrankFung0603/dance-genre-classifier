"""Spike 1: yt-dlp 解析 YouTube URL → direct media URL，量測解析耗時。

Usage:
    python spike1_ytdlp.py [youtube_url]

Output:
    - stdout: JSON 結果（含 resolve_seconds）
    - spikes/spike1_direct_url.txt: 完整 direct URL（供 spike2 使用）
"""
import json
import sys
import time
from pathlib import Path

from yt_dlp import YoutubeDL

sys.stdout.reconfigure(encoding="utf-8")  # Windows console cp950 保險

DEFAULT_URL = "https://www.youtube.com/watch?v=9bZkp7q19f0"  # PSY - GANGNAM STYLE（舞蹈、高可用性）
HERE = Path(__file__).resolve().parent

# 推理管線只需要視頻不需要音頻（播放端 IFrame 自帶音頻）。
# 優先 video-only DASH (avc1/H.264, https 直鏈)，退而求其次 <=480 的任何格式。
YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "format": "bestvideo[height<=480][vcodec^=avc1][protocol*=https]/bestvideo[height<=480]/best[height<=480]",
}


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    t0 = time.perf_counter()
    with YoutubeDL(YDL_OPTS) as ydl:
        info = ydl.extract_info(url, download=False)
    resolve_s = time.perf_counter() - t0

    result = {
        "url": url,
        "title": info.get("title"),
        "duration_s": info.get("duration"),
        "format_id": info.get("format_id"),
        "format": info.get("format"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "vcodec": info.get("vcodec"),
        "acodec": info.get("acodec"),
        "ext": info.get("ext"),
        "protocol": info.get("protocol"),
        "resolve_seconds": round(resolve_s, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    direct = info.get("url")
    if direct:
        (HERE / "spike1_direct_url.txt").write_text(direct)
        print(f"\ndirect URL saved -> {HERE / 'spike1_direct_url.txt'}")
        print(f"direct URL (truncated): {direct[:100]}...")
    else:
        print("\nWARNING: no direct url in info dict (format may be a manifest)")


if __name__ == "__main__":
    main()
