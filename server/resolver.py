"""輸入 URL → 可播放來源。

三種輸入：
- YouTube URL → yt-dlp 解析 direct URL（cache by video_id；direct URL 會過期，TTL 30min）
- http(s) 影片 URL（MP4 fallback）→ 直接交給 ffmpeg
- 本機存在路徑 → file 來源（僅開發/測試）
"""
import re
import sys
import time
from pathlib import Path
from typing import Literal

from yt_dlp import YoutubeDL

sys.stdout.reconfigure(encoding="utf-8")

YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/shorts/)([\w-]{11})"
)

# 與 Phase 0/1 相同的選片策略：video-only DASH avc1 ≤480p（解碼快、對 224² 輸入足夠）
YDL_FORMAT = "bestvideo[height<=480][vcodec^=avc1]/bestvideo[height<=480]/best[height<=480]"
YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "format": YDL_FORMAT,
}

_CACHE_TTL_S = 30 * 60
_cache: dict[str, tuple[float, str]] = {}  # video_id -> (monotonic_deadline, direct_url)


class ResolveError(Exception):
    pass


Source = tuple[Literal["youtube", "mp4", "file"], str]


def classify(url: str) -> Source:
    """把使用者輸入分類成 (kind, playable)。YouTube 不在此解析（lazy，見 resolve）。"""
    m = YOUTUBE_RE.search(url)
    if m:
        return ("youtube", m.group(1))
    if url.startswith(("http://", "https://")):
        return ("mp4", url)
    p = Path(url)
    if p.is_file():
        return ("file", str(p.resolve()))
    raise ResolveError(f"cannot handle input: {url!r} (not YouTube/URL/existing file)")


def resolve_youtube(video_id: str, force: bool = False) -> str:
    """video_id → direct media URL（帶 TTL cache；force 重解析）。"""
    now = time.monotonic()
    if not force:
        hit = _cache.get(video_id)
        if hit and hit[0] > now:
            return hit[1]
    try:
        with YoutubeDL(YDL_OPTS) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception as e:  # noqa: BLE001
        raise ResolveError(f"yt-dlp resolve failed for {video_id}: {e}") from e
    direct = info.get("url")
    if not direct:
        raise ResolveError(f"no direct url in yt-dlp result for {video_id}")
    _cache[video_id] = (now + _CACHE_TTL_S, direct)
    return direct
