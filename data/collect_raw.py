"""收集原始訓練影片（弱標籤：整支影片 = 一個 genre）。

- 每類目標 TARGET_PER_GENRE 支，60–300 秒，≤480p，video-only（訓練只需幀）
- download_archive 防重複：重跑腳本 = 斷點續收
- 進度寫 data/raw/collect_log.jsonl，stdout 印每支摘要

Usage:
    python collect_raw.py [--target 14]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from yt_dlp import YoutubeDL

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ARCHIVE = RAW / ".archive.txt"

GENRES: dict[str, list[str]] = {
    "ballet": ["ballet performance stage", "ballet class", "ballet variation solo"],
    # 弱三類（phase-1 互混嚴重）：補純風格、風格特徵強的查詢
    "breakdance": ["breakdance battle", "bboy breaking competition",
                   "breaking bboy battle final", "breakdance powermove"],
    # v3 增量：用戶實測 salsa 線上預測率低於測試集 F1 0.994 → 測試集分佈偏窄，
    # 補多樣場景（social/婚禮/街頭/比賽、不同燈光鏡位）
    "salsa": ["salsa dancing social", "salsa performance", "salsa congress performance",
              "salsa social dancing night", "salsa street performance",
              "casino rueda salsa dancing", "salsa dance festival"],
    # v3 增量：hiphop recall 0.680（誤入 contemporary），補多樣查詢
    "hiphop": ["hip hop dance choreography", "hip hop dance battle",
               "hip hop freestyle street dance", "urban dance choreography",
               "hip hop dance class routine", "street dance performance"],
    # v3 增量：contemporary F1 0.567 最弱，補足風格光譜（lyrical/modern/company/improv）
    "contemporary": ["contemporary dance performance", "modern dance solo",
                     "contemporary dance solo stage", "lyrical dance performance",
                     "contemporary dance company", "modern dance choreography",
                     "contemporary dance improvisation"],
    # Unknown / No Dance：明確非舞蹈的日常內容（長視頻也可，窗口數有 cap）
    "unknown": ["city walking tour street", "cooking tutorial", "tech conference talk",
                "news broadcast studio", "nature documentary wildlife", "dashcam drive"],
}

# 每類目標支數：v3 增量輪——弱類（contemporary 0.567、hiphop recall 0.68）+ unknown
# + salsa（用戶實測線上預測率低於測試集 → 分佈多樣化）
TARGETS_PER_GENRE: dict[str, int] = {
    "ballet": 26,
    "breakdance": 26,
    "salsa": 32,        # +6（用戶回饋：線上預測率不如測試集）
    "hiphop": 32,       # +6
    "contemporary": 34, # +8
    "unknown": 26,      # +4
}

MIN_DUR = 60
# unknown 放寬上限（非舞蹈內容常為長視頻；窗口 cap 在 filter_split 控制）
MAX_DUR_BY_GENRE: dict[str, int] = {"unknown": 900}
DEFAULT_MAX_DUR = 300
SEARCH_RESULTS_PER_QUERY = 40

YDL_BASE = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    # video-only 優先（avc1 解碼快），480p 對 224×224 輸入足夠
    "format": "bestvideo[height<=480][vcodec^=avc1]/bestvideo[height<=480]/best[height<=480]",
    "download_archive": str(ARCHIVE),
    "outtmpl": str(RAW / "%(genre)s" / "%(id)s.%(ext)s"),
}


def count_files(genre: str) -> int:
    d = RAW / genre
    if not d.exists():
        return 0
    return len([f for f in d.iterdir() if f.suffix in (".mp4", ".webm", ".mkv")])


def log(line: dict) -> None:
    with open(RAW / "collect_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(json.dumps(line, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=None,
                    help="覆寫所有類別的目標支數（預設用 TARGETS_PER_GENRE）")
    args = ap.parse_args()

    RAW.mkdir(exist_ok=True)

    for genre, queries in GENRES.items():
        target = args.target or TARGETS_PER_GENRE.get(genre, 14)
        have = count_files(genre)
        log({"event": "genre_start", "genre": genre, "have": have, "target": target})
        if have >= target:
            continue

        (RAW / genre).mkdir(exist_ok=True)
        max_dur = MAX_DUR_BY_GENRE.get(genre, DEFAULT_MAX_DUR)
        ydl = YoutubeDL({**YDL_BASE, "outtmpl": str(RAW / genre / "%(id)s.%(ext)s")})

        for query in queries:
            if count_files(genre) >= target:
                break
            try:
                # 搜索僅取條目列表（flat + ignoreerrors），單支不可用不致整個搜索失敗
                search_opts = {**YDL_BASE, "extract_flat": "in_playlist", "ignoreerrors": True,
                               "download_archive": None, "outtmpl": "-"}
                with YoutubeDL(search_opts) as sydl:
                    info = sydl.extract_info(f"ytsearch{SEARCH_RESULTS_PER_QUERY}:{query}", download=False)
            except Exception as e:  # noqa: BLE001
                log({"event": "search_error", "genre": genre, "query": query, "error": str(e)[:200]})
                continue

            for entry in info.get("entries") or []:
                if count_files(genre) >= target:
                    break
                vid = entry.get("id")
                if not vid:
                    continue
                dur = entry.get("duration")
                if dur and not (MIN_DUR <= dur <= max_dur):
                    continue
                url = f"https://www.youtube.com/watch?v={vid}"
                try:
                    ydl.download([url])
                    time.sleep(1.0)  # 對 YouTube 客氣一點
                    log({"event": "downloaded", "genre": genre, "id": vid,
                         "title": (entry.get("title") or "")[:80], "duration": dur})
                except Exception as e:  # noqa: BLE001
                    log({"event": "download_error", "genre": genre, "id": vid, "error": str(e)[:200]})

        log({"event": "genre_done", "genre": genre, "have": count_files(genre)})

    summary = {g: count_files(g) for g in GENRES}
    log({"event": "all_done", "summary": summary})
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
