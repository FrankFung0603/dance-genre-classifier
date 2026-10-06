"""過濾 windows（pseudo-label cleanup）+ group split by 來源視頻。

規則：
    - 舞蹈類 genre：保留 dance_prob >= --thr-dance 的窗口
    - unknown genre：保留 dance_prob <= --thr-unknown 的窗口
    - 風格矛盾過濾（僅 train）：K400 top1 明確指向「另一個」具體舞種且機率 >= 閾值
      → 視為弱標籤噪音，從 train 剔除。val/test 不剔除（保持測試誠實）。
      閾值可 per-genre（CONTRADICT_THR_BY_GENRE）：v3 起 contemporary 放寬到 0.55——
      K400 無 contemporary 對應類，lyrical/modern 被認成 ballet 屬預期行為，
      原 0.35 把 contemporary train 窗砍到僅 381（其他類 ~1,000），是 F1 0.567 的主因之一。
      K400 細粒度覆蓋：dancing ballet→ballet、breakdancing→breakdance、salsa dancing→salsa；
      hiphop/contemporary 無對應 K400 類，故僅能被「指向其他三類」的窗口觸發。
    - 每支影片最多 --cap-per-video 個窗口（均勻抽樣，避免單片獨大）
    - split 按 video_id 分組（同一影片的窗口絕不跨集合）：70/15/15，seed 固定
    - --pin-existing：增量輪用——舊影片凍結在原 split（v3 test 才能與 v2 對比、
      不會把 v2 test 影片練進 train），只對新影片做 70/15/15

輸出：
    data/splits.json: {"train": [key...], "val": [...], "test": [...]}，key = "video_id@t_start"
    data/splits_summary.json: 類別 x split 計數統計
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DANCE_GENRES = {"ballet", "breakdance", "salsa", "hiphop", "contemporary"}

# K400 細粒度類別 → 我們的 genre（僅精確對應；belly/tango 等不對應任何類，不觸發）
K400_TO_GENRE = {
    "dancing ballet": "ballet",
    "breakdancing": "breakdance",
    "salsa dancing": "salsa",
}

# per-genre 矛盾閾值（未列出的用 --thr-contradict）
CONTRADICT_THR_BY_GENRE: dict[str, float] = {"contemporary": 0.55}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr-dance", type=float, default=0.10)
    ap.add_argument("--thr-unknown", type=float, default=0.05)
    ap.add_argument("--thr-contradict", type=float, default=0.35)
    ap.add_argument("--cap-per-video", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pin-existing", action="store_true",
                    help="增量輪：splits.json 既有的 video_id 凍結原 split，只切新影片")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = list(csv.DictReader(open(HERE / "windows_manifest.csv", encoding="utf-8")))
    print(f"windows in manifest: {len(rows)}")

    # 0) 增量輪：讀舊 split，凍結舊影片的歸屬
    pinned: dict[str, str] = {}
    if args.pin_existing:
        old = json.loads((HERE / "splits.json").read_text(encoding="utf-8"))
        for s, keys in old.items():
            for k in keys:
                pinned[k.split("@")[0]] = s
        print(f"pin-existing: {len(pinned)} videos keep their old split")

    # 1) pseudo-label 過濾 + 標記風格矛盾窗口
    kept_by_genre_video = defaultdict(lambda: defaultdict(list))
    contradicted: set[str] = set()
    for r in rows:
        dp = float(r["dance_prob"])
        genre = r["genre"]
        if genre in DANCE_GENRES and dp >= args.thr_dance:
            key = f"{r['video_id']}@{r['t_start']}"
            m = K400_TO_GENRE.get(r["top1_label"])
            thr = CONTRADICT_THR_BY_GENRE.get(genre, args.thr_contradict)
            if m is not None and m != genre and float(r["top1_prob"]) >= thr:
                contradicted.add(key)
            kept_by_genre_video[genre][r["video_id"]].append(r)
        elif genre == "unknown" and dp <= args.thr_unknown:
            kept_by_genre_video[genre][r["video_id"]].append(r)
    print(f"contradicted windows (train-only filter): {len(contradicted)}"
          f" (contemporary thr={CONTRADICT_THR_BY_GENRE.get('contemporary', args.thr_contradict)},"
          f" others thr={args.thr_contradict})")

    # 2) 每影片 cap
    kept = defaultdict(list)
    for genre, by_video in kept_by_genre_video.items():
        for vid, wrs in by_video.items():
            if len(wrs) > args.cap_per_video:
                step = len(wrs) / args.cap_per_video
                wrs = [wrs[int(i * step)] for i in range(args.cap_per_video)]
            kept[genre].extend(wrs)

    print(f"\nkept windows after filtering (per genre):")
    total = 0
    for g in sorted(kept):
        n = len(kept[g])
        total += n
        print(f"  {g:14s} {n:5d}  ({len(kept_by_genre_video[g])} videos)")
    print(f"  {'TOTAL':14s} {total:5d}")
    if total == 0:
        sys.exit("ERROR: no windows kept — thresholds too strict?")

    # 3) group split by video_id（70/15/15，val/test 至少 1 支影片）
    #    --pin-existing：舊影片直接沿用 pinned split，只對新影片抽樣切分
    def split_ids(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        ids = sorted(set(ids))
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        return ids[n_val + n_test:], ids[:n_val], ids[n_val:n_val + n_test]

    video_split: dict[str, str] = {}
    for genre in sorted(kept):
        all_ids = list(kept_by_genre_video[genre].keys())
        old_ids = [v for v in all_ids if v in pinned] if args.pin_existing else []
        new_ids = [v for v in all_ids if v not in pinned]
        tr, va, te = split_ids(new_ids)
        for v in old_ids:
            video_split[v] = pinned[v]
        for v in tr:
            video_split[v] = "train"
        for v in va:
            video_split[v] = "val"
        for v in te:
            video_split[v] = "test"
        if old_ids:
            n_old = {"train": 0, "val": 0, "test": 0}
            for v in old_ids:
                n_old[pinned[v]] += 1
            print(f"  {genre:14s} pinned {len(old_ids)} (tr/va/te = "
                  f"{n_old['train']}/{n_old['val']}/{n_old['test']}), new {len(new_ids)}")

    splits = {"train": [], "val": [], "test": []}
    for genre in sorted(kept):
        for wrs in kept[genre]:
            key = f"{wrs['video_id']}@{wrs['t_start']}"
            splits[video_split[wrs["video_id"]]].append(key)

    # 4) 風格矛盾窗口只從 train 剔除（val/test 保留 = 誠實測試）
    n0 = len(splits["train"])
    splits["train"] = [k for k in splits["train"] if k not in contradicted]
    print(f"train windows: {n0} -> {len(splits['train'])} (removed {n0 - len(splits['train'])} contradicted)")

    (HERE / "splits.json").write_text(json.dumps(splits, indent=1))

    # 5) 統計：genre 從 video_id 反查；test 每類來源影片數（phase-1 僅 2 支的教訓）
    genre_of_video = {}
    for g, by_video in kept_by_genre_video.items():
        for vid in by_video:
            genre_of_video[vid] = g
    summary = {}
    for s in splits:
        cnt: dict[str, int] = defaultdict(int)
        vids: dict[str, set] = defaultdict(set)
        for k in splits[s]:
            g = genre_of_video[k.split("@")[0]]
            cnt[g] += 1
            vids[g].add(k.split("@")[0])
        summary[s] = {"total": len(splits[s]), "per_genre": dict(sorted(cnt.items()))}
        if s == "test":
            summary[s]["source_videos_per_genre"] = {g: len(v) for g, v in sorted(vids.items())}

    (HERE / "splits_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsplit summary:")
    print(json.dumps(summary, indent=2))
    print(f"\n-> {HERE / 'splits.json'}")


if __name__ == "__main__":
    main()
