"""把 splits 的所有窗口一次性解碼並快取為 uint8 memmap（N,16,256,256,3）。

訓練改讀 memmap（零 ffmpeg），解決：
    - 每 epoch 26 分鐘的解碼瓶頸（→ ~3 分鐘）
    - DataLoader worker 內 spawn ffmpeg 偶發的 native 崩潰

輸出：
    data/windows_u8.npy   uint8 memmap（physical row order = train全部 → val全部 → test全部，
                          與 train.py 的 key2row 公式一致）
    data/windows_meta.json {"train": [key...], "val": [...], "test": [...]}（對齊 memmap 列序）

--append：增量模式。舊 keys 依「新 meta 行位」fancy-copy（不解碼），只對新 keys 解碼；
          舊 keys 若已不在新 splits（過濾變嚴）自然丟棄。先寫暫存檔再替換。

Usage:
    python extract_windows.py [--append]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
sys.stdout.reconfigure(encoding="utf-8")
from train import decode_window  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_NPY = HERE / "windows_u8.npy"
OUT_META = HERE / "windows_meta.json"
N_FRAMES, SIZE = 16, 256
SPLIT_ORDER = ("train", "val", "test")


def key2row(meta: dict) -> dict[str, int]:
    k2r: dict[str, int] = {}
    base = 0
    for s in SPLIT_ORDER:
        for j, k in enumerate(meta[s]):
            k2r[k] = base + j
        base += len(meta[s])
    return k2r


def load_manifest() -> dict[str, dict]:
    import csv
    return {f"{r['video_id']}@{r['t_start']}": r
            for r in csv.DictReader(open(HERE / "windows_manifest.csv", encoding="utf-8"))}


def decode_into(arr, row: int, key: str, manifest: dict) -> None:
    r = manifest[key]
    try:
        arr[row] = decode_window(r["video_path"], float(r["t_start"]))
    except Exception as e:  # noqa: BLE001 — 壞窗口填 0（極少，訓練端可忽略）
        print(f"  WARN {key}: {e}", flush=True)
        arr[row] = 0


def run_append() -> None:
    old_meta = json.loads(OUT_META.read_text())
    old_k2r = key2row(old_meta)
    old_arr = np.load(OUT_NPY, mmap_mode="r")

    splits = json.loads((HERE / "splits.json").read_text())
    manifest = load_manifest()
    new_meta = {s: list(splits[s]) for s in SPLIT_ORDER}
    new_k2r = key2row(new_meta)

    missing = [k for s in SPLIT_ORDER for k in new_meta[s] if k not in manifest]
    if missing:
        sys.exit(f"ERROR: {len(missing)} split keys missing from manifest, e.g. {missing[:3]}")

    total = sum(len(new_meta[s]) for s in SPLIT_ORDER)
    n_new = [k for s in SPLIT_ORDER for k in new_meta[s] if k not in old_k2r]
    print(f"append: old {len(old_k2r)} windows | new total {total} | to decode {len(n_new)}")

    tmp = OUT_NPY.with_suffix(".new.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(total, N_FRAMES, SIZE, SIZE, 3))

    # 舊 rows 重定位（零解碼）：dst 依新順序、src 依舊行號
    moved = [k for s in SPLIT_ORDER for k in new_meta[s] if k in old_k2r]
    if moved:
        src = [old_k2r[k] for k in moved]
        dst = [new_k2r[k] for k in moved]
        arr[dst] = old_arr[src]
        arr.flush()
        print(f"  copied {len(moved)} old windows")
    # Windows：memmap handle 不關，稍後 unlink 舊檔會 WinError 32
    old_arr._mmap.close()
    del old_arr

    done = 0
    for s in SPLIT_ORDER:
        for k in new_meta[s]:
            if k in old_k2r:
                continue
            decode_into(arr, new_k2r[k], k, manifest)
            done += 1
            if done % 200 == 0:
                arr.flush()
                print(f"  decoded {done}/{len(n_new)}", flush=True)
    arr.flush()
    arr._mmap.close()
    del arr

    OUT_NPY.unlink()
    tmp.replace(OUT_NPY)
    OUT_META.write_text(json.dumps(new_meta, indent=1))
    # 記錄本輪新增的 keys（train.py --new-keys 用：增量續訓時新資料為主 + replay 舊資料）
    new_keys = {s: [k for k in new_meta[s] if k not in old_k2r] for s in SPLIT_ORDER}
    (HERE / "last_append_keys.json").write_text(json.dumps(new_keys, indent=1))
    print(f"\ndone (append): {OUT_NPY} ({OUT_NPY.stat().st_size / 1e9:.1f} GB, {total} windows)"
          f" | new keys: {[len(new_keys[s]) for s in SPLIT_ORDER]} -> last_append_keys.json")


def run_full() -> None:
    splits = json.loads((HERE / "splits.json").read_text())
    manifest = load_manifest()

    order: dict[str, list[str]] = {}
    rows_total = sum(len(splits[s]) for s in SPLIT_ORDER)
    idx = 0
    arr = np.lib.format.open_memmap(
        OUT_NPY, mode="w+", dtype=np.uint8,
        shape=(rows_total, N_FRAMES, SIZE, SIZE, 3),
    )
    for s in SPLIT_ORDER:
        order[s] = []
        for k in splits[s]:
            decode_into(arr, idx, k, manifest)
            order[s].append(k)
            idx += 1
            if idx % 200 == 0:
                arr.flush()
                print(f"  {idx}/{rows_total}", flush=True)
    arr.flush()

    OUT_META.write_text(json.dumps(order, indent=1))
    print(f"\ndone: {OUT_NPY} ({OUT_NPY.stat().st_size / 1e9:.1f} GB, {rows_total} windows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true",
                    help="增量：保留舊 windows（複製不解碼），只解碼 splits 新增的 keys")
    args = ap.parse_args()
    if args.append:
        if not (OUT_NPY.exists() and OUT_META.exists()):
            sys.exit("ERROR: --append needs existing windows_u8.npy + windows_meta.json")
        run_append()
    else:
        run_full()


if __name__ == "__main__":
    main()
