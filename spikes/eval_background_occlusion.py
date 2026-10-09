"""背景依賴測試（occlusion test）：量化模型是否靠「背景/場景」判斷舞種。

方法：測試窗（memmap 快取，與訓練/評測完全同源）center crop 224 後，
對「中心 box 以外」（背景）做 28px 馬賽克化，與原窗分別推理，對比：

    bg_mosaic  : 背景馬賽克、主體保留   → F1/flip 掉多少 = 背景細節多重要
    sub_mosaic : 主體馬賽克、背景保留   → 掉多少 = 主體多重要（對照組）

原窗 F1 同時做 sanity check（應 ≈ 0.879 官方值）。

Usage:
    python spikes/eval_background_occlusion.py [--box 112] [--limit 0]
        [--provider DmlExecutionProvider]
輸出：docs/background_occlusion_metrics.json + stdout 報表
"""
import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
CLASSES = ["ballet", "breakdance", "contemporary", "hiphop", "salsa", "unknown"]
NUM_FRAMES, CROP = 16, 224
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.221459, 0.216989], dtype=np.float32)
MOSAIC_B = 28  # 馬賽克區塊大小（224/28 = 8×8 網格，摧毀細節、保留粗色彩）

sys.stdout.reconfigure(encoding="utf-8")


def preprocess(frames: np.ndarray) -> np.ndarray:
    """uint8 (T,224,224,3) → float32 (1,3,16,224,224)，與 export_eval 一致。"""
    x = frames.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(3, 0, 1, 2)[None]


def mosaic(frames: np.ndarray) -> np.ndarray:
    T, H, W, _ = frames.shape
    small = frames.reshape(T, H // MOSAIC_B, MOSAIC_B, W // MOSAIC_B, MOSAIC_B, 3) \
                  .mean(axis=(2, 4)).astype(np.uint8)
    return np.repeat(np.repeat(small, MOSAIC_B, axis=1), MOSAIC_B, axis=2)


def blur_frames(frames: np.ndarray, radius: int = 12) -> np.ndarray:
    """高斯模糊（比 mosaic 溫和：接近景深/運動模糊，OOD 偏移小）。"""
    from PIL import Image, ImageFilter
    out = np.empty_like(frames)
    for t in range(frames.shape[0]):
        out[t] = np.asarray(Image.fromarray(frames[t]).filter(
            ImageFilter.GaussianBlur(radius=radius)))
    return out


def occlude(frames: np.ndarray, keep_box: tuple | None, soft: bool) -> np.ndarray:
    """keep_box=(y0,y1,x0,x1) 區域保留原樣，其餘破壞（馬賽克或模糊）；None = 全部破壞。"""
    m = blur_frames(frames) if soft else mosaic(frames)
    if keep_box is None:
        return m
    y0, y1, x0, x1 = keep_box
    out = frames.copy()
    mask = np.ones((CROP, CROP), dtype=bool)
    mask[y0:y1, x0:x1] = False
    out[:, mask] = m[:, mask]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=int, default=112, help="保留區邊長（中心正方形），0 = 無保留區")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 窗（0 = 全部 test）")
    ap.add_argument("--provider", default="DmlExecutionProvider")
    ap.add_argument("--soft", action="store_true", help="用高斯模糊取代馬賽克（OOD 偏移較小）")
    ap.add_argument("--onnx", default="models/dance_r2plus1d_fp16_v3.onnx",
                    help="評測哪個 ONNX（預設 v3 部署權重單輸出檔）")
    ap.add_argument("--tag", default="", help="輸出檔名後綴（如 _v4），避免覆蓋基準結果")
    args = ap.parse_args()

    half = args.box // 2
    keep = None if args.box == 0 else (CROP // 2 - half, CROP // 2 + half,
                                       CROP // 2 - half, CROP // 2 + half)

    manifest = {f"{r['video_id']}@{r['t_start']}": r
                for r in csv.DictReader(open(DATA / "windows_manifest.csv", encoding="utf-8"))}
    test_keys = json.loads((DATA / "splits.json").read_text())["test"]
    if args.limit:
        test_keys = test_keys[: args.limit]
    meta = json.loads((DATA / "windows_meta.json").read_text())
    key2row, base = {}, 0
    for s in ("train", "val", "test"):
        for j, k in enumerate(meta[s]):
            key2row[k] = base + j
        base += len(meta[s])
    arr = np.load(DATA / "windows_u8.npy", mmap_mode="r")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ROOT / args.onnx),
                                sess_options=so, providers=[args.provider, "CPUExecutionProvider"])
    print(f"model={Path(args.onnx).name} | provider={sess.get_providers()[0]} | "
          f"box={args.box} | n={len(test_keys)}", flush=True)

    off = (256 - CROP) // 2
    ys, pred_orig, pred_bg, pred_sub, conf_orig, conf_bg = [], [], [], [], [], []
    t0 = time.time()
    for i, k in enumerate(test_keys):
        row = arr[key2row[k]]
        x = row[:, off:off + CROP, off:off + CROP, :]
        ys.append(CLASSES.index(manifest[k]["genre"]))
        for out, store in ((x, pred_orig), (occlude(x, keep, args.soft), pred_bg), (occlude(x, None, args.soft), pred_sub)):
            logits = sess.run(None, {"frames": preprocess(np.ascontiguousarray(out))})[0][0]
            e = np.exp(logits - logits.max())
            p = e / e.sum()
            store.append(int(p.argmax()))
            if out is x:
                conf_orig.append(float(p.max()))
            elif keep is not None:
                conf_bg.append(float(p.max()))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(test_keys)} | {el:.0f}s | eta {el / (i + 1) * (len(test_keys) - i - 1):.0f}s",
                  flush=True)

    ys = np.array(ys)
    res = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": Path(args.onnx).name,
        "provider": sess.get_providers()[0],
        "n_test": len(ys),
        "keep_box": args.box,
        "corruption": "gaussian_blur_r12" if args.soft else f"mosaic{MOSAIC_B}px",
        "mosaic_block": MOSAIC_B,
        "orig_macro_f1": round(float(f1_score(ys, pred_orig, average="macro")), 4),
        "bg_mosaic_macro_f1": round(float(f1_score(ys, pred_bg, average="macro")), 4),
        "sub_mosaic_macro_f1": round(float(f1_score(ys, pred_sub, average="macro")), 4),
        "orig_accuracy": round(float(accuracy_score(ys, pred_orig)), 4),
        "bg_mosaic_accuracy": round(float(accuracy_score(ys, pred_bg)), 4),
        "sub_mosaic_accuracy": round(float(accuracy_score(ys, pred_sub)), 4),
        "bg_flip_rate": round(float(np.mean(np.array(pred_bg) != np.array(pred_orig))), 4),
        "sub_flip_rate": round(float(np.mean(np.array(pred_sub) != np.array(pred_orig))), 4),
        "mean_conf_orig": round(float(np.mean(conf_orig)), 4),
        "mean_conf_bg_mosaic": round(float(np.mean(conf_bg)), 4),
        "bg_flip_rate_by_class": {
            c: round(float(np.mean(np.array(pred_bg)[ys == ci] != np.array(pred_orig)[ys == ci])), 4)
            for ci, c in enumerate(CLASSES)
        },
        "bg_macro_f1_report": classification_report(ys, pred_bg, target_names=CLASSES,
                                                    digits=3, zero_division=0, output_dict=True),
    }
    out_name = ("background_occlusion_metrics" + ("_blur" if args.soft else "") + args.tag + ".json")
    (ROOT / "docs" / out_name).write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\n=== 背景依賴測試（keep_box={args.box}px, n={len(ys)}, corruption={res['corruption']}）===")
    print(f"原窗        : macro F1 {res['orig_macro_f1']} | acc {res['orig_accuracy']}（sanity，應 ≈0.879）")
    print(f"背景馬賽克  : macro F1 {res['bg_mosaic_macro_f1']} | acc {res['bg_mosaic_accuracy']} "
          f"| flip {res['bg_flip_rate']:.1%} | conf {res['mean_conf_orig']}→{res['mean_conf_bg_mosaic']}")
    print(f"主體馬賽克  : macro F1 {res['sub_mosaic_macro_f1']} | acc {res['sub_mosaic_accuracy']} "
          f"| flip {res['sub_flip_rate']:.1%}")
    print("背景遮蔽 flip 率（依真實類別）:")
    for c, f in res["bg_flip_rate_by_class"].items():
        print(f"  {c:12s} {f:.1%}")
    print(f"-> docs/{out_name}")


if __name__ == "__main__":
    main()
