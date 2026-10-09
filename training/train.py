"""Fine-tune R(2+1)D-18（Kinetics-400 預訓練）→ 6 類舞種分類。

- 輸入：windows manifest + splits.json（group split 已完成），16 frames @ 8fps，256² 解碼 → 224 crop
- 訓練：CUDA AMP、WeightedRandomSampler（類別平衡）、AdamW 分組 lr（head 5e-4 / backbone 5e-5）、
        cosine + warmup、label smoothing 0.1、早停（val macro F1, patience 4）
- 輸出：models/dance_finetuned.pt（state_dict + classes + val_f1）、training/log.csv

Usage:
    python train.py [--epochs 15] [--batch 16] [--seed 42] [--device cuda|cpu]
"""
import argparse
import csv
import glob
import json
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
CLASSES = ["ballet", "breakdance", "contemporary", "hiphop", "salsa", "unknown"]
NUM_FRAMES, DECODE_SIZE, CROP_SIZE, FPS = 16, 256, 224, 8
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.221459, 0.216989]).view(3, 1, 1, 1)


def find_ffmpeg() -> str:
    return glob.glob(str(ROOT / "tools" / "ffmpeg*" / "bin" / "ffmpeg.exe"))[0]


def decode_window(video_path: str, t_start: float) -> np.ndarray:
    """ffmpeg seek 後解碼 16 幀 → uint8 (16,256,256,3)。"""
    cmd = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{t_start:.3f}", "-i", video_path,
        "-frames:v", str(NUM_FRAMES),
        "-vf", f"fps={FPS},scale={DECODE_SIZE}:{DECODE_SIZE}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    need = NUM_FRAMES * DECODE_SIZE * DECODE_SIZE * 3
    if len(out) < need:  # 尾端不足 16 幀：重頭讀（clip 選取已避開尾端，此為保險）
        raise ValueError(f"short decode {len(out)}/{need} @ {video_path}@{t_start}")
    return np.frombuffer(out[:need], dtype=np.uint8).reshape(
        NUM_FRAMES, DECODE_SIZE, DECODE_SIZE, 3).copy()  # copy: writable, 脫離 buffer


class DanceWindowDataset(Dataset):
    """讀 memmap 快取（首選，worker 內 lazy-open）或即時 ffmpeg 解碼（fallback）。"""

    def __init__(self, rows: list[dict], train: bool, cache: dict | None = None):
        self.rows = rows
        self.train = train
        self._cache = cache  # {"npy_path": str, "key2row": dict}（只傳路徑，不傳 mmap 物件）
        self._arr = None

    def _open(self):
        if self._arr is None:
            self._arr = np.load(self._cache["npy_path"], mmap_mode="r")
        return self._arr

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.rows[i]
        if self._cache is not None:
            row = self._cache["key2row"][f"{r['video_id']}@{r['t_start']}"]
            frames = np.ascontiguousarray(self._open()[row])
        else:
            frames = decode_window(r["video_path"], float(r["t_start"]))
        x = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()  # C,T,H,W uint8
        if self.train:
            top = random.randint(0, DECODE_SIZE - CROP_SIZE)
            left = random.randint(0, DECODE_SIZE - CROP_SIZE)
            x = x[..., top:top + CROP_SIZE, left:left + CROP_SIZE]
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])  # (C,T,H,W) 的 W 維
        else:
            off = (DECODE_SIZE - CROP_SIZE) // 2
            x = x[..., off:off + CROP_SIZE, off:off + CROP_SIZE]
        return x, CLASSES.index(r["genre"])


def build_model() -> nn.Module:
    model = torchvision.models.video.r2plus1d_18(
        weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, np.ndarray]:
    model.eval()
    ys, ps, losses = [], [], []
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    for x, y in loader:
        x = x.to(device, non_blocking=True).float().div_(255.0)
        x = (x - MEAN.to(device)) / STD.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(x)
            losses.append(ce(out, y).item())
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return (float(f1_score(np.concatenate(ys), np.concatenate(ps), average="macro")),
            float(np.mean(losses)), np.concatenate(ps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation（等效 batch = batch × accum）")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu = 無 GPU 訓練（FP32、無 AMP、無 TF32）；與 cuda run 同配方對比訓練成本用")
    ap.add_argument("--resume", action="store_true", help="從 models/ckpt_latest.pt 續訓（斷點恢復，連 optimizer/排程）")
    ap.add_argument("--init-from", default=None,
                    help="續訓：只載入指定 checkpoint 的權重（如 models/dance_finetuned.pt），重置 optimizer/排程；"
                         "加新資料的輪次預設用這個，不用 K400 重訓")
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-backbone", type=float, default=5e-5,
                    help="--init-from 時建議調低（如 --lr-head 1e-4 --lr-backbone 1e-5）避免沖掉已學表徵")
    ap.add_argument("--new-keys", default=None,
                    help="data/last_append_keys.json（extract_windows --append 產出）：增量輪以新窗口為訓練主體")
    ap.add_argument("--replay-frac", type=float, default=0.3,
                    help="增量輪混入的舊 train 窗口比例（防災難性遺忘；0 = 純新資料，不建議）")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True   # 固定輸入形狀，一次性 autotune 後提速（無效也不影響正確性）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    manifest = {f"{r['video_id']}@{r['t_start']}": r
                for r in csv.DictReader(open(DATA / "windows_manifest.csv", encoding="utf-8"))}
    splits = json.loads((DATA / "splits.json").read_text())
    rows = {s: [manifest[k] for k in splits[s]] for s in ("train", "val", "test")}

    # 增量輪：train = 本輪新窗口 + 少量舊窗口 replay（防災難性遺忘）；val/test 恆用全量
    if args.new_keys:
        nk = json.loads((DATA / args.new_keys).read_text()) if not Path(args.new_keys).is_absolute() \
            else json.loads(Path(args.new_keys).read_text())
        new_set = set(nk["train"])
        key_of = lambda r: f"{r['video_id']}@{r['t_start']}"
        new_rows = [r for r in rows["train"] if key_of(r) in new_set]
        old_rows = [r for r in rows["train"] if key_of(r) not in new_set]
        rng = random.Random(args.seed)
        replay = rng.sample(old_rows, int(len(old_rows) * args.replay_frac))
        rows["train"] = new_rows + replay
        print(f"incremental train mix: {len(new_rows)} new + {len(replay)} replay "
              f"({args.replay_frac:.0%} of {len(old_rows)} old) | val/test full")

    print({s: len(r) for s, r in rows.items()})

    # memmap 快取（extract_windows.py 產出）；只傳路徑給 Dataset，worker 內各自 lazy-open
    cache = None
    meta_path = DATA / "windows_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        key2row = {}
        for s in ("train", "val", "test"):
            base = sum(len(meta[p]) for p in ("train", "val", "test")[: ("train", "val", "test").index(s)])
            for j, k in enumerate(meta[s]):
                key2row[k] = base + j
        cache = {"npy_path": str(DATA / "windows_u8.npy"), "key2row": key2row}
        print("cache ready (memmap lazy-open per worker, 零解碼訓練)")
    else:
        print("WARN: no window cache — falling back to live ffmpeg decode (slow)")

    device = args.device
    model = build_model().to(device)

    # 類別平衡取樣
    counts = np.bincount([CLASSES.index(r["genre"]) for r in rows["train"]], minlength=len(CLASSES))
    weights = (1.0 / counts)[[CLASSES.index(r["genre"]) for r in rows["train"]]]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(rows["train"]), replacement=True)

    def make_loader(batch: int, shuffle: bool, ds_train: bool, sampler_=None):
        return DataLoader(
            DanceWindowDataset(rows["train" if ds_train else "val"], train=ds_train, cache=cache),
            batch_size=batch, num_workers=args.workers, pin_memory=(device == "cuda"),
            sampler=sampler_, shuffle=shuffle and sampler_ is None,
            persistent_workers=True, prefetch_factor=4, timeout=60,
        )

    try:
        loader_tr = make_loader(args.batch, False, True, sampler)
    except torch.cuda.OutOfMemoryError:  # noqa: 建第一批時 OOM → 降 batch
        args.batch = 8
        print("OOM -> batch 8")
        loader_tr = make_loader(args.batch, False, True, sampler)
    loader_va = make_loader(args.batch, False, False)

    head_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)

    # ---------- 續訓：只載權重（排程/optimizer 全新起點） ----------
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck["model"])
        print(f"[init-from] {args.init_from} (epoch {ck.get('epoch', '?')}, "
              f"val {ck.get('val_macro_f1', '?')}) | lr head {args.lr_head} / backbone {args.lr_backbone}",
              flush=True)

    # ---------- sanity check：單 batch 前向，確認模型與資料形狀/分佈正常 ----------
    san_x, san_y = next(iter(DataLoader(DanceWindowDataset(rows["train"], train=False, cache=cache),
                                        batch_size=args.batch, num_workers=0)))
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
        san_out = model(san_x.to(device).float().div_(255.0))
    print(f"[sanity] batch {tuple(san_x.shape)} | logits mean {san_out.float().mean():.3f} "
          f"std {san_out.float().std():.3f} | pred classes {san_out.float().argmax(1).tolist()} | true {san_y.tolist()}",
          flush=True)
    model.train()

    warmup = 1
    accum = args.accum
    steps_per_epoch = max(1, len(loader_tr) // accum)
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup * steps_per_epoch:
            return (step + 1) / (warmup * steps_per_epoch)
        p = (step - warmup * steps_per_epoch) / max(1, total_steps - warmup * steps_per_epoch)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---------- resume ----------
    ckpt_path = ROOT / "models" / "ckpt_latest.pt"
    best_path = ROOT / "models" / "dance_finetuned.pt"
    start_epoch, step, best_f1, bad = 1, 0, -1.0, 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        step = ck["step"]
        best_f1 = ck["best_f1"]
        bad = ck.get("bad", 0)
        print(f"[resume] from epoch {start_epoch}, step {step}, best_f1 {best_f1:.3f}", flush=True)

    log_mode = "a" if (args.resume and ckpt_path.exists()) else "w"
    log_f = open(HERE / "log.csv", log_mode, newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    if log_mode == "w":
        log_w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "lr_head", "seconds"])
        log_f.flush()

    def save_latest(epoch: int) -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "step": step, "best_f1": best_f1, "bad": bad,
            "classes": CLASSES,
        }, tmp)
        tmp.replace(ckpt_path)  # 原子替換

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        win_seen = 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loader_tr):
            x = x.to(device, non_blocking=True).float().div_(255.0)
            # 亮度/色度抖動（per-sample per-channel 通道縮放）＋ 正規化，皆在 GPU
            gain = 1 + (torch.rand(x.size(0), 3, 1, 1, 1, device=device) - 0.5) * 0.3
            x = (x * gain).clamp_(0, 1)
            x = (x - MEAN.to(device)) / STD.to(device)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda"):
                loss = ce(model(x), y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 50 == 0:
                    mem = f"gpu {torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if device == "cuda" else "cpu"
                    rate = win_seen / max(1e-9, time.time() - t0)
                    print(f"  step {step:4d} | loss(avg50) {np.mean(losses[-50 * accum:]):.3f} "
                          f"| lr {opt.param_groups[1]['lr']:.2e} | {rate:.1f} win/s | {mem}",
                          flush=True)
                if step % 200 == 0:
                    save_latest(epoch)
            losses.append(loss.item() * accum)
            win_seen += x.size(0)

        f1, vloss, _ = evaluate(model, loader_va, device)
        lr_head = opt.param_groups[1]["lr"]
        log_w.writerow([epoch, round(float(np.mean(losses)), 4), round(vloss, 4),
                        round(f1, 4), lr_head, round(time.time() - t0, 1)])
        log_f.flush()
        print(f"epoch {epoch:2d} | train {np.mean(losses):.3f} | val {vloss:.3f} "
              f"| val macro-F1 {f1:.3f} | lr {lr_head:.2e} | {time.time()-t0:.0f}s", flush=True)

        if f1 > best_f1:
            best_f1, bad = f1, 0
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "epoch": epoch, "val_macro_f1": f1}, best_path)
            print(f"  -> saved best (F1 {f1:.3f})", flush=True)
        else:
            bad += 1
            if bad >= 4:
                print(f"early stop @ epoch {epoch} (best {best_f1:.3f})", flush=True)
                save_latest(epoch)
                break
        save_latest(epoch)

    log_f.close()
    print(f"\nbest val macro-F1: {best_f1:.4f} -> {best_path.name}", flush=True)


if __name__ == "__main__":
    main()
