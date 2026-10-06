"""訓練管線冒煙測試：解碼真實窗口 → 前向/反向一次 → 確認 CUDA + 形狀 + 條件正確。

Usage:
    python training/smoke_test.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from train import CLASSES, build_model, decode_window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

vids = sorted((ROOT / "data" / "raw" / "breakdance").glob("*.mp4"))
assert vids, "no breakdance videos"
video = vids[0]
print("video:", video.name)

frames = decode_window(str(video), 30.0)
print("decoded:", frames.shape, frames.dtype)  # expect (16,256,256,3) uint8
assert frames.shape == (16, 256, 256, 3) and frames.dtype == np.uint8

# 標準化 + center crop（與 Dataset eval 路徑一致）
x = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
off = (256 - 224) // 2
x = x[..., off:off + 224, off:off + 224]
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.221459, 0.216989]).view(3, 1, 1, 1)
x = ((x - MEAN) / STD).unsqueeze(0).cuda()

model = build_model().cuda()
print("classes:", CLASSES)

with torch.autocast("cuda"):
    out = model(x)
    loss = out.float().square().mean()  # 假 loss，僅測反向
loss.backward()
print("forward:", tuple(out.shape), "| backward ok | cuda mem:",
      round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
assert out.shape == (1, 6)
print("SMOKE TEST PASS")
