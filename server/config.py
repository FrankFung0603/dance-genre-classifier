"""集中設定：路徑、模型、串流、平滑參數。"""
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODEL_ONNX = ROOT / "models" / "dance_r2plus1d_fp16.onnx"

# 容器部署：ffmpeg 在系統 PATH（apt），以 env 指定；本機 dev 維持 tools/ 搜尋
FFMPEG_BIN = os.environ.get("FFMPEG_BIN")


def find_ffmpeg() -> str:
    if FFMPEG_BIN:
        return FFMPEG_BIN
    hits = glob.glob(str(ROOT / "tools" / "ffmpeg*" / "bin" / "ffmpeg.exe"))
    if not hits:
        raise RuntimeError("ffmpeg.exe not found under project/tools/")
    return hits[0]


CLASSES = ["ballet", "breakdance", "contemporary", "hiphop", "salsa", "unknown"]

# ---- 抽幀 / 窗口（與訓練一致：8fps、decode 256、center crop 224、16 幀窗口）----
STREAM_FPS = 8
STREAM_SIZE = 256
CROP_SIZE = 224
WIN_FRAMES = 16          # 2s @ 8fps
STRIDE_FRAMES = 8        # 1s stride = 50% 重疊 → 1Hz 預測
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.221459, 0.216989], dtype=np.float32)

# ---- Unknown / 時序平滑 ----
UNKNOWN_THRESHOLD = 0.45   # EMA 後 top-1 信心低於此 → Unknown
EMA_ALPHA = 0.4
MIN_DURATION_UPDATES = 2   # 新類需連續 N 次勝出才切換
HYSTERESIS_MARGIN = 0.10   # 新類 EMA 需超過當前類 + margin

# ---- 串流行為 ----
SEEK_DEBOUNCE_S = 0.5
# （原 SEEK_WARMUP_S=2.0 已移除：seek 重啟改為 offset=target，避免預測流天生落後 2s——
#   解碼慢的來源會一直追不回，詳見 worker._restart 註解與 docs/decisions.md）
LOOKAHEAD_S = 8.0          # 預測領先播放頭上限（pacing，避免整支影片提前算完）
PLAYHEAD_STALE_S = 3.0     # 超過此時間沒收到客戶端 playhead → 退化為 1× 牆鐘估計
TOPK = 3
