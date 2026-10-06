# Spike 結果 — 階段 0 風險驗證

日期：2026-09-30　環境：Windows 11 / Python 3.12.3（venv）/ RTX 4060 Laptop 8GB / ffmpeg 9.0.2（static）/ yt-dlp 2026.8.19 / onnxruntime-directml 1.24.4 / mediapipe 1.0.1

測試影片：PSY - GANGNAM STYLE (`9bZkp7q19f0`)，252s，24fps

## 結果總覽

| Spike | 內容 | 結果 | 關鍵數字 |
|-------|------|------|----------|
| 1 | yt-dlp 解析 direct URL | ✅ | 解析耗時 **1.58s**；480p avc1 video-only DASH 直鏈 |
| 2 | ffmpeg pipe rawvideo 抽幀 | ✅ | 首幀 **0.68s**；吞吐 **20.6 fps**（目標 8 fps 的 **2.58×**） |
| 3 | ONNX 16 幀推理（R(2+1)D-18 K400） | ✅ | CPU **385 ms** / GPU(DML) **90 ms**，1Hz 需求餘量 11× |
| 4 | 前端 IFrame 250ms 回報 | ✅（人工驗證） | `getCurrentTime()`/state 每 250ms 穩定回報 |
| 5 | MediaPipe Pose 單幀（後端） | ✅ | lite **16.9 ms** / full **19.9 ms** 穩態；33 landmarks，偵測到人體 |

## 結論：架構假說全部成立

- **兩條流 + media_time 對齊**：後端讀取速度是即時播放的 2.58 倍，預測可以持續超前播放進度，前端 buffer 對齊方案可行。
- **幀到預測 < 1s**：首幀 0.68s + GPU 推理 0.09s ≈ **0.77s**，在預算內（CPU 路徑 1.07s 略超，但部署走 GPU）。
- **1Hz 更新**：GPU 推理 90ms，1 秒一次綽綽有餘；CPU 也可行（385ms < 1s）。
- **Pose 負擔**：full 模型 4 FPS 採樣 = CPU 每秒約 80ms，可忽略。

## 過程中發現與修正

1. **YouTube 已無 progressive 混流格式**（format 18 不存在於測試影片）：全部是 DASH（視頻/音頻分離）或 HLS。
   → **決策**：推理管線抓 **video-only DASH（avc1 優先）**。我們不需要音頻（播放端 IFrame 自帶），file 更小、直鏈 https 無 manifest 延遲。
2. **Windows console cp950 編碼**會讓非 ASCII 標題 print 崩潰 → 所有腳本 `sys.stdout.reconfigure(encoding="utf-8")` 或 env `PYTHONIOENCODING=utf-8`。
3. **新版 torch（2.14）的 ONNX exporter**：需要 `onnxscript`；權重預設寫到外部檔 `*.onnx.data`（兩檔必須放一起）；opset 17 會觸發版本轉換失敗警告 → **之後統一 opset 18**。
4. **mediapipe 1.0.1 移除 legacy `mp.solutions`** → 使用 Tasks API（`PoseLandmarker`），模型檔需另行下載（`models/pose_landmarker_{lite,full}.task`）。

## 已知問題 / 限制（更新至 limitations.md）

- yt-dlp direct URL 有 IP 綁定與約 6 小時時效，且可能被限速——正式版需 MP4 fallback 與 session 內重新解析。
- DML EP 推理數字來自 Windows DirectML；Docker/Linux 部署改用 CUDA EP，數字需重測。
- Spike 2 的 `co located POCs unavailable` 為 h264 解碼器對流開頭的無害警告，幀資料完整（0 truncated）。
- 本機 GPU 8GB，多 session 併發時 DML 記憶體行為需在階段 2 壓測。

## 重現指令

```bash
# 1. 解析
.venv/Scripts/python.exe spikes/spike1_ytdlp.py
# 2. 抽幀
.venv/Scripts/python.exe spikes/spike2_ffmpeg.py --duration 12
# 3. 匯出 + 基準
.venv/Scripts/python.exe spikes/spike3_onnx_export.py
.venv/Scripts/python.exe spikes/spike3_onnx_bench.py
# 4. 前端（瀏覽器開 http://localhost:8123）
cd spikes/spike4_iframe && ../../.venv/Scripts/python.exe -m http.server 8123
# 5. Pose
.venv/Scripts/python.exe spikes/spike5_pose.py --runs 20
```
