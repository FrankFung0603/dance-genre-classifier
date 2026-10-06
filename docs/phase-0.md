# Phase 0 — 風險 Spike（完成）

日期：2026-09-30

## 做了什麼

五個風險 Spike 全部執行完畢，**全部通過**。完整數據見 [`spike-results.md`](spike-results.md)。

| # | 驗證目標 | 狀態 |
|---|----------|------|
| 1 | yt-dlp 解析 direct URL | ✅ 1.58s |
| 2 | ffmpeg pipe rawvideo 224×224 不落盤 | ✅ 首幀 0.68s、吞吐 2.58× 即時 |
| 3 | ONNX 16 幀推理 CPU/GPU | ✅ 385ms / 90ms |
| 4 | IFrame `getCurrentTime()` 250ms 回報 | ✅（http://localhost:8123 人工驗證） |
| 5 | MediaPipe Pose 後端單幀 | ✅ full 19.9ms、33 landmarks |

## 關鍵決策

1. **推理管線採用 video-only DASH**（avc1 優先）——YouTube 已不給 progressive 混流格式，而推理本來就不需要音頻。
2. **GPU 推理走 onnxruntime-directml（DML EP）** 作為 Windows 本機開發路徑；Docker 內改 CUDA EP（數字需重測）。
3. **ONNX 匯出統一 opset 18**（opset 17 觸發新版 exporter 轉換失敗）。
4. **Pose 用 mediapipe 1.0 Tasks API**（legacy solutions 已移除），模型檔存 `models/`。

## 驗收指令與結果

見 `spike-results.md` 底部「重現指令」。五個 spike 均可重跑，數字會因網路/GPU 狀態小幅波動。

## 已知問題

- torch 2.14 exporter 權重寫外部檔（`*.onnx.data`），兩檔需同目錄——phase 1 匯出時改為單檔 inline。
- yt-dlp URL 時效/限速問題未在本 spike 壓測（屬階段 2 session 管理範疇）。

## 下一步（Phase 1）

資料與模型：Kinetics-700 舞蹈類別挑選、group split 資料集建構、X3D-M/R(2+1)D-18 fine-tune、ONNX 匯出、完整評估指標。
