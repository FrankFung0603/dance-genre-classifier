# 舞蹈類型即時分類系統（Dance Genre Classifier）

貼上 YouTube 連結（或 direct MP4）→ 瀏覽器播放影片 → 後端串流抽幀 → 時空模型**每秒**分類舞種 → overlay 按播放時間對齊顯示。每條預測附帶**即時 CAM 歸因熱圖**（模型在看哪裡，用戶可開關），👍/👎 反饋閉環入庫 PostgreSQL。

**類別：** Ballet / Breakdance / Salsa / Hiphop / Contemporary + **Unknown** fallback

**主要結果（group split held-out 測試集）：** 三輪訓練 macro-F1 **0.619 → 0.823 → 0.879**（v3 與 v2 同測試集對比）。Unknown recall **1.000** —— 寧可說 unknown 也不硬猜。

> English documentation: [README.md](README.md) · 技術報告：[docs/report.docx](docs/report.docx)（Word 交付版）· [docs/report.md](docs/report.md)（原始碼）

---

## 架構

```
┌────────────────────────────── 瀏覽器 ────────────────────────────────┐
│  YouTube IFrame 播放器 (react-youtube)      Next.js 16 + TS + Tailwind │
│  每 250ms 回報 playhead ─┐                  預測 buffer 按 media_time │
│  pause/seek 事件         │                  對齊顯示 → Canvas overlay │
│                          ▼                  + 儀表板                   │
└──────────────────────────┼─────────────────────────▲──────────────────┘
                           │ HTTP + WebSocket        │ 預測 {media_time,
                           ▼                         │ genre, confidence, topk}
┌────────────────────── nginx :8080 ─────────────────┴─────────────────┐
│         same-origin 反向代理 + WS upgrade（/ws/* → backend）          │
└──────────┬───────────────────────────────────────────────┬───────────┘
           ▼                                               ▼
┌─────────────────────────────┐               ┌────────────────────────┐
│ frontend (Next.js :3000)    │               │ backend (FastAPI :8000)│
└─────────────────────────────┘               │  POST /session         │
                                              │  WS   /ws/{sid}        │
                                              │  POST/GET /feedback    │
                                              │  InferenceWorker：     │
                                              │   yt-dlp → direct URL  │
                                              │   ffmpeg pipe（raw RGB │
                                              │   224²、8fps、不落盤） │
                                              │   ONNX R(2+1)D-18 1Hz  │
                                              │  CAM 熱圖/每條預測     │
                                              └───────┬───────┬────────┘
                                                      │       │
                                         控制訊息     │       │ 反饋
                                        （in-proc，   ▼       ▼
                                         Redis-ready） ┌──────────┐ ┌─────────┐
                                          ┌──────────┐ │postgres16│ │ redis:7 │
                                          │ models/  │ └──────────┘ └─────────┘
                                          │ (volume) │  👍/👎 紀錄   控制匯流排
                                          └──────────┘
```

核心設計規則（已強制執行，不是口號）：

- **播放與預測是兩條流。** 預測帶 `media_time`，前端播到該時間點才顯示。這是核心同步機制——seek/pause/buffering 全部由此自然推出。
- **inference worker 直接持有 ffmpeg 進程。** raw 幀一律不過訊息匯流排；Redis 只傳控制消息（session 生命週期、seek 指令）——本 demo 為單進程，控制契約跑在 in-proc queue 上，訊息格式與未來 Redis pub/sub 完全一致。
- **不下載完整影片。** yt-dlp 解析 direct URL，ffmpeg pipe 出 raw RGB 幀（8 fps、解碼 256² 再 center crop 224，與訓練前處理完全一致）直接進模型窗口緩衝。
- **真歸因用 CAM，不用替身。** R(2+1)D-18 尾端是 GAP+FC，Grad-CAM 權重恰好退化為 FC 權重：`cam = ReLU(Σ w_class·feat)`——免反傳播。ONNX 匯出時把最後卷積特徵圖作為第二輸出；每條預測帶 14×14 歸因網格，前端即時渲染成熱圖（用戶可開關）。熱圖歸因「當前顯示標籤」——是空間歸因，不是概念層解釋。
- **Direct MP4 是正式輸入路徑**（同一條 pipeline）——YouTube 反爬或限速不會讓 demo 開天窗。

## 快速開始（Docker）

前置：Docker Desktop（Windows 需 WSL2 backend）。GPU 非必需。

```bash
docker compose up --build
# → 開啟 http://localhost:8080
```

就這樣。整套：`nginx :8080` → `frontend :3000` + `backend :8000` + `postgres 16` + `redis 7`。模型檔以 volume 掛載 `./models`（不烤進 image，換模型免重建）。

**GPU：** backend image 內建 `onnxruntime-gpu`（CUDA 12.4 + cuDNN 9 基底）。`docker-compose.yml` 預設有 `gpus: all` → ONNX Runtime 走 CUDA EP。**純 CPU 主機：刪掉 backend 服務的 `gpus: all` 那行即可，同一個 image 落到 CPU EP**（provider 自動偵測：CUDA > DirectML > CPU）。CPU 單 clip 延遲約 385 ms，1 Hz 預算內綽綽有餘。

**驗證：**

```bash
curl http://localhost:8080/healthz
# {"ok":true,"ffmpeg":true,"model":true,"db":true,"redis":true,"active_sessions":0}
```

然後貼一支 YouTube 舞蹈影片；或用 direct MP4 做確定性 demo，例如：
`https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4`（非舞蹈——應全程 **Unknown**）。

## 本機開發（不經 Docker）

- 後端：Python 3.12 venv（`requirements.txt`；Windows 用 `onnxruntime-directml`）、ffmpeg 以 `FFMPEG_BIN` 環境變數指定或解壓到 `tools/ffmpeg-<版本>/bin/`（Windows dev：到 gyan.dev 下載 essentials build——repo 不附二進位）、`uvicorn server.main:app --port 8000`。Postgres 用任何 reachable 的 `DATABASE_URL`。
- 前端：`cd frontend && npm install && npm run dev`（跨 origin 開發設 `NEXT_PUBLIC_API_BASE=http://localhost:8000`，預設值即是）。
- 訓練/資料管線：見 `training/` 與 `data/`（demo 不需要）。

## 模型

| | |
|---|---|
| 架構 | R(2+1)D-18，Kinetics-400 init fine-tune |
| 輸入 | 16 幀 @ 8 fps（2 秒窗）、224×224、50% 重疊 → 1 Hz 預測 |
| 部署檔 | `models/dance_r2plus1d_fp16.onnx`（opset 18、FP16、static batch-1、62.8 MB） |
| 平滑 | EMA α=0.4 → unknown 閾值 0.45 → hysteresis 0.10 → 最小持續 2 次更新 |
| 訓練資料 | 176 支自收集影片（弱標籤）→ pseudo-label 過濾 → **按來源影片** group split |

測試集 macro-F1 演進（group split held-out）：

| 輪次 | macro-F1 | 說明 |
|---|---|---|
| v1 | 0.619 | 84 支影片；hiphop/breakdance/contemporary 互混（hiphop F1 0.08） |
| v2 | 0.823 | 152 支 + 風格矛盾過濾；hiphop 0.79、breakdance 0.92 |
| v3 | **0.879** | +24 支（補 contemporary/salsa），自 v2 權重增量訓練 + 30% replay；v2 同測試集 = 0.852 |

完整 per-class P/R、confusion matrix、訓練 log：[`docs/phase-1.md`](docs/phase-1.md)、原始紀錄在 `training/*.log`。

## 延遲（實測）

| 階段 | 數值 |
|---|---|
| worker 內 幀→預測 | mean ~118–140 ms（三種來源一致，含 CAM） |
| 單 clip 推理 | 85.9 ms（DML EP，Windows）/ ~385 ms（CPU EP） |
| `POST /session` 後首個預測 | ~2–3 s —— 下限是 2 秒窗口填充 + URL 解析（~1.6 s）+ 首幀（~0.7 s） |
| WS 連上後首個預測（預熱後） | MP4 ~4.2 s（含播放器啟動）/ YouTube ~0.5 s |
| 更新率 | 預測 1 Hz，每條附 14×14 CAM 歸因網格 |

依 `CLAUDE.md` §6 的定義：**不**承諾首幀 500 ms（那是 YouTube 播放器決定）；「幀到 overlay < 1 s」由 125 ms worker 路徑 + 網路滿足。

## 擴展（1 → 100 併發）

單進程 demo 的形狀讓擴展是「拓撲問題」而非「概念問題」：session 狀態移進 Redis、in-proc 控制佇列換成 Redis pub/sub（訊息契約從第一天就按匯流排契約設計）、inference worker 變成容器池——每 worker 一個活躍 session、sticky 路由、WS 斷線即回收（session 生命週期本來就等於連線生命週期）。raw 幀照樣不過匯流排，每個 worker 自己持有 ffmpeg。按實測數字算容量：一張筆電級 GPU 在 85ms/clip 下撐 ~11 個 session，CPU 層當溢流；跨 session 批次推理估 2–3× 但**未實測**。真正的 100× 瓶頸在上游：yt-dlp 解析（IP 綁定——用 ~6 小時 TTL 的解析快取 + proxy pool 緩解）與 ffmpeg 解碼成本（約每 session 0.3 core，估算值）。細節：[docs/report.md §5](docs/report.md)。

## 目錄結構

```
frontend/          Next.js（播放器、overlay、儀表板、反饋 UI）
server/            FastAPI 後端 + InferenceWorker + Dockerfile
nginx/             反向代理設定（WS upgrade）
models/            ONNX 模型（雙輸出：logits + 特徵圖）+ CAM FC 權重（runtime 以 volume 掛載）
training/          訓練/匯出腳本 + 訓練 log
data/              資料集管線（manifest、切窗、切分）
docs/              階段報告、決策記錄、已知限制、指標
spikes/            風險 spike 腳本與 E2E 測試
```

## 已知限制

完整誠實清單在 [`docs/limitations.md`](docs/limitations.md)。重點：

- **contemporary 是最弱類**（v3 F1 0.756）——與 lyrical/modern、ballet 易混；hiphop 影片可能出現 hiphop/contemporary 交替顯示。
- 未見過影片在風格切換處預測可能震盪；hysteresis 用 1–2 秒切換延遲換穩定。
- yt-dlp direct URL 有 IP 綁定與約 6 小時時效；worker 對 session 內斷流可自動重解析恢復（最多 5 次後放棄），MP4 fallback 兜底。
- CAM 歸因空間解析度粗（14×14 網格，一格約 32 px），回答「看哪裡」而非概念層「為什麼」；unknown 類的熱圖最難解讀。
- 已量化邊緣案例：非舞蹈 fallback（unknown recall 1.000）、風格切換。**未**量化（無標註集）：多舞者準確率、運鏡穩定性。
