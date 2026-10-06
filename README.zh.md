# 舞蹈類型即時分類系統（Dance Genre Classifier）

貼上 YouTube 連結（或 direct MP4）→ 瀏覽器播放影片 → 後端串流抽幀 → 時空模型**每秒**分類舞種，overlay 即時顯示，附**即時 CAM 歸因熱圖**（模型在看哪裡，可開關）與誠實的 **Unknown** fallback。👍/👎 反饋入庫 PostgreSQL。

五類：Ballet / Breakdance / Salsa / Hiphop / Contemporary + **Unknown**。

> English documentation: [README.md](README.md)
> 分析與數字（架構理由、資料集、訓練、benchmark、擴展設計）：[docs/report.md](docs/report.md) · 全部文檔：[docs/](docs/)

---

## 快速開始（Docker）

前提：Docker Desktop 已啟動（Windows 上是 WSL2 backend）。GPU 可有可無。

```bash
docker compose up --build
# → 開 http://localhost:8080
```

五個容器在 nginx :8080 後面——frontend（Next.js）、backend（FastAPI + 進程內推理）、redis（控制消息）、postgres（反饋）。模型檔以 volume 從 `./models` 掛載——換模型不用重建 image。

**GPU / CPU：** backend 服務的 `gpus: all` → CUDA EP。純 CPU 主機刪掉那行即可——同一個 image 自動落 CPU（provider 自動偵測：CUDA > DirectML > CPU）。

**檢查：**

```bash
curl http://localhost:8080/healthz
# {"ok":true,"ffmpeg":true,"model":true,"db":true,"redis":true,"active_sessions":0}
```

## 使用方式

1. **貼 YouTube 舞蹈連結**（或 direct MP4——YouTube 反爬弄不壞的確定性 fallback 路徑）
2. 預測以 1 Hz 串流到 overlay；第一條要 ~2–3 秒（2 秒窗口得先填充——物理下限，不是卡）
3. **CAM on/off** 開關歸因熱圖——紅黃 = 模型為了當前標籤正在看的區域
4. **自由拖進度條**——後端從新位置重串流（debounce 500ms）
5. **👍/👎** 把錯誤預測記進 PostgreSQL（DB 掛了也能用——UI 顯示「本地暫存」）；影片播完彈出全片投票結算卡

## 本機開發（不靠 Docker）

- **後端**：Python 3.12 venv（`requirements.txt`；Windows 用 `onnxruntime-directml`）、ffmpeg 以 `FFMPEG_BIN` 環境變數指定或解壓到 `tools/ffmpeg-<版本>/bin/`（repo 不附二進位——到 gyan.dev 下載 essentials build），然後
  `.venv\Scripts\python.exe -m uvicorn server.main:app --port 8000`
- **前端**：`cd frontend && npm install && npm run dev` → http://localhost:3000（預設打 :8000 的後端）
- **反饋 DB**：任何 reachable 的 `DATABASE_URL` Postgres——沒有也能跑；`/feedback` 回 503、UI 落到本地暫存

## 架構

```
┌────────────────────────────── 瀏覽器 ────────────────────────────────┐
│  YouTube IFrame 播放器 (react-youtube)      Next.js 16 + TS + Tailwind │
│  每 250ms 回報 playhead ─┐                  預測 buffer 按 media_time │
│  pause/seek 事件         │                  對齊顯示 → Canvas CAM 熱圖│
│                          ▼                  + 儀表板                   │
└─────────────────────────┼──────────────────────────▲──────────────────┘
                          │ HTTP + WebSocket         │ 預測 {media_time,
                          ▼                          │ genre, conf, CAM grid}
┌────────────────────── nginx :8080 ────────────────┴──────────────────┐
│            same-origin 反代 + WS upgrade (/ws/* → backend)           │
└──────────┬──────────────────────────────────────────────┬────────────┘
           ▼                                              ▼
┌─────────────────────────────┐              ┌────────────────────────┐
│ frontend (Next.js :3000)    │              │ backend (FastAPI :8000)│
└─────────────────────────────┘              │  POST /session         │
                                             │  WS   /ws/{sid}        │
                                             │  POST/GET /feedback    │
                                             │  InferenceWorker:      │
                                             │   yt-dlp → direct URL  │
                                             │   ffmpeg pipe（raw RGB │
                                             │   224²、8fps、不落盤）  │
                                             │   ONNX R(2+1)D-18 1Hz  │
                                             │  CAM 熱圖/每條預測      │
                                             └───────┬───────┬────────┘
                                                     │       │
                                        控制訊息      │       │ 反饋
                                        (in-proc,    ▼       ▼
                                         Redis-ready)  ┌──────────┐ ┌──────────┐
                                          ┌──────────┐ │postgres16│ │ redis:7  │
                                          │ models/  │ └──────────┘ └──────────┘
                                          │ (volume) │   👍/👎 rows    控制匯流排
                                          └──────────┘
```

為什麼這樣蓋（兩條流＋`media_time`、幀不過匯流排、免反傳播的 CAM、seek 語義）：[docs/report.md](docs/report.md) · 逐條決策：[docs/decisions.md](docs/decisions.md)

## 結果與文檔

所有分析性內容都在 docs——刻意不在 README 重複：

- **[docs/report.md](docs/report.md)** — 技術報告：F1 歷史（0.619 → 0.823 → 0.879，group-split held-out）、per-class P/R、混淆矩陣、延遲實測、資料策劃、訓練方法、1 → 100 擴展設計
- **[docs/limitations.md](docs/limitations.md)** — 已知限制，能量測的都誠實量化
- **`training/*.log`** — 原始訓練 log · **`docs/*.json`** — 指標與 E2E 結果

## 專案結構

```
frontend/          Next.js 應用（播放器、CAM 熱圖、儀表板、反饋 UI）
server/            FastAPI 後端 + InferenceWorker + Dockerfile
nginx/             反向代理設定（WS upgrade）
models/            ONNX 模型（雙輸出：logits + 特徵圖）+ CAM FC 權重（runtime 以 volume 掛載）
training/          訓練/匯出腳本 + 訓練 log
data/              資料集管線（manifest、切窗、切分）
docs/              技術報告、phase 報告、決策記錄、限制、指標
spikes/            風險 spike 腳本與 E2E 測試
```
