# Phase 5 — 部署（docker-compose 一鍵起）

日期：2026-10-06

## 1. 交付物 ✅

| 檔案 | 內容 |
|---|---|
| `docker-compose.yml` | 5 服務：nginx / frontend / backend / redis / postgres |
| `server/Dockerfile` | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` 基底 + ffmpeg + Python 3.12 + onnxruntime-gpu + mediapipe + yt-dlp（建置時最新，不 pin） |
| `frontend/Dockerfile` | node:22-alpine multi-stage（npm ci → build → runtime） |
| `nginx/nginx.conf` | 統一入口 8080→80；`/session|/feedback|/healthz|/ws` 反代 backend（WS upgrade、read_timeout 1h），其餘走 frontend |
| `models/`（volume） | 模型不進 image，bind-mount `./models:/app/models:ro`，換模型免重建 |
| README.md / README.zh.md | 架構圖、setup、訓練 log、benchmark、已知限制（雙語） |

## 2. 拓撲與決策

- **inference 不拆獨立容器**（decisions #16）：worker in-proc 於 backend；redis 服務保留（healthz 回報連通；拆分時 pub/sub 契約現成）。
- **GPU/CPU 同一 image**（decisions #17）：ONNX provider 自動偵測 CUDA > DML > CPU；無 GPU 主機刪 compose 的 `gpus: all` 一行即為 CPU 模式。
- **same-origin 前端**（decisions #18）：`NEXT_PUBLIC_API_BASE=""`，WS 用 `location.host` 組絕對 URL。
- **容器 Python 3.12**（decisions #16b）：規格文件寫 3.11，但本機 dev venv 實為 3.12.3 且 numpy 2.5.x 鎖 ≥3.12——更正規格而非偏離。
- 競態處理：`docker-compose.yml` 用 healthcheck + `depends_on.condition`（backend 等 postgres/redis healthy；nginx 等兩者 healthy），`start_period` 60s 容納 lifespan prewarm。

## 3. E2E 驗收（2026-10-06 實測，`spikes/test_compose_e2e.py`）

對 nginx :8080（對外唯一入口）打完整 demo 路徑：**20/20 PASS**（`docs/phase5_e2e_result.json`）。

| 驗收項 | 結果 |
|---|---|
| healthz（nginx → backend → db/redis 全鏈路） | `ok/ffmpeg/model/db/redis` 全 true |
| direct MP4 預測流 | 20 筆/14s，**0.70s/pred**，media_time 嚴格遞增 |
| 幀→預測延遲（容器內，CUDA EP） | **mean 102ms / max 320ms**（< 1s） |
| 非舞蹈 fallback（Big Buck Bunny） | ≥80% unknown（少數窗漏成 contemporary，見 §5） |
| seek 30s → 恢復 | state(seek) 後新預測 t=31.0（容差 ±2s 內） |
| 反饋閉環 | POST ×2 入庫、GET 讀回 count=2、非法 genre 400 |
| YouTube（yt-dlp 路徑） | salsa 影片預熱後穩定鎖定 **salsa**；pose 58 筆/14s ≈ 4Hz |
| 一鍵起 | `docker compose up -d` 全服務 healthy |

GPU/CPU 雙模式驗證：

- **GPU**：stack 預設 `gpus: all` → 容器內 `nvidia-smi` 見 RTX 4060 Laptop，CUDA EP 生效，prewarm 1.3s（DML 為 8–15s）。
- **CPU fallback**：同 image 不掛 GPU → ORT 自動 "Falling back to CPUExecutionProvider"，`get_providers()` = [CPU]，單 clip **611ms**（< 1s @1Hz 預算）。worker 另加 try/except 純 CPU 兜底（雙保險）。

## 4. 重現指令

```bash
docker compose up -d --build          # 一鍵起
curl http://localhost:8080/healthz
.venv/Scripts/python.exe spikes/test_compose_e2e.py --yt "<舞蹈影片 URL>"
```

## 5. 已知問題 / 後續

- BBB（卡通）在容器 CUDA EP 下偶有 contemporary 漏判（約 1–4 窗/20），主機 DML 時代為全片 unknown——兩 EP 數值微小差異在 unknown 閾值邊緣被放大；≥80% unknown 的驗收線仍過，但「卡通必 unknown」不再是絕對。
- `POST /session` 成功但 WS 從未連上時，session dict 條目殘留到進程重啟（無 worker、零算力成本）；單客戶端 demo 可接受。
- 本機 dev 用的手動容器 `dance-postgres`（host 5432）保留給 dev；compose 的 postgres 不發布 host 埠，兩者不衝突。
- TensorRT EP 在 image 內可用（ORT 報告 TensorrtExecutionProvider），未啟用——INT8 校準為 stretch goal，未排程。
- 瀏覽器端 same-origin WS（`window.location.host` 路徑）由腳本驗證了 nginx 側 WS upgrade；真機瀏覽器操作待 demo 前人工點檢一次。
