# Dance Genre Classifier — 專案執行規範（CLAUDE.md）

## 0. 你的角色

你是本專案的自主執行 AI。任務：從零建出一個可 demo 的全棧舞蹈分類系統。
你必須嚴格遵守下面的架構決策、階段順序、禁止事項與驗收標準。
遇到不確定：把假設記錄到 `docs/decisions.md`，繼續推進，不要停下來問。
例外：破壞性操作（刪除檔案、覆蓋未提交變更、git reset/force push）仍需先確認。

## 1. 專案一句話

使用者貼 YouTube URL → 瀏覽器播放影片 → 後端流式抽幀 → 時空模型每 1 秒預測舞種 → overlay 按 `media_time` 對齊顯示，並支援反饋與可解釋性。

## 2. 硬性架構決策（不可違反）

- 前端播放：YouTube IFrame API（`react-youtube`）。**不可**用後端 proxy 轉 WebRTC/HLS 給前端播放。
- 後端拉流：`yt-dlp`（保持最新版，**勿 pin 舊版**，YouTube 改版會弄壞舊版）解析 direct URL → `ffmpeg` pipe 抽幀，**不可寫入硬碟**，**不可下載完整影片**。
- 預測與播放是**兩條流**。後端預測帶 `media_time`，前端按 `media_time` 對齊顯示。這是核心同步機制。
- **inference worker 直接持有 FFmpeg 進程**。Redis 只傳控制消息（session 生命週期、seek 指令），**raw 幀不過 Redis**。
- 模型：X3D-M（16×224）或 R(2+1)D-18。**不可**用 Video Swin / SlowFast 重模型。
- 推理：ONNX Runtime GPU（FP16），ONNX Runtime CPU 為備胎；TensorRT INT8 為 stretch goal。
- 抽幀：8 FPS（`-vf fps=8`）→ `-f rawvideo -pix_fmt rgb24`、224×224。**不可** 30 FPS、**不可**逐幀 CNN、**不可**經 JPEG 編解碼。
- 可解釋性：**即時 CAM 歸因**（2026-10-06 起取代原 MediaPipe Pose overlay 方案，見 decisions #19）。模型尾端 GAP+FC ⇒ Grad-CAM 權重恰等於 FC 權重，免反傳播；worker 每窗算 14×14 歸因 grid 掛進 prediction payload，前端渲染熱圖（用戶可開關）。CAM 是空間歸因（看哪裡），不得聲稱為概念層解釋。
- Unknown / No Dance 必須存在：負樣本 + 置信度閾值 + EMA 時序平滑 + 最小持續時間（可加 hysteresis：新類置信度需超過當前類 + margin 才切換）。
- Seek 處理須 **debounce ~500ms**，重啟 ffmpeg 到 `seek_time - 2s`，清空 buffer。
- 必須有 **direct MP4 URL 輸入** 作為 fallback（走同一條 pipeline），避免 YouTube 反爬/限速讓 demo 開天窗。

## 3. 技術棧（鎖定，不要替換）

- 前端：Next.js (App Router) + TypeScript + TailwindCSS + `react-youtube` + **原生 WebSocket** + Canvas
  - 不用 socket.io：FastAPI 原生 WebSocket 直接對接瀏覽器原生 WS API，少一層協議適配
- 後端：Python 3.11 + FastAPI + `yt-dlp` + ffmpeg（**subprocess pipe**，raw 幀走 stdout）+ `onnxruntime-gpu` + `mediapipe` + `redis` + `sqlalchemy`
- 佇列：Redis（僅控制消息 pub/sub）
- DB：PostgreSQL（反饋）
- 部署：Docker + docker-compose
- 反向代理：nginx（含 WebSocket upgrade 配置）

## 4. 階段與任務（嚴格按順序）

### 階段 0：風險 Spike ✅（2026-09-30 完成，見 docs/phase-0.md）
- [x] `yt-dlp` 解析一個公開 YouTube URL，取得 direct media URL，記錄解析耗時（1.58s；採 video-only DASH avc1）
- [x] `ffmpeg` 從 direct URL pipe 出 rawvideo rgb24 224×224 幀流，不落盤；量測首幀延遲與吞吐（0.68s / 2.58× 即時）
- [x] 載入任一 ONNX 模型（R(2+1)D-18 K400），對 16 frames 推理，GPU 與 CPU 各量測（DML 90ms / CPU 385ms；匯出統一 opset 18）
- [x] 前端 YouTube IFrame 播放，`getCurrentTime()` 每 250ms 回報
- [x] 後端 MediaPipe Pose 對單幀輸出 landmarks，量測單幀耗時（mediapipe 1.0 Tasks API，full 19.9ms）——※此 spike 功能已於 2026-10-06 由即時 CAM 取代（decisions #19）
- **驗收**：五個 spike 全部可跑，baseline 延遲數字寫入 `docs/spike-results.md`

### 階段 1：資料與模型 ✅（2026-10-01 完成，見 docs/phase-1.md）
- [x] 自收集 84 支影片（6 類 × 14，yt-dlp，video-only DASH ≤480p；未用 Kinetics-700 全量——自收集 + K400 打分過濾已足夠且更乾淨）
- [x] 9,216 窗 → pseudo-label cleanup（dance_prob 閾值）→ 4,545 窗；memmap 快取（windows_u8.npy，零解碼訓練）
- [x] group split by 來源視頻：train 3,286 / val 628 / test 631
- [x] Fine-tune R(2+1)D-18（K400 init），16f@8fps；val macro-F1 **0.848**
- [x] ONNX opset 18 單檔 + FP16（62.8MB，parity 32/32）；延遲 85.9ms/clip（DML）
- [x] 測試集指標：macro F1 **0.619**（v1，弱類互混）→ **0.823**（v2 重訓：152 支影片、風格矛盾過濾、test 每類 4 支來源；hiphop 0.08→0.79、breakdance→0.92；contemporary 0.567 仍最弱，改進方向見 phase-1.md 重訓 v2 節）
- **驗收**：✅ 指標齊全（F1/per-class/confusion matrix）、推理 84ms < 1s、Unknown recall 1.00 有效

### 階段 2：串流與推理後端 ✅（2026-10-01 完成，見 docs/phase-2.md）
- [x] FastAPI 端點：`POST /session`（收 YouTube URL 或 direct MP4 URL）、`WS /ws/{sid}`（推預測與 CAM 歸因 grid）
- [x] inference worker：`yt-dlp` 解析 → **自己持有 ffmpeg pipe** → 滑動窗口 16 幀，每 1 秒推理一次（50% 重疊）
- [x] 預測 payload：`{ media_time, genre, confidence, topk }`；CAM grid 併入同 payload（media_time 對齊復用前端既有 buffer 機制）——原 landmarks 獨立消息已移除（decisions #19）
- [x] Redis：僅控制消息 —— ⚠️ 本機無 Redis runtime，dev 以 in-proc queue 承接（訊息契約同未來 Redis 匯流排，見 docs/decisions.md），Phase 5 compose 接入
- [x] 處理 pause / seek：debounce ~500ms，重啟 ffmpeg 到 `seek_time - 2s`，清空 buffer
- **驗收**：✅ E2E 15/15 × 3 來源（file / YouTube / direct MP4）：幀到預測 mean 121–127ms < 1s、1Hz、seek 精確恢復（30.0）、MP4 fallback 可用（卡通全程 unknown，fallback 有效）

### 階段 3：前端與時間對齊 ✅（2026-10-01 完成，見 docs/phase-3.md）
- [x] Next.js 頁面：URL 輸入 + YouTube IFrame + Dashboard（+ direct MP4 `<video>` fallback 播放器）
- [x] 每 250ms 回報 `currentTime` / paused / seek（狀態變化時立即送 + 心跳；buffering 狀態機兩輪迭代修復）
- [x] 預測 buffer：按 `media_time` 存放，播放到該時間才顯示
- [x] Dashboard：當前 genre、confidence bar、最近 10 秒 history（+ top3、inference 延遲）
- [x] Canvas overlay：繪製後端傳來的 CAM 歸因熱圖（原 Pose landmarks，decisions #19）；UI 提供開關
- [x] 反饋 UI：👍 / 👎 + 正確類別下拉（Phase 4 入庫）
- **驗收**：✅ 用戶瀏覽器實測通過（overlay 同步、seek 快、首預測 ~2s、影片結束乾淨收工）

### 階段 4：AI-native 與邊緣處理 ✅（2026-10-04 完成，見 docs/phase-4.md）
- [x] 反饋寫入 PostgreSQL：session_id、URL、media_time、預測、信心、正確類別（SQLAlchemy + docker postgres:16；DB 掛掉降級本地暫存、/feedback 回 503）
- [x] 可解釋 UI：即時 CAM 熱圖 + 文字說明（原 Pose overlay，decisions #19）—— CAM 是真歸因，非替身
- [x] 低分辨率、多舞者、鏡頭移動、突然切換的處理與量化結果（unknown/切換已量化；多舞者/鏡頭移動無標註集，誠實記入 limitations.md）
- [x] Unknown：閾值 + EMA + 最小持續時間（+ hysteresis）—— Phase 2 已實作（smoothing.py），本階段補參數量化文件
- **驗收**：✅ 反饋可查（POST ×2 → psql 2 rows → GET 讀回；非法 genre 400；healthz db:true），邊緣案例量化範圍誠實記錄

### 階段 5：部署 ✅（2026-10-06 完成，見 docs/phase-5.md）
- [x] Dockerfile：後端含 ffmpeg、yt-dlp（建置時裝最新）、onnxruntime-gpu、mediapipe
- [x] docker-compose：nginx / frontend / backend（inference in-proc，decisions #16）/ redis / postgres
- [x] GPU passthrough 已驗證（CUDA EP，容器內 RTX 4060）；**CPU fallback 實測可跑**（同 image 無 GPU → CPU EP 611ms/clip）
- [x] `docker-compose up` 一鍵起（healthcheck + depends_on.condition）
- [x] README：架構圖、setup、訓練 log、benchmark、已知限制（README.md 英文 + README.zh.md 中文）
- **驗收**：✅ compose E2E 20/20 PASS（`spikes/test_compose_e2e.py` → docs/phase5_e2e_result.json）：MP4/YouTube 預測流 1Hz、容器延遲 mean 102ms、seek 恢復、反饋入庫讀回、非舞蹈 ≥80% unknown

### 階段 6：交付 ⚠️ 部分完成（2026-10-06，用戶指示收斂範圍）
- [ ] 2 分鐘錄屏 —— 用戶決定不做
- [x] 2–3 頁技術報告：`docs/report.docx`（Word 交付版）／`docs/report.md`（source of truth；`tools/make_report_docx.py` 可重生成）——架構/資料策劃/訓練方法/性能分析，英文
- [ ] 10 分鐘 demo + Q&A 腳本 —— 用戶決定不做
- [x] 1 → 100 併發擴展設計章節：report §5 + README 雙語摘要
- **驗收**：Brief 硬性覆蓋項全數達成；錄屏/demo 腳本經用戶確認不在交付範圍

## 5. 禁止事項

- ❌ 後端 proxy YouTube 轉 WebRTC/HLS 給前端播放
- ❌ 下載完整影片再模擬播放
- ❌ 30 FPS 抽幀或逐幀 CNN
- ❌ 經 JPEG 編解碼（直接 rawvideo pipe）
- ❌ raw 幀過 Redis Stream（Redis 只傳控制消息）
- ❌ 按 clip 隨機切 train/val（必須 group split by 來源視頻）
- ❌ 即時 Grad-CAM
- ❌ 只靠 Kinetics 預訓練不 fine-tune
- ❌ 承諾 YouTube 首幀 500ms
- ❌ 前端 iframe 內跑 MediaPipe（跨域抓不到幀）
- ❌ 把與模型無關的輔助視覺化（如骨骼點）說成模型歸因——歸因只能來自 CAM 這類權重計算
- ❌ 寫死 YouTube 依賴，無 fallback
- ❌ pin 舊版 yt-dlp

## 6. 延遲定義（必須在 README 明確定義）

- 首幀播放：YouTube IFrame 決定，best effort，不承諾 500ms
- 首個預測：貼 URL 到第一個 overlay，目標 2–3s（2s 窗口填充是下限）
- 幀到 overlay：ffmpeg 出幀到前端更新，目標 < 1s
- 更新頻率：≥ 1 Hz

## 7. 每個階段的產出格式

每完成一個階段，產出：
1. `docs/phase-N.md`：做了什麼、決策、數據、已知問題
2. 可執行的程式碼或設定
3. 驗收測試指令與結果
4. 更新 `docs/limitations.md`

## 8. 遇到衝突時

優先順序：**可 demo > 可量測 > 可解釋 > 效能極致 > 功能完整**。
任何做不到的要求，寫進 `docs/limitations.md`，附數據與替代方案，不要假裝做到。

## 9. 完成定義（DoD）

- `docker-compose up` 一鍵起
- 貼 URL → 播放 → overlay 1Hz 更新 → CAM 熱圖（可開關）→ 反饋，全流程可跑（含 MP4 fallback 路徑）
- 5 類 + Unknown 有 F1 / confusion matrix（group split 的 held-out）
- README 含架構圖、setup、log、benchmark、限制
- 2 分鐘錄屏 + 2–3 頁報告 + 10 分鐘 demo 腳本
