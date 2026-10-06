# 決策記錄（假設與偏差，按時間追加）

## 2026-10-06（晚）即時 CAM 歸因取代 Pose overlay

19. **移除 MediaPipe Pose overlay（含依賴），可解釋性改由即時 CAM 歸因正面滿足**。
    Why: Pose 與模型判斷無關（模型吃 RGB 像素，不吃關節座標），把它當「Show why」的答案牽強——用戶明確要求「不要用不像歸因的東西充數」。R(2+1)D-18 尾端 GAP+FC ⇒ Grad-CAM 權重恰等於 FC 權重（cam = ReLU(Σ w_class·feat)），免反傳播。實作：ONNX 匯出雙輸出（logits + 最後卷積特徵圖 512×2×14×14，`training/export_cam.py`，與舊單輸出版 parity 16/16、max prob diff 0；fp16 轉換以 node block-list 迭代修復 `node_mean` 型別不一致）+ `models/fc_weights.npy`；worker 每窗以 numpy einsum 算 14×14 歸一化 grid 掛進 prediction payload（歸因「當前顯示標籤」）；前端 Pose canvas 改 CAM 熱圖（jet 色階、依強度透明），UI 加用戶開關。E2E 斷言 pose → cam_grid（本地 15/15、compose 20/20，cam_streaming 20/20）。requirements/Dockerfile 移除 mediapipe 與 GL 套件群。代價與邊界：CAM 是空間歸因（看哪裡），非概念層解釋；歸因的是平滑後顯示標籤（EMA 可能落後 raw argmax 一窗）。`models/pose_landmarker_*.task` 檔案保留於磁碟（用戶指示資料檔保留到最終交付）。

## 2026-10-06 Phase 5（部署）

16. **inference 不拆獨立容器：worker 維持 in-proc 於 backend 容器**（compose 服務 = nginx / frontend / backend / redis / postgres，架構文件原列的 inference 服務併入 backend）。
    Why: 單客戶端 demo 模型（session = WS 連線、第二連線 4409），拆分只增加跨進程橋接（預測經 Redis 中轉）的延遲與故障面，無 demo 價值；「worker 直接持有 ffmpeg、raw 幀不過 Redis」的硬規則在 in-proc 模式下天然成立。控制訊息 in-proc queue 的訊息契約即 Redis pub/sub 契約（decisions #1），1→100 併發擴展章節說明拆分路徑。compose 中 redis 服務保留：healthz 回報連通狀態，拆分時直接啟用。
16b. **容器 Python 用 3.12（規格文件寫 3.11）**。Why: 本機 dev venv 實際是 3.12.3，且 numpy 2.5.3 鎖 `Requires-Python >=3.12`——3.11 連依賴都裝不起來。容器對齊 dev 實際 runtime，消除「規格寫一套、跑一套」的落差。屬對技術棧文件的更正而非偏離。
17. **容器基底 `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`，image 內建 onnxruntime-gpu；worker 的 ONNX provider 改自動偵測（CUDA > DML > CPU）**。Why: 同一 image 覆蓋 GPU（`gpus: all` → CUDA EP）與 CPU fallback（無 GPU 主機刪 compose 一行即可），不再維護兩份設定；Windows 主機 dev 不受影響（DML EP 自動選中）。代價：image 較大（CUDA/cuDNN runtime ~3GB），換來零設定差異。
18. **nginx 統一入口（8080:80），前端 `NEXT_PUBLIC_API_BASE=""` same-origin**；WebSocket 以 `location.host` 組絕對 URL（瀏覽器 WS constructor 相對 URL 支援度不一，不賭）。後端路由 `/session|/feedback|/healthz|/ws` 反代含 WS upgrade header、`proxy_read_timeout 1h`（session = 連線生命週期）。

## 2026-10-04 穩定性與 seek 語義修正

12. **seek/resume 重啟改為 `offset = target`（原 CLAUDE.md 規格「重啟到 seek_time − 2s + 丟棄 warmup 窗」廢止）**。
    Why: bracket 仿真實測（spikes/test_seek_bracket.py）證明 −2s 設計讓預測流天生落後播放頭 2s 起步；解碼吞吐 <2× 的來源（HTTP range 重連慢啟動、googlevideo 節流）會一直追不回——用戶實測「seek 後 current genre 永遠空白、last 10s 卻持續更新」。offset=target 後第一窗即真實內容，落後僅剩啟動延遲 ~1s。代價：seek 點的預測晚 ~1s 出現。屬對架構文件的有意識偏離，意圖（seek 後預測快速恢復且不含舊內容）不變。
13. **前端 fallback 容差 0.7s → 2.5s，>2.5s 改顯示「預測追趕中…」而非過時預測**。Why: 追趕期顯示 5–9s 前的舞種是「別的時間點的答案」，對 demo 傷害大於空白；2.5s 內顯示最新已知預測 + 追趕 chip 是誠實與可用的平衡。
14. **worker 加三層防護**：run() top-level try/except → fatal（原：任何例外 = thread 靜默死亡 = 前端無限轉圈）；ffmpeg `-rw_timeout 10s`（原：來源卡住 → stdout.read 無限阻塞）；seek 後雙失敗明確 fatal（原：誤報 ended，前端像沒反應）。
15. **模型/pose 啟動預熱（lifespan prewarm）**。Why: ORT DML session 建立與 mediapipe import 合計 8–15s，原為 lazy 載入、全落在用戶首個 session；移到後端啟動後，實測 WS 後首預測 MP4 4.2s / YouTube 0.5s。

## 2026-10-01 Phase 3（前端狀態機）

7. **播放狀態機以 YouTube player state 為準：pause/resume 只認 1↔2，buffering(3) 一律不送控制訊息**。
   Why: 拖進度條 = seek → buffering → 播放。把 buffering 當 pause 會造成一次 seek 連開三次 ffmpeg（每次都有 YouTube 首幀延遲）；把 seek 偵測限定在 playing 之間又會漏掉整個 seek（狀態經過 3）。時間跳躍在 1/3 皆須觸發，用 `playingSeen` 旗標防初始載入誤判。
8. **播放器與 session 建立並行**（URL 提交即播影片，yt-dlp 解析在背景）。Why: 體感首預測延遲 3–4s 中的 1.6s 是解析擋在播放器前面；並行後影片立即播放，第一個 overlay ~2s 內出現。
9. **初始啟動/斷流恢復不做 warmup（warmup_windows=0），僅 seek 保留 -2s 熱身**。Why: seek 的第一窗是 t-2s 起的舊內容（需要跳過）；初始啟動與恢復的 offset 就是目標本身，第一窗即真實內容，跳過 warmup 省 ~1s 首預測。
10. **影片自然結束 ≠ 串流錯誤**：worker 以「recover 重啟後推進 <1s 又 EOF」判定媒體結束，發 `state(reason=ended)` 而非 fatal；前端偵測 player ended 主動送 pause。Why: 否則每支影片播完都彈紅色錯誤（用戶實測發現）。
11. **data/raw 為 video-only DASH（無音軌）**：訓練只需幀，省一半頻寬；代價是一般播放器（Windows Media Player）無法播放，屬預期行為，看片用 VLC。

## 2026-10-01 Phase 2

1. **Redis 暫以 in-proc queue 承接**（Phase 2 本機）。
   原因：本機無 Redis/Docker runtime；控制訊息（pause/seek/playhead）量小且單進程內 queue 語意等價。介面訊息格式即未來 Redis pub/sub 的訊息契約，Phase 5 compose 拆進程時替換，raw 幀一律不過匯流排（架構硬規則不受影響）。
2. **Session 生命週期 = WS 連線**。POST /session 只做解析與驗證（錯誤走 HTTP status）；WS 連上才啟動 worker（省算力），斷線即停並註銷。單客戶端 demo 模型，第二條連線回 4409。
3. **pause = 殺串流、resume = seek(paused_time)**。暫停期間持續解碼只會累積 media_time 漂移（幀流與播放頭脫鉤），統一走 restart(target − 2s) 一條代碼路徑最簡且正確，代價是 resume 有 ~1–2s 熱身。
4. **幀流解碼 256²、worker 內 center crop 224**（非 ffmpeg 直接出 224）。與訓練/評估路徑完全一致（256 解碼 → center crop 224），避免串流精度與離線評估出現前處理分歧。
5. **file 來源僅限 dev/測試**：E2E 用本機檔案保證可重現；demo 對外只提供 YouTube / direct MP4 兩種輸入。
6. **Phase 2 期間平行做資料擴充**（弱三類各補至 26 支 + unknown 22）：5 類分辨是硬性要求，Phase 1 的 hiphop F1 0.08 不可交付；下載為背景作業，與後端開發無資源衝突。重訓採 K400 init 重新 fine-tune（非續訓舊 checkpoint），避免舊噪音標籤的記憶被帶入。
