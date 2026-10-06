# Phase 2 — 串流與推理後端（完成）

日期：2026-10-01

## 做了什麼

### 架構（`server/`，FastAPI 單進程 dev 拓撲）

```
客戶端 ──WS /ws/{sid}── FastAPI(main.py)
                          │  控制指令（playhead/pause/resume/seek，seek 經 500ms debounce）
                          ▼
                InferenceWorker 執行緒（worker.py）
                  ├─ FFmpegStream（frames.py）── stdout 直讀 rawvideo rgb24 256²@8fps，不落盤
                  ├─ ONNX FP16 DML（靜態 batch-1，進程級快取 session）
                  ├─ LabelSmoother（smoothing.py）── EMA + Unknown 閾值 + hysteresis + 最小持續
                  └─ PoseRunner（pose.py）── 同一幀流 4Hz 抽樣，landmarks（輔助解釋）
```

- **來源解析（resolver.py）**：YouTube → yt-dlp（video-only avc1 ≤480p，TTL 30min cache，斷流 force 重解析）；http(s) → direct MP4 直通；本機路徑 → file（僅 dev/測試）。三種來源走**同一條** ffmpeg pipe。
- **滑窗推理**：16 幀（2s）窗口，stride 8 幀（1s）= 50% 重疊 → 1Hz 預測；`media_time = offset + frame_idx/8`，seek 重啟即重置。
- **預測 payload**：`{media_time, t_start, t_end, genre, confidence, genre_raw, probs×6, topk×3, latency_ms}`；pose 獨立訊息 `{media_time, landmarks[33][3]}`。
- **pacing**：解碼位置領先播放頭 > 8s 即暫停讀幀（播放頭無回報時退化為 1× 牆鐘估計）——預測適度超前供前端 buffer，又不整支白算。
- **pause / seek**：pause 殺串流記住時間點；resume/seek 統一走 restart(target − 2s warmup)，清 buffer、清平滑狀態。seek debounce 500ms 只吃最後一個。
- **容錯**：斷流自動 force-reparse + 從當前位置恢復一次；再失敗發 `fatal`。壞 URL/解析失敗在 `POST /session` 就回 HTTP 錯誤。

### 與規範的偏差（記入 decisions.md）

- **Redis 未接入**（本機無 Redis/Docker runtime）：控制訊息暫以 in-proc queue 承接，介面與訊息格式即未來 Redis 控制匯流排的訊息契約；raw 幀本來就不過匯流排。Phase 5 compose 拆進程時接入。
- Session 生命週期 = WS 連線（單客戶端 demo 模型）；第二條 WS 連同 session 回 4409。

## 驗收結果（E2E：`server/e2e_test.py`，15 檢查 × 3 來源全過）

| 檢查 | file | YouTube | direct MP4 |
|---|---|---|---|
| session 建立 / ready / stream_started | ✅ | ✅ | ✅ |
| 預測流（12s 內 ≥6 則，~0.71s/則） | ✅ | ✅ | ✅ |
| media_time 單調遞增 | ✅ | ✅ | ✅ |
| payload schema（6 類 probs + topk） | ✅ | ✅ | ✅ |
| pose 4Hz | ✅ 61 則 | ✅ 61 則 | ✅ 15 則 |
| 幀到預測 <1s | — | **mean 127ms / max 250ms** | **mean 121ms / max 390ms** |
| seek(30s) 後首預測位置 | ✅ | ✅ 30.0 | ✅ 30.0 |
| pause 停預測 / resume 恢復 | ✅ | ✅ | ✅ |
| 斷線清理 session | ✅ | ✅ | ✅ |

- YouTube 來源正確輸出 ballet；direct MP4（Big Buck Bunny 卡通）**全程 unknown**——Unknown fallback 在真實串流路徑上有效。
- 「不下載完整影片」：三種來源皆 pipe 直讀，無落盤路徑。

## 已知問題

- direct URL（含 YouTube）IP 綁定 + 時效限制仍在（見 limitations.md）；worker 的 force-reparse 恢復路徑已驗證（人工斷流測試），但未對 6h 時效做長跑驗證。
- Big Buck Bunny 的 google sample bucket 從本網路 403（demo 備援清單要挑過、寫死在 README 的候選需定期驗證）。
- pose 在廣景/多人/剪影畫面會無偵測（正常行為，前端需處理 landmarks 空窗）。
- 幀到預測延遲是在 worker 內量測（最後一幀讀完 → emit）；含 ffmpeg 解碼的端到端延遲未單獨拆項（首預測總延遲由 limitations.md 的 3–4s 口徑涵蓋）。
