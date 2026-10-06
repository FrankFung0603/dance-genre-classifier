# Phase 3 — 前端與時間對齊（完成，2026-10-01）

日期：2026-10-01（用戶瀏覽器實測通過；seek 狀態機修復後待複測一次）

## 做了什麼

### 頁面與組件（`frontend/`，Next.js 16.3.8 + TS + Tailwind）
- `app/page.tsx`：URL 輸入 → **播放器與 session 建立並行**（影片立刻播，AI 背景啟動，省 ~1.6s yt-dlp 解析的體感等待）
- `components/PlayerPanel.tsx`：YouTube IFrame（react-youtube）+ Pose Canvas overlay
- `components/Mp4Panel.tsx`：direct MP4 fallback 播放器（`<video>`，同一條後端 pipeline）
- `components/Dashboard.tsx`：當前 genre、confidence bar、top-3、最近 10 秒 history、inference 延遲
- `components/FeedbackBar.tsx`：👍/👎 + 正確類別下拉（Phase 3 本地記錄，Phase 4 入庫）
- `lib/useMediaClock.ts`：**YouTube/MP4 共用播放頭時鐘**（250ms 心跳 + pause/resume/seek 偵測）
- `lib/useDanceSession.ts`：WS 生命週期 + **media_time buffer**（核心同步機制）

### 時間對齊（核心機制）
- 預測/pose 由後端帶 `media_time` 串流 → 前端 buffer（`[t_start, t_end)` 命中顯示；±0.7s 內退最近中心）
- 播放頭每 250ms 回報；buffer 自動清 TTL 30s 舊項
- **顯示的是「播放頭所在時間」的預測，不是「最新一筆」**——預測可領先播放頭 ~8s（後端 pacing），畫面永遠對齊

### 播放狀態機（兩輪迭代修出來的教訓）
YouTube player state：1 播放 / 2 暫停 / 3 buffering / 0 結束。
- ❌ 初版：任意非播放狀態 = pause → **拖進度條的 buffering 被誤判**，一次 seek 連開三次 ffmpeg（pause 殺串流 → resume 重啟 → debounce seek 再重啟），seek 後等很久
- ❌ 二版：只認 1↔1 的時間跳躍 → buffering 期間 seek 偵測被擋，**seek 後不再出預測**
- ✅ 定版：pause/resume 只認 1↔2 乾淨轉換；ended (0) 送 pause 乾淨收工；seek 跳躍偵測在 1/3 狀態皆觸發（`playingSeen` 防初始載入誤判）；seek 後 UI 顯示「重新同步中…」直到新串流第一個預測抵達

## 驗收（CLAUDE.md 階段 3）
- [x] URL 輸入 + YouTube IFrame + Dashboard
- [x] 250ms 回報 currentTime / paused / seek（狀態變化立即送 + 心跳）
- [x] 預測 buffer 按 media_time 存放，播放到該時間才顯示
- [x] Dashboard：genre / confidence bar / 10 秒 history
- [x] Canvas overlay：Pose landmarks（輔助解釋標註）
- [x] 反饋 UI：👍/👎 + 正確類別下拉
- 用戶實測：「網站沒有問題」；seek 延遲修復後「快了很多」

## 已知問題
- Pose overlay 以正規化座標拉伸至播放器矩形；影片長寬比與播放器不同時骨骼會輕微變形（展示可接受）
- MP4 fallback 的 `<video>` 未處理縱向影片特別佈局（demo 用橫向來源）
- 反饋目前僅本地 console/state，Phase 4 接 PostgreSQL
