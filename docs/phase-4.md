# Phase 4 — AI-native 與邊緣處理

日期：2026-10-04

## 1. 反饋閉環（PostgreSQL）✅

### 架構

```
前端 FeedbackBar (👍/👎 + 正確類別下拉)
  → POST /feedback { session_id, url, kind, media_time, predicted, confidence, correct, verdict }
  → SQLAlchemy → PostgreSQL (feedback 表)
  → GET /feedback?session_id=...（反饋可查）
```

- **表結構**（`server/db.py`）：

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | serial PK | |
| session_id | varchar, indexed | 對應 POST /session 發的 sid |
| url | text | 輸入來源（YouTube / direct MP4） |
| kind | varchar | youtube / mp4 / file |
| media_time | float | 反饋當下的播放頭媒體時間 |
| predicted | varchar | 當下（平滑後）預測，含 unknown |
| confidence | float | 平滑後信心 0–1 |
| correct | varchar nullable | 👍 = NULL；👎 = 使用者選的正確類別 |
| verdict | varchar | up / down |
| created_at | timestamptz | |

- **連線**：`DATABASE_URL` 環境變數，dev 預設 `postgresql+psycopg2://dance:dance@localhost:5432/dance_feedback`（docker 容器）。
- **降級策略（demo 不開天窗）**：DB 不可用時服務照常起（`/healthz` 顯示 `db:false`），`/feedback` 回 503，前端顯示「後端離線，僅本地暫存」並保留本地記錄。
- **輸入驗證**：`predicted`/`correct` 必須屬於 CLASSES，否則 400。

### E2E 驗收（2026-10-04 實測）

```
POST /feedback ×2（up 與 down+correct） → {"stored":true} ×2
GET /feedback?session_id=e2e-test-001  → count=2，欄位齊全
POST predicted="foo"                   → 400
psql SELECT                           → 2 rows（up: correct=NULL；down: correct=hiphop）
/healthz                               → {"db":true,...}
```

✅ 驗收「反饋可查」達成。

### 反饋的用途（誠實範圍）

本階段反饋是**資料收集**：累積 (media_time, 預測, 正確類別) 標註對，可作為下一輪增量訓練的修正樣本來源（配合既有 `--append` 增量管線）。並未做到線上即時學習——demo 期間不做模型熱更新。

## 2. 可解釋 UI ✅（Phase 3 已上線，本階段補文件）

- Pose 骨骼 overlay（後端 MediaPipe 4Hz 抽樣 → WS → Canvas，霓虹發光繪製）
- UI 明確標註「Pose landmarks 為輔助解釋，非模型歸因依據」——**不是** saliency、**不是** Grad-CAM

## 3. Unknown 策略 ✅（Phase 2 實作，本階段補參數量化文件）

實作：`server/smoothing.py`（LabelSmoother），每秒一次：

1. **EMA**：`probs_ema = 0.4·p + 0.6·probs_ema`（`EMA_ALPHA=0.4`）
2. **Unknown 條件**（二選一）：`unknown_ema > known_top1_ema` 或 `known_top1_ema < 0.45`（`UNKNOWN_THRESHOLD=0.45`）
3. **Hysteresis**：新類 EMA 需壓過當前類 + 0.10（`HYSTERESIS_MARGIN=0.10`）
4. **最小持續時間**：需連續 2 次勝出才切換（`MIN_DURATION_UPDATES=2`）
5. seek/重啟即 reset（時間軸斷裂，平滑狀態無意義）

量化效果（v2 模型，test n=1,174 窗，group split）：

| 指標 | 數值 |
|---|---|
| unknown F1 / recall | 0.810 / **1.000** |
| unknown 誤殺（舞蹈被判 unknown） | test 集零發生 |
| 非舞蹈誤判為舞種 | Big Buck Bunny 全片 unknown（Phase 2 E2E 實測） |

recall 1.000 意味：策略偏向「寧可 unknown 也不要硬掰舞種」，符合 demo 誠實原則。

## 4. 邊緣案例：處理機制與量化結果

| 情境 | 處理機制 | 量化結果 | 狀態 |
|---|---|---|---|
| 非舞蹈內容（卡通/會議） | unknown 負樣本訓練 + 閾值 + EMA | BBB 全片 unknown；unknown recall 1.000 | ✅ 已量化 |
| 突然切換（風格變化） | hysteresis margin 0.10 + 最小持續 2s → 切換延遲 1–2s | 訓練窗本身含風格混合影片；切換延遲未單獨標註 | ⚠️ 機制有、延遲未量化 |
| 低分辨率 | 抽幀固定 224×224（來源 ≤480p 已是訓練常態） | 訓練/測試全部來自 ≤480p，實戰解析度 ≈ 訓練分佈 | ✅ 分佈內 |
| 多舞者 | 模型隱式學習（無特別處理）；pose overlay 取顯著人形 | 未單獨標註量化 | ❌ 未量化，記入 limitations |
| 鏡頭移動 | R(2+1)D 對全域運動敏感，無專門增強 | 未單獨標註量化 | ❌ 未量化，記入 limitations |

誠實結論：unknown 與切換場景有機制與部分數據；多舞者/鏡頭移動需要專門標註的測試集才能量化，本專案未建立（見 limitations.md）。

## 5. 已知問題

- 反饋的 `correct` 依賴使用者從 6 類下拉選擇；「其實是第五類」這種自由文字情境不支援。
- `GET /feedback` 無分頁（limit 上限 500）；demo 規模足夠。
- Docker 化的 postgres 目前是 `docker run` 單容器，Phase 5 會併入 docker-compose 統一管理。
