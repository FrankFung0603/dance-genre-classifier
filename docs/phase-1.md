# Phase 1 — 資料與模型（完成）

日期：2026-09-30 ～ 2026-10-01

## 做了什麼

### 資料（全部 yt-dlp 自收集，弱標籤）
- 84 支影片（6 類 × 14 支，60–300s；unknown 類放寬至 ≤900s），1.4GB，video-only DASH ≤480p
- 查詢策略：每類 2–3 組關鍵詞；`download_archive` 斷點續收；flat search + `ignoreerrors` 避開死鏈
- 2s 滑窗（stride 2s）切出 **9,216 窗**，K400 預訓練 R(2+1)D-18（ONNX DML，static batch-1）逐窗打分
- **Pseudo-label cleanup**：舞蹈類保留 `dance_prob ≥ 0.10`（K400 的 16 個舞蹈相關類機率和）；unknown 保留 `≤ 0.05`
- 過濾後 **4,545 窗**（每類 684–828，均衡），快取為 `data/windows_u8.npy`（uint8 memmap，14.3GB，訓練零解碼）

### 切分（group split by 來源視頻）
- 每類 14 支按影片分組 70/15/15 → train 3,286 / val 628 / test 631 窗；**同一影片的窗口絕不跨集合**

### 訓練
- R(2+1)D-18（Kinetics-400 初始化）→ 6 類頭；16f @ 8fps，256² 解碼 → 224 random/center crop
- batch 4 × grad accum 4（等效 16）、AMP fp16、AdamW 分組 lr（head 5e-4 / backbone 5e-5）、cosine + 1 epoch warmup、label smoothing 0.1、WeightedRandomSampler 類別平衡、早停 patience 4
- 15 epochs 約 100 分鐘（epoch 6.6 分，`cudnn.benchmark=True` autotune 提速 4×）
- 訓練曲線：train loss 1.145 → 0.459；**val macro-F1 0.765 → 0.848**（best @ epoch 11）

### 匯出與部署驗證
- ONNX opset 18，**static batch-1**（DML EP 不支援 dynamic-batch Reshape——關鍵教訓）
- FP16（onnxconverter-common，keep_io_types）：125.3MB → **62.8MB**
- **Parity：FP32 vs FP16 top-1 一致 32/32，max prob diff 0.0009** → 精度無損
- 延遲（DML EP）：**FP16 85.9 ms/clip**、FP32 87.0 ms（FP16 省一半記憶體，此 GPU 上無延遲增益——誠實記錄）

## 測試集結果（FP16 ORT = 部署等價路徑，n=631）

**macro F1 = 0.619**（注意：顯著低於 val 0.848，分析見下）

| class | precision | recall | F1 |
|-------|-----------|--------|-----|
| ballet | 0.980 | 0.877 | **0.926** |
| breakdance | 0.430 | 0.634 | 0.513 |
| contemporary | 0.369 | 0.339 | 0.353 |
| hiphop | 0.110 | 0.068 | **0.084** |
| salsa | 0.974 | 0.991 | **0.982** |
| unknown | 0.797 | 0.922 | **0.855** |

混淆矩陣見 `phase1_confusion_matrix.png`。結構非常清楚：

- **三類強**：salsa、ballet、unknown 近乎對角（89–99% 正確）
- **一個崩壞集群**：hiphop / breakdance / contemporary 互混——hiphop 僅 8/117 正確（61 窗→breakdance、47 窗→contemporary）

### 為什麼 val 0.848 vs test 0.619？

1. **每類僅 14 支來源影片 → val/test 各只有 2 支/類**。單支影片風格不純就主導整類指標，兩個估計值變異數都極大；val 另用於模型選擇，天然樂觀。
2. **弱標籤噪音集中在街舞集群**：「hip hop dance choreography」類搜尋結果大量是當代/混合風格編舞——標籤邊界本身模糊，非模型容量問題。

### 對應的改進方向（依性價比排序）

1. 街舞三類各補 10–20 支**風格更純**的來源影片（搜尋詞更嚴：battle/competition 類 vs choreography 類）
2. Genre-specific 過濾：hiphop 窗要求 K400 對 hiphop-相關類的機率，而非泛舞蹈機率
3. 人工抽查來源影片標籤（每類 14 支的抽查成本可接受）
4. 報告層面：demo 時聚焦 salsa/ballet + Unknown fallback（已可用），誠實說明街舞集群限制

## 已知問題

- 本機 GPU（RTX 4060 Laptop）訓練吞吐異常低：batch 4 fwd+bwd 實測 2.3 win/s（~430ms/step），慢於同級硬體一個量級。已排除降頻/功耗/溫度/數據管線/memmap；疑似 torch 2.11+cu128 與驅動（592.00）的 kernel 選擇問題。`cudnn.benchmark=True` 後仍獲 4× 提速（1576s→396s/epoch）。未繼續追（不影響正確性，影響疊代速度）
- 中途發生一次 native crash（epoch 5，exit 127）——checkpoint/resume 機制驗證有效，零進度損失
- FP16 無延遲增益（DML EP），僅省 50% 模型體積

## 驗收對照（CLAUDE.md 階段 1）

- ✅ held-out（group split）指標齊全：macro F1、per-class P/R、confusion matrix
- ✅ GPU 單 clip 推理 < 1s：實測 86ms
- ✅ Unknown 有效：unknown 類 F1 0.855，recall 0.922
- ⚠️ 整體指標：macro F1 0.619——三類強、街舞集群弱，限制已誠實記錄於 `limitations.md`

## 產出檔案

- `models/dance_finetuned.pt`（best val F1 0.848 @ epoch 11）、`models/ckpt_latest.pt`（可續訓）
- `models/dance_r2plus1d.onnx`（FP32 單檔）、`models/dance_r2plus1d_fp16.onnx`（部署用）
- `training/log.csv`（每 epoch 曲線）、`docs/phase1_metrics.json`、`docs/phase1_confusion_matrix.png`
- 管線腳本：`data/collect_raw.py` → `data/build_manifest.py` → `data/filter_split.py` → `data/extract_windows.py` → `training/train.py` → `training/export_eval.py`

## 重訓 v2（2026-10-02，弱類修復）

Phase 2/3 完成後，針對「5 類分辨是硬性要求」做的資料擴充 + 重訓。**策略變更輪（資料翻倍 + 過濾規則改變 + split 重切）→ K400 init 重訓；之後增量資料輪走續訓（`--init-from` + replay，見 decisions.md）。**

### 資料擴充
- 84 → **152 支**（舞類 26 × 5 + unknown 22；弱三類補純風格查詢詞：breaking final / freestyle street / contemporary solo stage 等）
- 窗口 9,216 → 16,752；pseudo-label 過濾後 8,218 → 風格矛盾過濾後 **7,520**（train/val/test = 5,184/1,162/1,174）
- **風格矛盾過濾（新）**：K400 top1 明確指向另一個具體舞種且 prob ≥ 0.35 → 從 train 剔除（698 窗）；**val/test 不剔除**（保持測試誠實）
- **test 每類 2 → 4 支來源影片**（unknown 3 支）——指標變異數的根因直接修復

### 訓練
- 同 phase-1 配方，15 epochs + 早停 patience 4；中途 native crash 一次（同已知環境問題）→ `--resume` 零損失恢復
- best = **epoch 3 val macro F1 0.705**（新 val 集 1,162 窗；⚠️ 不可與舊 val 0.848 直接比較——舊集僅 2 支/類，量測本就不準）

### 測試結果（FP16 ORT，n=1,174，group split held-out）
| 類別 | v1 F1（n=631, 2支/類） | **v2 F1（n=1,174, 4支/類）** |
|---|---|---|
| ballet | 0.930 | 0.858 |
| breakdance | 混雜 | **0.923** |
| contemporary | 混雜 | 0.567 |
| hiphop | **0.080** | **0.786** |
| salsa | 0.980 | 0.994 |
| unknown | 0.855 | 0.810（recall 1.000）|
| **macro** | **0.619** | **0.823** |

- **弱三類救活兩個**：hiphop 0.08 → 0.79（precision 0.93）、breakdance → 0.92（recall 0.99）
- 剩餘互混集中在 **contemporary**（F1 0.567）：hiphop→contemporary 55 窗、ballet→contemporary 34 窗、contemporary→unknown 53 窗。根因：contemporary train 窗被矛盾過濾砍最狠（381 窗 vs 其他類 ~1,000）+ 風格本身異質（lyrical/modern 視覺上跨界）。改進方向：補 contemporary 來源影片、對當代舞放寬矛盾閾值（K400 把 lyrical 認成 ballet 屬正常）
- latency 不變（FP16 84.2ms）；parity 32/32
- v1 模型已備份：`models/*_v1.onnx`；舊指標：`docs/phase1_metrics_v1.json`、`training/log_v1.csv`

### 增量訓練基建（本輪順帶完成）
- `build_manifest.py --append`：只打分新影片
- `extract_windows.py --append`：舊窗口 memmap 重定位（零解碼）+ 只解碼新窗；自動記錄 `data/last_append_keys.json`
- `train.py --init-from`（續訓權重）+ `--new-keys` + `--replay-frac`（增量輪：新窗口為主 + 30% 舊窗 replay 防災難性遺忘；val/test 恆全量）

## 重訓 v3（2026-10-04，增量輪：弱類補強 + salsa 分佈多樣化）

**資料**：176 支（+24：contemporary +8、hiphop +6、salsa +6、unknown +4）。salsa 增量動機 = 用戶實測線上預測率低於測試集 → 分佈多樣化（social/街頭/節日）。
**過濾策略變更**：contemporary 矛矛盾閾值 0.35 → **0.55**（per-genre，filter_split.CONTRADICT_THR_BY_GENRE）——K400 無 contemporary 對應類，lyrical 被認成 ballet 屬預期；舊閾值把其 train 窗砍到 381。放寬後 **381 → 729 窗（+91%）**。contemporary 被剔除窗 676→無統計但大幅減少。
**split**：--pin-existing 凍結舊影片歸屬（v2 test 影片不動），只對新影片 70/15/15 → 新 test n=1,414（舊 1,174 + 新 240）。
**訓練**：--init-from v2 best、908 新窗 + 30% replay（1,555 舊窗）、8 epochs、lr head 1e-4 / backbone 1e-5；best = epoch 7 val 0.759（⚠️ val 集已擴，不可與 v2 的 0.705 直接比）。
**同集對比（v2 vs v3 都跑在同一個新 test 上，FP16 ORT 部署等價路徑）**：

| 類別 | v2 | v3 | Δ |
|---|---|---|---|
| ballet | 0.853 | 0.810 | −4.3 |
| breakdance | 0.918 | 0.929 | +1.1 |
| contemporary | 0.646 | **0.756** | **+11.0** |
| hiphop | 0.837 | 0.831 | −0.6 |
| salsa | 0.995 | 0.984 | −1.1 |
| unknown | 0.864 | **0.965** | **+10.1** |
| **macro** | **0.852** | **0.879** | **+2.7** |

- unknown 進步本質 = precision 0.760 → 0.957（誤報大減），recall 1.000 → 0.973 可接受
- ballet −4.3 是放寬閾值的預期代價（部分 ballet 窗讓給 contemporary），淨收益為正
- 延遲不變：FP16 DML 100ms/clip、parity 32/32
- **v3 已部署**；v2 完整備份：`models/*_v2.{pt,onnx}`；v2 同集評測 log：`training/export_v2_sametest_final.log`

## CPU 推理對比（2026-10-09，同模型同測試集，僅換 execution provider）

**動機**：面試跟進要求提供 CPU result。同一 v3 部署權重（`dance_r2plus1d_fp16_v3.onnx`，權重=線上部署檔）、同一 test set（n=1,414）、同一前處理路徑（與 `export_eval.py` 逐字共用 decode/preprocess），唯一變數 = DML EP → CPUExecutionProvider。
**工具**：`spikes/eval_cpu_inference.py`；完整 log `spikes/eval_cpu_inference_v3.log`；指標 JSON `docs/cpu_inference_metrics.json`

| | GPU（DML FP16，官方 v3） | CPU（CPU EP，同檔 FP16） |
|---|---|---|
| macro F1 | 0.8793 | **0.8799** |
| accuracy | 0.874 | 0.875 |
| 延遲（bench mean，隨機輸入） | 100.0 ms/clip | **486.8 ms/clip（~4.9×）** |
| 測試集逐窗推理 mean | — | 513.2 ms/clip |

- per-class F1 幾乎逐項相同（最大差異 ballet 0.810→0.813）→ **精度對硬體不敏感，0.879 在 CPU 上原樣保留**（FP16 在 CPU EP 只影響亂序浮點誤差級差異）
- 對照 Phase 5：容器內 CPU fallback 611 ms/clip > 本機 487 ms（容器執行緒限制），量級一致
- 結論：1 Hz 更新預算下 CPU 單窗 ~0.5–0.6s 仍可負擔；compose E2E 已驗證 CPU fallback 全流程可跑（無 GPU 機器可部署，只是延遲預算變緊）
- train.py 順帶支援 `--device cpu`（FP32、無 AMP；預設 cuda 行為不變）；CPU 訓練對比實驗暫緩——GPU 側基準已有：7.7 win/s、376 s/epoch、8 epochs 共 ~50 分鐘
