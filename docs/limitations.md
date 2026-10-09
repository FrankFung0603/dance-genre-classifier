# 已知限制（持續更新）

## 模型（v2 重訓後，2026-10-02）

- **測試集 macro F1 = 0.823**（group split held-out，n=1,174，每類 4 支來源影片）。v1 為 0.619（n=631，每類僅 2 支）——兩輪 test 集組成不同，方向性結論「弱類大幅改善」可信，個別數字仍是點估計。
- **contemporary 仍是最弱類（F1 0.567）**：與 hiphop（55 窗誤入 contemporary）、ballet（34 窗）互混，另 53 窗落入 unknown。根因：contemporary train 窗被風格矛盾過濾砍最狠（381 窗 vs 其他類 ~1,000）+ 風格本身異質（lyrical/modern 視覺跨界）。改進：補 contemporary 影片、放寬其矛盾閾值。
- hiphop recall 0.680：約 1/3 的 hiphop 窗被分到 contemporary——demo 時 hiphop 影片會出現「hiphop 與 contemporary 交替」的顯示，屬已知行為。
- unknown recall 1.000 / precision 0.681：寧可說 unknown 也不硬猜舞蹈，與 fallback 策略一致。
- 152 支來源影片的「有效樣本量」仍是每類 ~26 個獨立來源；要生產級可靠需每類 100+ 支 + 人工驗證標籤。
- v1 模型備份於 `models/*_v1.onnx`，回滾 = 改回檔名重啟。

## 訓練基設

- 本機 GPU 訓練吞吐異常（batch 4 fwd+bwd ≈ 430ms，慢同級硬體 ~10×）；疑似 torch 2.11+cu128 與驅動 592.00 的 kernel 問題，`cudnn.benchmark=True` 後 epoch 26→6.6 分鐘（4× 提速）。不影響正確性。
- 訓練中途曾發生一次 native crash（CUDA 層）；checkpoint（每 200 步 + 每 epoch，原子寫入）+ `--resume` 已驗證可零損失恢復。
- Windows spawn 模式下 numpy memmap 物件**不可** pickle 進 DataLoader worker（truncated pickle / Errno 22）——必須 worker 內 lazy-open 路徑。

## 推理

- ONNX 模型必須 **static batch-1**：DML EP 不支援 dynamic-batch 的 Reshape（CPU EP 可，易誤導）；ORT 錯誤訊息含二進位亂碼，會包成 UnicodeDecodeError 掩蓋真錯誤。
- **容器部署數字（Phase 5 實測）**：CUDA EP 幀→預測 mean 102ms / max 320ms；同 image 無 GPU 時自動落 CPU EP（單 clip 611ms，仍在 1Hz 預算內）。TensorRT INT8 為 stretch goal，未排程。
- **EP 間數值差異可見**：Big Buck Bunny（卡通）在主機 DML 為全片 unknown，容器 CUDA EP 下偶有 1–4 窗漏判 contemporary（unknown 閾值邊緣的浮點差異被放大）。「非舞蹈必 unknown」不成立，僅「≥80% unknown」。
- FP16 於 DML EP 無延遲增益（85.9 vs 87.0 ms），僅省 50% 體積。
- 容器基底含 CUDA/cuDNN runtime（~3GB）；GPU passthrough 需 WSL2 + NVIDIA 驅動（本機已驗證），純 CPU 主機刪 compose 的 `gpus: all` 一行即可。

## 串流

- **yt-dlp direct URL 有 IP 綁定與約 6 小時時效**，且 YouTube 可能對下載限速。緩解：session 內偵測斷流後自動重新解析恢復（Phase 2 已實作，單次斷流可救）；提供 direct MP4 URL 輸入 fallback（同一條 pipeline）。
- YouTube 介面/格式策略隨時變動（已遇到 progressive 格式消失、搜索結果死鏈），yt-dlp 需保持最新版。
- 首個預測天然下限 ≈ 2s（16 幀 @ 8fps 窗口填充）+ 解析 1.6s + 首幀 0.7s；「貼 URL 到第一個 overlay」實際約 3–4s。
- 公開 MP4 備援候選需定期驗證（Big Buck Bunny 的 google sample bucket 已從本網路 403；archive.org 副本目前可用，見 phase-2.md）。
- CAM 歸因為空間歸因（14×14 網格，一格 ≈32px）：回答「模型看哪裡」，不回答概念層「為什麼是這種舞」；unknown 類熱圖最難解讀。fp16 轉換時 `node_mean` 節點保留 FP32（block-list 迭代修復），雙輸出模型 logits 與舊單輸出版 parity 16/16 一致。
- worker 幀到預測延遲 mean ~125ms（worker 內量測，三來源一致；CAM 為同次 forward 的 numpy 後處理，毫秒級）；含 ffmpeg/網路的端到端首預測延遲以 3–4s 口徑承諾。

## 前端

- IFrame 跨域限制：無法讀取幀像素，所以分析與歸因都在後端跑（CAM 由 worker 同一條幀流計算，前端只負責渲染）。
- 首幀播放延遲由 YouTube player 決定，不可控、不承諾 500ms。

## 邊緣案例量化範圍（Phase 4，2026-10-04）

- **已量化**：非舞蹈 fallback（unknown recall 1.000、BBB 全片 unknown）、風格切換的機制參數（hysteresis 0.10 + 最小持續 2s，切換延遲 1–2s 但未逐幀標註）。
- **未量化**（無專門標註測試集）：多舞者場景的類別準確率、鏡頭移動/運鏡下的穩定性。模型對這兩者無專門處理，表現依賴訓練分佈（≤480p 自收集影片）。
- 反饋閉環的 `correct` 標籤來自 demo 使用者主觀判斷，未經交叉驗證；僅作為下一輪訓練的候選修正樣本，不直接回寫測試指標。


## 背景依賴（occlusion test，2026-10-09）

模型確實**部分依賴背景/場景細節**判斷舞種（弱標籤 + 每類影片場景集中的預期後果），已用遮蔽實驗量化（`spikes/eval_background_occlusion.py`，test n=1,414，中心 112² 保留）：

| 變異 | macro F1 | flip 率 | 解讀 |
|---|---|---|---|
| 原窗（sanity） | 0.879 | — | 與官方值一致 |
| **背景模糊**（主體保留） | **0.562** | **34.7%** | 背景細節貢獻真實存在（−0.32） |
| **主體模糊**（背景保留） | **0.102** | 80.5% | 主體（舞者區）仍是主訊號 |

- per-class 背景模糊 flip：breakdance **82.5%**（街頭/圍觀人群場景先驗強）≫ ballet 35.8% ≈ contemporary 33.5% ≈ salsa 32.3% > hiphop 27.8%；unknown 僅 3.8%（非舞蹈判斷對背景模糊穩健）
- **馬賽克版數字（F1 0.204、塌縮到 hiphop、信心反升 0.90）是 OOD 偽象**：模型沒見過馬賽克，輸入分佈偏移導致過度自信亂猜，不代表背景依賴度——引用時一律用模糊版
- 結論：模型讀「舞者為主、場景為輔」，不是只看背景；但背景細節去掉後 1/3 預測會變，面對「換場景的同类影片」（如街頭芭蕾、舞台 breakdance）可靠度會下降
- 緩解方向（未實作）：場景多樣化的資料收集（同舞種跨場景）、背景區域增強（隨機模糊/替換）、把 occlusion test 列為常規 robustness 指標

**v4 更新（2026-10-09，bg-augment 續訓後已部署）**：背景模糊 F1 0.562→**0.835**、flip 34.7%→**10.5%**（breakdance 82.5%→2.3%）；乾淨 F1 代價 −1.1（0.879→0.868，主要 hiphop recall −15）。contemporary flip 28.4% 仍最高；「反場景素材」（街頭芭蕾等）可靠度已大幅改善但未完全消除。

## 範圍外舞蹈風格的歸類行為（2026-10-09 實測）

類別集 = 5 種舞 + unknown（unknown 定義為**非舞蹈**），不存在「其他舞蹈」出口。範圍外舞種會歸入**最近的已知原型**，實測（v4，每風格 ~86 窗直播預測）：

| 風格 | 歸宿 | 信心 |
|---|---|---|
| krump（世界賽） | breakdance 47% + unknown 51% | — |
| locking | hiphop 62% | 0.60 |
| K-pop 練習室 | hiphop 99% | **0.94** |

- 並非全部湧向 hiphop（krump→breakdance）——各歸最近視覺/動作原型，非殘餘桶行為
- **信心值無法區分「真 hiphop」與「最近原型 hiphop」**（locking 0.60 ≈ 真實 hiphop 常見的 ~0.60）——封閉集分類的本質限制；可參考 UI top-3 margin（margin 大 = 判斷更「純」）
- 5 類範圍內素材判斷可信（test hiphop precision 0.88）；demo/評估請使用範圍內素材
- 若未來需要真正處理範圍外舞種：新增類別需 K400 重訓 + 每類 ~10-14 支資料（決策 #23 同源評估流程），當前決定不做
