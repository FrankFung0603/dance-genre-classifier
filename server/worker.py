"""Inference worker thread：直接持有 ffmpeg，滑窗推理 + CAM 歸因 + 控制指令。

一次循環：
    ffmpeg 幀 → deque(16) → 每 8 幀（1s, 50% 重疊）推理一次 → 平滑 → emit 預測
    每次推理同時算 CAM 歸因熱圖（GAP+FC 結構 ⇒ cam = ReLU(Σ fc_w[class]·feat)，
    免反傳播）→ 掛進 prediction payload（media_time 對齊由前端 buffer 機制復用）
控制：
    pause → 殺串流、記住時間點；resume/seek → restart(target)、清平滑
    playhead → 客戶端播放頭回報（pacing 用，避免預測領先太多白算）
media_time 對齊：media_time(frame i) = offset + i / 8（offset 是本次串流的媒體起點）
"""
import collections
import os
import queue as queue_mod
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from server.config import (
    CROP_SIZE,
    LOOKAHEAD_S,
    MEAN,
    PLAYHEAD_STALE_S,
    STREAM_FPS,
    STREAM_SIZE,
    STRIDE_FRAMES,
    STD,
    TOPK,
    WIN_FRAMES,
    CLASSES,
)
from server.frames import FFmpegStream
from server.smoothing import LabelSmoother

FRAME_BYTES = STREAM_SIZE * STREAM_SIZE * 3

# DML session 建立要數秒 → 全進程快取，多 session 共用（session.run 執行緒安全）
_ORT_LOCK = threading.Lock()
_ORT_SESSIONS: dict[str, ort.InferenceSession] = {}
_FC_LOCK = threading.Lock()
_FC_WEIGHTS: dict[str, np.ndarray | None] = {}


def _fc_weights(model_path: str) -> np.ndarray | None:
    """(6,512) 分類層權重（行序=CLASSES）。缺檔 → None（預測照常，僅無 cam）。"""
    with _FC_LOCK:
        path = str(Path(model_path).parent / "fc_weights.npy")
        if path not in _FC_WEIGHTS:
            try:
                _FC_WEIGHTS[path] = np.load(path)
            except Exception:
                _FC_WEIGHTS[path] = None
        return _FC_WEIGHTS[path]


def _cam_grid(feats: np.ndarray, w: np.ndarray) -> list[float]:
    """feats (C,T,H,W)、單類權重 w (C,) → 歸一化 H×W grid（row-major）。"""
    cam = np.maximum(np.einsum("k,kthw->thw", w, feats), 0.0).mean(axis=0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return [round(float(v), 3) for v in cam.ravel()]


def _providers() -> list[str]:
    """GPU EP 偏好序：CUDA（Linux 容器）> DML（Windows 主機）> CPU（永遠在場）。

    DANCE_ORT_PROVIDER=cpu 可強制 CPU 推理（實測 ~0.5–0.6s/clip，1Hz 預算內）；
    容器跑在無 GPU 環境時 CUDA EP 不在 available list / 初始化失敗 → ORT
    自動落到 CPUExecutionProvider（CPU fallback 模式，CLAUDE.md Phase 5 要求）。
    """
    if os.environ.get("DANCE_ORT_PROVIDER", "").strip().lower() == "cpu":
        return ["CPUExecutionProvider"]
    avail = set(ort.get_available_providers())
    gpu = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail]
    return gpu + ["CPUExecutionProvider"]


def get_session(model_path: str) -> ort.InferenceSession:
    with _ORT_LOCK:
        sess = _ORT_SESSIONS.get(model_path)
        if sess is None:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            try:
                sess = ort.InferenceSession(
                    model_path, sess_options=so, providers=_providers(),
                )
            except Exception:
                # GPU EP 在場但初始化失敗（驅動/裝置缺失）→ 純 CPU 兜底
                sess = ort.InferenceSession(
                    model_path, sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
            print(f"[ort] model={Path(model_path).name} providers={sess.get_providers()}", flush=True)
            _ORT_SESSIONS[model_path] = sess
        return sess


def prewarm(model_path: str) -> dict:
    """啟動時預熱：ORT session + dummy 推理（暖 DML kernel）+ CAM 權重載入。

    這些原本 lazy 發生在「用戶貼第一支影片」的當下（加起來可觀），
    移到 lifespan 後用戶首個 session 只付管線固有的 ~3s（解析+首幀+窗口填充）。
    """
    t0 = time.monotonic()
    sess = get_session(model_path)
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.zeros(shape, dtype=np.float32)
    sess.run(None, {inp.name: dummy})
    _fc_weights(model_path)
    return {"model_s": round(time.monotonic() - t0, 2)}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class InferenceWorker(threading.Thread):
    """source_getter(force) -> (kind, playable_url)；kind ∈ {youtube, mp4, file}。"""

    def __init__(self, model_path: str, source_getter, sink, *,
                 start_offset: float = 0.0):
        super().__init__(daemon=True, name="inference-worker")
        self.model_path = model_path
        self.source_getter = source_getter
        self.sink = sink
        self.start_offset = start_offset
        self.cq: queue_mod.Queue = queue_mod.Queue()
        self._stopping = False

        self.stream = FFmpegStream("")
        self.smoother = LabelSmoother()
        self.fc_w = _fc_weights(model_path)

    # ---- public（任意執行緒呼叫） ----
    def control(self, msg: dict) -> None:
        self.cq.put(msg)

    def stop_and_join(self, timeout: float = 5.0) -> None:
        self.control({"type": "stop"})
        self.join(timeout=timeout)
        self.stream.stop()

    # ---- 內部 ----
    def _emit(self, msg: dict) -> None:
        try:
            self.sink(msg)
        except Exception:
            pass  # sink 失敗（客戶端斷線）不該弄死 worker

    def _handle_control(self, msg: dict) -> None:
        t = msg["type"]
        if t == "stop":
            self._stopping = True
        elif t == "pause":
            self._pause()
        elif t in ("resume", "seek"):
            # media_time=0 是合法位置（片頭）——只對 None 退化到 paused_at
            mt = msg.get("media_time")
            target = float(mt) if mt is not None else float(self.paused_at or 0.0)
            self._restart(target, reason=t)
        elif t == "playhead":
            self.client_playhead = float(msg["media_time"])
            self.client_ts = time.monotonic()


    def _pause(self) -> None:
        if not self.playing:
            return
        # 暫停點優先用客戶端播放頭（更準）；退化解碼位置
        if time.monotonic() - self.client_ts < PLAYHEAD_STALE_S:
            self.paused_at = self.client_playhead
        else:
            self.paused_at = self.offset + self.frames_read / STREAM_FPS
        self.playing = False
        self.stream.stop()
        self._emit({"type": "state", "playing": False, "media_time": round(self.paused_at, 3)})

    def _restart(self, target: float, reason: str, force_refresh: bool = False) -> None:
        """重啟串流到 target。offset=target 本身（不吃 warmup）：
        舊設計（target-2s + 丟棄第一窗）讓首個預測剛好蓋住 seek 點，但代價是
        預測流天生落後播放頭 2s 起步——解碼慢的來源（YouTube 節流）會一直追不回，
        儀表板永遠空白。offset=target 第一窗即真實內容，落後只剩啟動延遲 ~1s。"""
        try:
            kind, playable = self.source_getter(force=force_refresh)
            self.stream.url = playable
            offset = max(0.0, target)
            self.stream.restart(offset)
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "fatal", "detail": f"restart failed ({reason}): {e}"})
            self._stopping = True
            return
        self.offset = offset
        self.frames_read = 0
        self.frames_at_restart = 0
        self.last_restart_reason = reason
        self.window_buf.clear()
        self.smoother.reset()
        self.playing = True
        self.wall_at_restart = time.monotonic()
        # 注意：不在這裡清 consecutive_failures——重啟後若立刻又讀不到幀，
        # 計數要跨重啟累積才會觸發 fatal（否則無限重啟循環）。
        self._emit({"type": "state", "playing": True,
                    "media_time": round(target, 3), "reason": reason})

    def _playhead_estimate(self) -> float:
        fresh = time.monotonic() - self.client_ts < PLAYHEAD_STALE_S
        if fresh:
            return self.client_playhead + (time.monotonic() - self.client_ts)
        # 沒回報 → 假設 1× 播放：重啟點 + 牆鐘差
        return self.offset + (time.monotonic() - self.wall_at_restart)

    def _preprocess(self, window: np.ndarray) -> np.ndarray:
        """uint8 (16,256,256,3) → float32 (1,3,16,224,224)。與 export_eval 完全一致。"""
        x = window.astype(np.float32) / 255.0
        off = (STREAM_SIZE - CROP_SIZE) // 2
        x = x[:, off:off + CROP_SIZE, off:off + CROP_SIZE, :]
        x = (x - MEAN) / STD
        return x.transpose(3, 0, 1, 2)[None]

    # ---- 主循環 ----
    def run(self) -> None:
        """防護殼：任何未捕捉例外都要變成 fatal 推給前端，嚴禁 thread 靜默死亡
        （靜默死亡 = WS 開著但永遠沒訊息 = 前端「加載不了」且無提示）。"""
        try:
            self._run_loop()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._emit({"type": "fatal", "detail": f"worker crashed: {e}"})

    def _run_loop(self) -> None:
        self.sess = get_session(self.model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.window_buf: collections.deque = collections.deque(maxlen=WIN_FRAMES)
        self.client_playhead, self.client_ts = 0.0, 0.0
        self.consecutive_failures = 0

        self.playing = True
        self.paused_at = None
        self._restart(self.start_offset, reason="start")
        if self._stopping:
            return
        self._emit({"type": "stream_started", "media_time": round(self.offset, 3)})

        while not self._stopping:
            # 1) 控制指令
            try:
                while True:
                    self._handle_control(self.cq.get_nowait())
            except queue_mod.Empty:
                pass
            if self._stopping:
                break
            if not self.playing:
                time.sleep(0.05)
                continue

            # 2) pacing：解碼位置超過播放頭 + lookahead 就等（避免整支提前算完）
            decoded_t = self.offset + (self.frames_read + 1) / STREAM_FPS
            if decoded_t - self._playhead_estimate() > LOOKAHEAD_S:
                time.sleep(0.05)
                continue

            # 3) 讀幀
            raw = self.stream.read_frame()
            if raw is None:
                self.consecutive_failures += 1
                progressed = self.frames_read - self.frames_at_restart
                if self.consecutive_failures >= 2:
                    if progressed < STREAM_FPS:
                        if self.last_restart_reason in ("seek", "resume"):
                            # 使用者 seek/resume 後解不出內容：超出片長或來源失效。
                            # 明確 fatal，別誤報 ended（前端看起來像「沒反應」）。
                            cur = self.offset + self.frames_read / STREAM_FPS
                            self._emit({"type": "fatal",
                                        "detail": f"seek 到 {cur:.1f}s 後串流無法恢復"
                                                  "（可能超出影片長度或來源失效）"})
                        else:
                            # 起播即在 EOF → 影片內容自然結束，非錯誤
                            self._emit({"type": "state", "playing": False, "reason": "ended",
                                        "media_time": round(
                                            self.offset + self.frames_read / STREAM_FPS, 3)})
                    else:
                        self._emit({"type": "fatal",
                                    "detail": "stream failed twice, giving up"})
                    self._stopping = True
                else:
                    if getattr(self, "recover_count", 0) >= 5:
                        self._emit({"type": "fatal",
                                    "detail": "too many stream recoveries, giving up"})
                        self._stopping = True
                        continue
                    self.recover_count = getattr(self, "recover_count", 0) + 1
                    # direct URL 過期/網路抖動 → 強制重解析 + 從當前位置恢復（offset=當前）
                    cur = self.offset + self.frames_read / STREAM_FPS
                    self._emit({"type": "stream_error",
                                "detail": "ffmpeg EOF, restarting",
                                "media_time": round(cur, 3)})
                    self._restart(cur, reason="recover", force_refresh=True)
                continue
            self.consecutive_failures = 0  # 讀到幀才恢復信任

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                STREAM_SIZE, STREAM_SIZE, 3).copy()
            self.window_buf.append(frame)
            i = self.frames_read
            self.frames_read += 1
            frame_done_t = time.monotonic()  # 幀到預測延遲的起點（最後一幀讀完）

            # 4) 推理（每 8 幀）＋ CAM 歸因（與推理同一次 forward，特徵圖免費取得）
            if len(self.window_buf) == WIN_FRAMES and i % STRIDE_FRAMES == STRIDE_FRAMES - 1:
                window = np.stack(self.window_buf)
                outs = self.sess.run(None, {self.input_name: self._preprocess(window)})
                logits = outs[0][0]
                probs = softmax(logits.astype(np.float32))
                topk_idx = np.argsort(probs)[::-1][:TOPK]
                r = self.smoother.update(probs.tolist())
                t_start = self.offset + (i + 1 - WIN_FRAMES) / STREAM_FPS
                # 診斷：解碼流落後播放頭 >3s 記一次 log（節流 5s）——
                # 這是「seek 後儀表板空白但 last 10s 有更新」的唯一成因路徑
                lag = self._playhead_estimate() - (t_start + 1.0)
                if lag > 3.0 and time.monotonic() - getattr(self, "_last_lag_log", 0) > 5:
                    self._last_lag_log = time.monotonic()
                    print(f"[lag] 解碼落後播放頭 {lag:.1f}s "
                          f"(player≈{self._playhead_estimate():.1f} pred≈{t_start + 1.0:.1f})",
                          flush=True)
                payload = {
                    "type": "prediction",
                    "media_time": round(t_start + 1.0, 3),  # 窗口中心（2s 窗）
                    "t_start": round(t_start, 3),
                    "t_end": round(t_start + WIN_FRAMES / STREAM_FPS, 3),
                    **r,
                    "probs": {CLASSES[j]: round(float(probs[j]), 4)
                              for j in range(len(CLASSES))},
                    "topk": [{"genre": CLASSES[int(j)], "prob": round(float(probs[j]), 4)}
                             for j in topk_idx],
                    "latency_ms": round((time.monotonic() - frame_done_t) * 1000, 1),
                }
                # CAM：歸因「當前顯示標籤」——熱區 = 推高該類分數的畫面區域
                if self.fc_w is not None and len(outs) > 1:
                    try:
                        feats = np.asarray(outs[1], dtype=np.float32)[0]  # (C,T,H,W)
                        h, w = feats.shape[2], feats.shape[3]
                        payload["cam_grid"] = _cam_grid(feats, self.fc_w[CLASSES.index(r["genre"])])
                        payload["cam_size"] = [h, w]
                    except Exception:
                        pass  # CAM 失敗不影響主分類
                self._emit(payload)
