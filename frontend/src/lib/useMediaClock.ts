"use client";

/**
 * 播放頭時鐘（YouTube / MP4 兩種 player 共用）：
 * 每 250ms 取時間 → 心跳回報 + pause/resume/seek 偵測（狀態變化立即送）。
 *
 * 重要：buffering（YouTube state 3）不可視為 pause——拖進度條時 seek 之後必接 buffering，
 * 若把 buffering 當 pause 送出，後端會殺串流 → resume 又重啟 → debounce seek 再重啟，
 * 一次 seek 開三次 ffmpeg（每次都有首幀延遲）。只對 1↔2 的乾淨轉換送 pause/resume。
 */
import { useEffect, useRef } from "react";
import type { SubmitControl } from "./useDanceSession";

const REPORT_INTERVAL_MS = 250;
const SEEK_JUMP_S = 1.2;

interface ClockOpts {
  getTime: () => number | null;
  getState: () => number; // YouTube player state（MP4：1 播放 / 2 暫停 即可）
  report: SubmitControl;
  onTime: (t: number) => void;
  onEnded?: () => void; // 播放自然結束（state 0）
  onPlay?: () => void; // 從暫停/結束恢復播放
}

export function useMediaClock(opts: ClockOpts) {
  const refs = useRef(opts);
  refs.current = opts;
  const prev = useRef({ t: -1, state: -1, ts: 0 });
  const playingSeen = useRef(false);
  // 已送出 pause 且尚未送出 resume——resume 的偵測不能只認 2→1：
  // 暫停久了之後按播放，影片 buffer 已被逐出，YouTube 會走 2→3（buffering）→1，
  // 只認 2→1 會永遠漏掉 resume → backend worker 卡在 paused、從此沒有預測。
  const pauseOutstanding = useRef(false);

  useEffect(() => {
    const iv = setInterval(() => {
      const { getTime: gt, getState: gs, report: rep, onTime: cb,
              onEnded, onPlay } = refs.current;
      const t = gt();
      if (t == null) return;
      const now = Date.now();
      const st = gs();
      if (st === 1) playingSeen.current = true;
      const { t: prevT, state: prevSt, ts: prevTs } = prev.current;

      if (prevT >= 0) {
        if (st === 2 && (prevSt === 1 || prevSt === 3)) {
          // 1→2 乾淨暫停；3→2 = 播放前 buffering 中用戶又暫停（pause 冪等，重送無害）
          rep({ type: "pause" });
          pauseOutstanding.current = true;
        } else if (st === 1 && pauseOutstanding.current && prevSt !== 0 && prevSt !== -1) {
          // 恢復播放：涵蓋 2→1 與 2→3→1 兩種路徑（0/-1 是初始載入，不送 resume）
          rep({ type: "resume", media_time: t });
          pauseOutstanding.current = false;
          onPlay?.();
        } else if (st === 1 && (prevSt === 0 || prevSt === -1)) {
          onPlay?.();
        } else if (st === 0 && prevSt === 1) {
          rep({ type: "pause" }); // 播放自然結束：乾淨收工，避免後端把 EOF 誤報為錯誤
          pauseOutstanding.current = false;
          onEnded?.();
        } else if (
          // seek：時間跳躍在 buffering（3）期間也要抓——拖進度條是 1→3→1，
          // 只認 1→1 會永遠漏掉。playingSeen 避免初始載入的 0→起點跳躍誤判。
          playingSeen.current &&
          (st === 1 || st === 3) && (prevSt === 1 || prevSt === 3) &&
          Math.abs(t - prevT) > SEEK_JUMP_S
        ) {
          rep({ type: "seek", media_time: t });
        }
      }
      prev.current = { t, state: st, ts: now };
      rep({ type: "playhead", media_time: t });
      cb(t);
    }, REPORT_INTERVAL_MS);
    return () => clearInterval(iv);
  }, []);
}
