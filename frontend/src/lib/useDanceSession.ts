"use client";

/**
 * 核心同步機制：預測（含 CAM 歸因熱圖）由後端按「媒體時間」串流，前端 buffer
 * 存放，播放頭走到該時間才顯示。播放與預測是兩條流，靠 media_time 對齊。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type PredictionMsg,
  type ServerMsg,
  type SessionInfo,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
// same-origin 部署（nginx 反代，NEXT_PUBLIC_API_BASE=""）時 WebSocket 需絕對 URL
const WS_BASE = API_BASE === ""
  ? `${typeof window !== "undefined" && window.location.protocol === "https:" ? "wss" : "ws"}://${typeof window !== "undefined" ? window.location.host : ""}`
  : API_BASE.replace(/^http/, "ws");
const BUFFER_TTL_S = 30; // buffer 只留播放頭前後這個範圍

export type Phase = "idle" | "starting" | "live" | "error";

export interface DanceSessionState {
  phase: Phase;
  error: string | null;
  playing: boolean;
  syncing: boolean; // seek 後等新串流的第一個預測期間 = true
  latest: PredictionMsg | null; // 最近一次預測（raw，供 debug 面板）
  latencyMs: number | null;
}

export interface SubmitControl {
  (msg:
    | { type: "playhead"; media_time: number }
    | { type: "pause" }
    | { type: "resume"; media_time: number }
    | { type: "seek"; media_time: number }): void;
}

export function useDanceSession() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [latest, setLatest] = useState<PredictionMsg | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionRef = useRef<SessionInfo | null>(null);
  const urlRef = useRef<string>("");
  // media_time → 預測（key 取 0.1s 解析度；值域 [t_start, t_end)）
  const predBuf = useRef<PredictionMsg[]>([]);
  const playheadRef = useRef(0);
  // 全場計票（不隨 buffer TTL 清理）：每個「媒體秒數」一票，重看同一秒以最後一次為準
  // → 直線播完 = 全片計票；seek 跳過的段落沒票；倒轉重看不重複灌票
  const voteBySecond = useRef<Map<number, string>>(new Map());

  const start = useCallback(async (url: string) => {
    setPhase("starting");
    setError(null);
    urlRef.current = url;
    predBuf.current = [];
    voteBySecond.current = new Map();
    try {
      const res = await fetch(`${API_BASE}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail ?? `HTTP ${res.status}`);
      }
      const info: SessionInfo = await res.json();
      sessionRef.current = info;

      const ws = new WebSocket(`${WS_BASE}${info.ws_url}`);
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        const m: ServerMsg = JSON.parse(ev.data);
        if (m.type === "prediction") {
          predBuf.current.push(m);
          if (predBuf.current.length > 600) predBuf.current.splice(0, 200);
          voteBySecond.current.set(Math.round(m.media_time), m.genre);
          setLatest(m);
          setLatencyMs(m.latency_ms);
          setSyncing(false);
        } else if (m.type === "state") {
          setPlaying(m.playing);
        } else if (m.type === "fatal") {
          setPhase("error");
          setError(m.detail);
        }
      };
      ws.onopen = () => setPhase("live");
      ws.onclose = (ev) => {
        if (phase !== "error") {
          setPhase((p) => (p === "error" ? p : ev.wasClean ? "idle" : "error"));
          if (!ev.wasClean) setError("WebSocket 連線中斷");
        }
      };
      ws.onerror = () => setError("WebSocket 錯誤");
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : String(e));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setPhase("idle");
    setLatest(null);
    setPlaying(false);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  /** 播放頭更新（Player 每 250ms 呼叫）：推給後端 + 清舊 buffer */
  const reportPlayhead = useCallback<SubmitControl>((msg) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (msg.type === "playhead") {
      playheadRef.current = msg.media_time;
      const t = msg.media_time;
      predBuf.current = predBuf.current.filter((p) => p.t_end > t - BUFFER_TTL_S);
    } else if (msg.type === "seek") {
      setSyncing(true); // 新串流的第一個預測抵達時清除
    }
    ws.send(JSON.stringify(msg));
  }, []);

  /** 播放頭所在時間的預測（[t_start, t_end) 命中；否則退最近中心 ±2.5s）。
   *  fallback 從 0.7s 放寬到 2.5s：seek 後解碼流要追趕播放頭，慢網路下會落後
   *  1–2.5s，與其讓儀表板空白，顯示最新已知預測（Dashboard 會標「追趕中」）。 */
  const getPredictionAt = useCallback((t: number): PredictionMsg | null => {
    const buf = predBuf.current;
    for (let i = buf.length - 1; i >= 0; i--) {
      const p = buf[i];
      if (t >= p.t_start && t < p.t_end) return p;
    }
    let best: PredictionMsg | null = null;
    let bestD = 2.5;
    for (const p of buf) {
      const d = Math.abs(p.media_time - t);
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }, []);

  /** 最近 10 秒歷史（依 media_time 升冪） */
  const getHistory = useCallback((t: number): PredictionMsg[] => {
    return predBuf.current.filter((p) => p.media_time >= t - 10 && p.media_time <= t + 1);
  }, []);

  /** 播放結束總結：全片每秒一票（seek 跳過不計、重看不重複），非 unknown ≥30% 給主判 */
  const getSessionSummary = useCallback((): {
    counts: Record<string, number>;
    total: number;
    dominant: string | null;
    dominantShare: number;
  } => {
    const counts: Record<string, number> = {};
    for (const g of voteBySecond.current.values()) {
      counts[g] = (counts[g] ?? 0) + 1;
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const known = Object.entries(counts).filter(([g]) => g !== "unknown");
    const knownTotal = known.reduce((a, [, c]) => a + c, 0);
    let dominant: string | null = null;
    let dominantShare = 0;
    if (known.length > 0 && knownTotal >= total * 0.3) {
      const [g, c] = known.reduce((a, b) => (b[1] > a[1] ? b : a));
      dominant = g;
      dominantShare = total > 0 ? c / total : 0;
    }
    return { counts, total, dominant, dominantShare };
  }, []);

  /** Phase 4：反饋入庫（PostgreSQL）。帶 session 資訊 POST /feedback；回傳是否成功入庫。 */
  const submitFeedback = useCallback(async (r: {
    media_time: number;
    predicted: string;
    confidence: number;
    correct: string | null;
    verdict: "up" | "down";
  }): Promise<boolean> => {
    const s = sessionRef.current;
    if (!s) return false;
    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: s.session_id, url: urlRef.current, kind: s.kind, ...r,
        }),
      });
      return res.ok;
    } catch {
      return false;
    }
  }, []);

  return {
    state: { phase, error, playing, syncing, latest, latencyMs } satisfies DanceSessionState,
    session: sessionRef.current,
    start,
    stop,
    reportPlayhead,
    getPredictionAt,
    getHistory,
    getSessionSummary,
    submitFeedback,
  };
}
