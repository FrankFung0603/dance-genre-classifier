"use client";

import { useCallback, useMemo, useState } from "react";
import Dashboard, { GENRE_COLORS } from "@/components/Dashboard";
import FeedbackBar, { type FeedbackRecord } from "@/components/FeedbackBar";
import Mp4Panel from "@/components/Mp4Panel";
import PlayerPanel from "@/components/PlayerPanel";
import SummaryCard, { type SessionSummary } from "@/components/SummaryCard";
import { useDanceSession } from "@/lib/useDanceSession";

const YT_RE = /(?:youtube\.com\/watch\?.*v=|youtu\.be\/|youtube\.com\/shorts\/)([\w-]{11})/;

export default function Home() {
  const [input, setInput] = useState("");
  const [committedUrl, setCommittedUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [feedbackLog, setFeedbackLog] = useState<FeedbackRecord[]>([]);
  const [showCam, setShowCam] = useState(true); // CAM 歸因熱圖開關
  const {
    state, session, start, stop, reportPlayhead, getPredictionAt, getHistory,
    getSessionSummary, submitFeedback,
  } = useDanceSession();
  const [summary, setSummary] = useState<SessionSummary | null>(null);

  // 播放器與 session 建立並行：URL 一送出就播影片，AI 在背景啟動（省 ~1.6s yt-dlp 解析的體感等待）
  const videoId = useMemo(() => committedUrl?.match(YT_RE)?.[1] ?? null, [committedUrl]);
  const isMp4 = useMemo(
    () => /^https?:\/\/\S+\.(mp4|mov|webm|mkv)(\?\S*)?$/i.test((committedUrl ?? "").trim()),
    [committedUrl],
  );
  const live = committedUrl !== null && (state.phase === "live" || state.phase === "starting");

  const pred = live ? getPredictionAt(currentTime) : null;
  const history = live ? getHistory(currentTime) : [];
  const lagS = pred ? Math.abs(pred.media_time - currentTime) : null;

  const onSubmitFeedback = useCallback(async (r: FeedbackRecord) => {
    // Phase 4：POST /feedback 入庫；後端不可用時本地暫存（demo 降級）
    const stored = await submitFeedback(r);
    setFeedbackLog((l) => [...l, { ...r, stored }]);
    return stored;
  }, [submitFeedback]);

  const doStop = useCallback(() => {
    stop();
    setCommittedUrl(null);
    setSummary(null);
    setInput("");
  }, [stop]);

  const onEnded = useCallback(() => setSummary(getSessionSummary()), [getSessionSummary]);
  const onPlay = useCallback(() => setSummary(null), []);

  const accent = live && pred ? GENRE_COLORS[pred.genre] : null;

  return (
    <main className="mx-auto max-w-6xl flex-1 space-y-5 p-6 text-zinc-100">
      <header className="flex items-end justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2.5">
            {/* 產品標記：頻譜脈衝（描繪的 SVG，非字符 icon） */}
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
              <rect x="2" y="9" width="2.6" height="4" rx="1.3" fill="#38bdf8" />
              <rect x="6.8" y="5" width="2.6" height="12" rx="1.3" fill="#38bdf8" opacity="0.75" />
              <rect x="11.6" y="2" width="2.6" height="18" rx="1.3" fill="#38bdf8" opacity="0.5" />
              <rect x="16.4" y="7" width="2.6" height="8" rx="1.3" fill="#38bdf8" opacity="0.75" />
            </svg>
            <h1 className="text-2xl font-bold tracking-[-0.02em]">Dance·AI</h1>
            <span className="text-sm" style={{ color: "var(--muted)" }}>即時舞種分類引擎</span>
          </div>
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            YouTube 貼上 → 即時舞種分類（ballet / breakdance / contemporary / hiphop / salsa + Unknown）
          </p>
        </div>
        <div className="flex items-center gap-2">
          {live ? (
            <span className="panel flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium text-emerald-300">
              <span className="live-dot" />
              LIVE · {state.phase}
            </span>
          ) : (
            <span className="panel rounded-full px-3 py-1 text-xs text-zinc-400">
              {state.phase}
            </span>
          )}
          {session && (
            <span className="panel rounded-full px-3 py-1 font-mono text-[11px] text-zinc-400">
              {session.session_id} · {session.kind}
            </span>
          )}
        </div>
      </header>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (input.trim()) {
            setCommittedUrl(input.trim());
            void start(input.trim());
          }
        }}
      >
        <div className="panel flex flex-1 items-center rounded-xl px-3 transition-colors focus-within:border-sky-500/60">
          <input
            className="w-full bg-transparent py-2.5 text-sm outline-none placeholder:text-zinc-500"
            placeholder="貼上 YouTube 連結，或 direct MP4 URL…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={live}
          />
        </div>
        {live ? (
          <button
            type="button"
            className="panel rounded-xl px-5 text-sm font-medium text-zinc-200 transition-colors hover:bg-white/10"
            onClick={doStop}
          >
            stop
          </button>
        ) : (
          <button
            type="submit"
            className="rounded-xl bg-sky-500 px-6 text-sm font-semibold text-[#06121d] transition-colors hover:bg-sky-400 disabled:opacity-40"
            disabled={!input.trim()}
          >
            start →
          </button>
        )}
      </form>

      {state.error && (
        <div className="panel rounded-xl border-red-500/30 bg-[#1d0d12] px-4 py-2.5 text-sm text-red-300">
          {state.error}
        </div>
      )}

      {live && (videoId || isMp4) && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[2fr_1fr]">
          {videoId ? (
            <PlayerPanel
              videoId={videoId}
              reportPlayhead={reportPlayhead}
              pred={pred}
              showCam={showCam}
              onToggleCam={() => setShowCam((s) => !s)}
              accent={accent}
              onTime={setCurrentTime}
              onEnded={onEnded}
              onPlay={onPlay}
            />
          ) : (
            <Mp4Panel
              src={input.trim()}
              report={reportPlayhead}
              pred={pred}
              showCam={showCam}
              onToggleCam={() => setShowCam((s) => !s)}
              accent={accent}
              onTime={setCurrentTime}
              onEnded={onEnded}
              onPlay={onPlay}
            />
          )}
          {summary && (
            <div className="fixed inset-0 z-50">
              <SummaryCard summary={summary} onDismiss={() => setSummary(null)} />
            </div>
          )}
          <div className="space-y-4">
            <Dashboard
              pred={pred}
              history={history}
              latencyMs={state.latencyMs}
              starting={state.phase === "starting"}
              syncing={state.syncing}
              lagS={lagS}
            />
            <FeedbackBar pred={pred} currentTime={currentTime} onSubmit={onSubmitFeedback} />
          </div>
        </div>
      )}

      {live && !videoId && !isMp4 && (
        <div className="panel rounded-xl px-4 py-3 text-sm text-amber-300/90">
          無法從輸入辨識 YouTube video ID 或 direct 影片 URL。
        </div>
      )}

      {feedbackLog.length > 0 && (
        <div className="font-mono text-[11px]" style={{ color: "var(--faint)" }}>
          反饋 {feedbackLog.length} 筆（入庫 {feedbackLog.filter((f) => f.stored).length}）：
          最新{" "}
          {JSON.stringify(feedbackLog[feedbackLog.length - 1])}
        </div>
      )}
    </main>
  );
}
