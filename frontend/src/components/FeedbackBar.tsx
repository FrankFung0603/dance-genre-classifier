"use client";

/**
 * 反饋 UI（Phase 3：UI + 本地暫存；Phase 4 接 PostgreSQL）。
 * 收集：session、URL、播放頭時間、當下預測、正確類別。
 */
import { useState } from "react";
import { CLASSES, type Genre, type PredictionMsg } from "@/lib/types";

export interface FeedbackRecord {
  media_time: number;
  predicted: Genre;
  confidence: number;
  correct: Genre | null; // 👍 = null（預測正確）；👎 = 選的正確類別
  verdict: "up" | "down";
  stored?: boolean; // Phase 4：是否成功入庫（false = 後端不可用，僅本地暫存）
}

interface Props {
  pred: PredictionMsg | null;
  currentTime: number;
  onSubmit: (r: FeedbackRecord) => boolean | Promise<boolean>;
}

export default function FeedbackBar({ pred, currentTime, onSubmit }: Props) {
  const [correct, setCorrect] = useState<Genre>("hiphop");
  const [sent, setSent] = useState<{ text: string; ok: boolean } | null>(null);

  const submit = async (verdict: "up" | "down") => {
    if (!pred) return;
    const stored = await onSubmit({
      media_time: currentTime,
      predicted: pred.genre,
      confidence: pred.confidence,
      correct: verdict === "up" ? null : correct,
      verdict,
    });
    setSent(
      stored
        ? { text: verdict === "up" ? "👍 已入庫" : "👎 已入庫", ok: true }
        : { text: "後端離線，僅本地暫存", ok: false },
    );
    setTimeout(() => setSent(null), 2500);
  };

  return (
    <div className="panel flex flex-wrap items-center gap-2 rounded-2xl p-3.5 text-sm">
      <span className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--muted)" }}>
        feedback
      </span>
      <button
        className="rounded-full border border-white/10 bg-white/5 px-3.5 py-1 transition-all hover:scale-105 hover:border-emerald-400/40 hover:bg-emerald-400/10 disabled:opacity-40 disabled:hover:scale-100"
        onClick={() => submit("up")}
        disabled={!pred}
        title="預測正確"
      >
        👍
      </button>
      <button
        className="rounded-full border border-white/10 bg-white/5 px-3.5 py-1 transition-all hover:scale-105 hover:border-red-400/40 hover:bg-red-400/10 disabled:opacity-40 disabled:hover:scale-100"
        onClick={() => submit("down")}
        disabled={!pred}
        title="預測錯誤"
      >
        👎
      </button>
      <select
        className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-zinc-300 outline-none focus:border-violet-400/50"
        value={correct}
        onChange={(e) => setCorrect(e.target.value as Genre)}
      >
        {CLASSES.map((c) => (
          <option key={c} value={c} className="bg-zinc-900">
            {c}
          </option>
        ))}
      </select>
      {sent ? (
        <span
          className={`ml-auto rounded-full px-2.5 py-0.5 text-xs ${
            sent.ok
              ? "bg-emerald-400/10 text-emerald-300"
              : "bg-amber-400/10 text-amber-300"
          }`}
        >
          {sent.text}
        </span>
      ) : (
        <span className="ml-auto text-[11px]" style={{ color: "var(--faint)" }}>👎 時選正確類別</span>
      )}
    </div>
  );
}
