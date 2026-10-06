"use client";

/**
 * 播放結束總結卡：整場預測的 genre 票數結算（每秒一票），
 * 非 unknown 票 ≥ 30% 時給出主判，否則判定以 Unknown 為主。
 */
import { GENRE_COLORS } from "./Dashboard";
import { CLASSES, type Genre } from "@/lib/types";

export interface SessionSummary {
  counts: Record<string, number>;
  total: number;
  dominant: string | null;
  dominantShare: number;
}

export default function SummaryCard({
  summary,
  onDismiss,
}: {
  summary: SessionSummary;
  onDismiss: () => void;
}) {
  const rows = CLASSES.map((g) => ({
    genre: g as Genre,
    count: summary.counts[g] ?? 0,
    share: summary.total > 0 ? (summary.counts[g] ?? 0) / summary.total : 0,
  })).sort((a, b) => b.count - a.count);
  const domColor = summary.dominant
    ? GENRE_COLORS[summary.dominant as Genre]
    : "#fde047";

  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="panel w-full max-w-md space-y-4 rounded-2xl bg-[#0d0f1a] p-6 shadow-[0_24px_64px_-16px_rgba(0,0,0,0.8)]">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold tracking-[-0.02em]">影片總結</h2>
          <span className="rounded-full bg-white/5 px-2.5 py-0.5 font-mono text-[11px] text-zinc-400">
            {summary.total} 秒預測結算
          </span>
        </div>

        <div className="rounded-xl bg-white/[0.04] p-4 text-center">
          <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--muted)" }}>
            主判舞蹈類別
          </div>
          <div
            className="mt-1 text-3xl font-bold tracking-[-0.02em]"
            style={{ color: domColor }}
          >
            {summary.dominant ?? "Unknown / 非舞蹈"}
          </div>
          <div className="mt-0.5 text-sm text-zinc-300">
            {summary.dominant
              ? `佔全片 ${(summary.dominantShare * 100).toFixed(0)}% 的預測`
              : "多數時間非舞蹈內容"}
          </div>
        </div>

        <div className="space-y-1.5">
          {rows.map((r) => (
            <div key={r.genre} className="flex items-center gap-2 text-xs">
              <span className="w-24 truncate text-zinc-300">{r.genre}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${r.share * 100}%`,
                    background: GENRE_COLORS[r.genre],
                  }}
                />
              </div>
              <span className="w-10 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                {(r.share * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>

        <div className="text-right">
          <button
            className="rounded-xl bg-sky-500 px-5 py-1.5 text-sm font-semibold text-[#06121d] transition-colors hover:bg-sky-400"
            onClick={onDismiss}
          >
            關閉
          </button>
        </div>
      </div>
    </div>
  );
}
