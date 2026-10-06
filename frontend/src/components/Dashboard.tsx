"use client";

/**
 * Dashboard：當前舞種、信心 bar、最近 10 秒 history、top-3。
 * 資料來源 = 按 media_time 對齊後的預測（getPredictionAt(t)），非「最新一筆」。
 */
import { CLASSES, type Genre, type PredictionMsg } from "@/lib/types";

const GENRE_COLORS: Record<Genre, string> = {
  ballet: "#a78bfa",
  breakdance: "#fb923c",
  contemporary: "#34d399",
  hiphop: "#38bdf8",
  salsa: "#f472b6",
  unknown: "#fde047", // 亮黃：深色背景上清楚可見（原灰太難看清）
};

interface Props {
  pred: PredictionMsg | null;
  history: PredictionMsg[];
  latencyMs: number | null;
  starting?: boolean;
  syncing?: boolean;
  lagS?: number | null; // 顯示中的預測離播放頭幾秒（null = 無預測）
}

export default function Dashboard({ pred, history, latencyMs, starting, syncing, lagS }: Props) {
  const conf = pred?.confidence ?? 0;
  const color = pred ? GENRE_COLORS[pred.genre] : "#52525b";
  const lag = lagS ?? 0;
  // >2.5s 的預測是「別的時間點的答案」，顯示出來反而誤導 → 誠實顯示追趕狀態
  const showStale = (!pred && history.length > 0) || (!!pred && lag > 2.5);
  const staleChip = !!pred && !showStale && lag > 1.5;
  return (
    <div className="panel space-y-5 rounded-2xl p-5">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--muted)" }}>
          Current genre
        </div>
        <div className="mt-1.5 flex items-center gap-3">
          {pred && !showStale && (
            <span className="eq" style={{ color }}>
              <span />
              <span />
              <span />
            </span>
          )}
          <span
            className="text-4xl font-bold tracking-[-0.02em]"
            style={{ color: showStale ? "#a1a1aa" : color, transition: "color 0.4s ease" }}
          >
            {showStale ? "預測追趕中…" : pred?.genre ?? "—"}
          </span>
          {pred && !showStale && (
            <span className="flex flex-col text-[11px] leading-tight" style={{ color: "var(--muted)" }}>
              <span className="tabular-nums text-zinc-300">{(conf * 100).toFixed(0)}%</span>
              <span>raw: {pred.genre_raw}</span>
            </span>
          )}
          {starting && (
            <span className="animate-pulse text-xs text-sky-400">模型啟動中…</span>
          )}
          {!starting && syncing && (
            <span className="animate-pulse text-xs text-amber-400">seek 重新同步中…</span>
          )}
          {!starting && !syncing && staleChip && (
            <span className="rounded-full bg-amber-400/10 px-2 py-0.5 text-[10px] text-amber-300">
              預測追趕中（+{lag.toFixed(1)}s）
            </span>
          )}
        </div>
      </div>

      <div>
        <div className="mb-1.5 flex justify-between text-[11px]" style={{ color: "var(--muted)" }}>
          <span className="uppercase tracking-[0.18em]">confidence</span>
          <span className="tabular-nums">{pred ? `${(conf * 100).toFixed(0)}%` : ""}</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${showStale ? 0 : conf * 100}%`,
              background: pred ? color : "#3f3f46",
            }}
          />
        </div>
      </div>

      {pred && !showStale && (
        <div className="space-y-1.5">
          {pred.topk.map((k) => (
            <div key={k.genre} className="flex items-center gap-2 text-xs">
              <span className="w-24 truncate text-zinc-300">{k.genre}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${k.prob * 100}%`,
                    background: GENRE_COLORS[k.genre],
                  }}
                />
              </div>
              <span className="w-10 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                {(k.prob * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      <div>
        <div className="mb-1.5 text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--muted)" }}>
          last 10s
        </div>
        <div className="flex h-5 gap-[3px] overflow-hidden rounded-lg bg-white/[0.04] p-1">
          {history.length === 0 && (
            <div className="flex-1 rounded bg-white/5" />
          )}
          {history.map((p, i) => (
            <div
              key={`${p.media_time}-${i}`}
              title={`${p.media_time.toFixed(1)}s ${p.genre} ${(p.confidence * 100).toFixed(0)}%`}
              className="flex-1 rounded-sm transition-all duration-300"
              style={{
                background: GENRE_COLORS[p.genre],
                opacity: 0.35 + p.confidence * 0.65,
              }}
            />
          ))}
        </div>
      </div>

      <div
        className="flex justify-between border-t border-white/5 pt-3 font-mono text-[11px]"
        style={{ color: "var(--faint)" }}
      >
        <span>inference {latencyMs != null ? `${latencyMs.toFixed(0)} ms/clip` : "—"}</span>
        <span>{pred ? `@ ${pred.media_time.toFixed(1)}s` : ""}</span>
      </div>
    </div>
  );
}

export { GENRE_COLORS, CLASSES };
