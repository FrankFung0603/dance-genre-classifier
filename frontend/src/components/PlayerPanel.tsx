"use client";

/**
 * YouTube IFrame 播放器 + CAM 熱圖 overlay。
 * 時鐘/seek 偵測由 useMediaClock 共用；overlay 隨當前預測（media_time 對齊）更新。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import YouTube, { type YouTubePlayer } from "react-youtube";
import { drawCamOverlay } from "@/lib/cam";
import { useMediaClock } from "@/lib/useMediaClock";
import type { SubmitControl } from "@/lib/useDanceSession";
import type { PredictionMsg } from "@/lib/types";

interface Props {
  videoId: string;
  reportPlayhead: SubmitControl;
  pred: PredictionMsg | null; // 播放頭所在時間的預測（含 CAM grid）
  showCam: boolean;
  onToggleCam: () => void;
  accent?: string | null; // 當前預測舞種色：邊框光暈跟著預測變色
  onTime: (t: number) => void;
  onEnded?: () => void;
  onPlay?: () => void;
}

export default function PlayerPanel({ videoId, reportPlayhead, pred, showCam, onToggleCam, accent, onTime, onEnded, onPlay }: Props) {
  const playerRef = useRef<YouTubePlayer | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [muted, setMuted] = useState(false);

  useEffect(() => {
    drawCamOverlay(
      canvasRef.current, wrapRef.current,
      showCam ? pred?.cam_grid : undefined, pred?.cam_size,
    );
  }, [pred, showCam]);

  useMediaClock({
    getTime: () => {
      try {
        return playerRef.current?.getCurrentTime() ?? null;
      } catch {
        return null;
      }
    },
    getState: () => playerRef.current?.getPlayerState?.() ?? -1,
    report: reportPlayhead,
    onTime,
    onEnded,
    onPlay,
  });

  return (
    <div className="space-y-2">
      <div
        ref={wrapRef}
        className="scanline relative w-full overflow-hidden rounded-2xl bg-black ring-1 ring-white/10 transition-[box-shadow] duration-500"
        style={{
          aspectRatio: "16 / 9",
          boxShadow: accent
            ? `0 0 0 1px ${accent}55, 0 12px 56px -12px ${accent}99`
            : undefined,
        }}
      >
        <YouTube
          videoId={videoId}
          className="absolute inset-0 h-full w-full"
          iframeClassName="absolute inset-0 h-full w-full"
          opts={{ playerVars: { autoplay: 1, modestbranding: 1, rel: 0 } }}
          onReady={(e) => {
            playerRef.current = e.target;
            e.target.playVideo();
          }}
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-10" />
      </div>
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <button
          className="glass rounded-full px-3 py-1 font-medium text-zinc-300 transition-colors hover:bg-white/10"
          onClick={() => {
            const pl = playerRef.current;
            if (!pl) return;
            if (muted) pl.unMute();
            else pl.mute();
            setMuted(!muted);
          }}
        >
          {muted ? "unmute" : "mute"}
        </button>
        <button
          aria-pressed={showCam}
          className={`glass rounded-full px-3 py-1 font-medium transition-colors hover:bg-white/10 ${
            showCam ? "bg-sky-500/20 text-sky-300" : "text-zinc-400"
          }`}
          onClick={onToggleCam}
        >
          CAM {showCam ? "on" : "off"}
        </button>
        <span>熱區 = CAM 歸因：模型權重算出的高貢獻區域（支持當前標籤）</span>
      </div>
    </div>
  );
}
