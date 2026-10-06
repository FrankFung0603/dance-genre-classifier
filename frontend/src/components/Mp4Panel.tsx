"use client";

/**
 * Direct MP4 fallback 播放器（同一條後端 pipeline；YouTube 被反爬擋下時的保命路徑）。
 */
import { useEffect, useRef, useState } from "react";
import { drawCamOverlay } from "@/lib/cam";
import { useMediaClock } from "@/lib/useMediaClock";
import type { SubmitControl } from "@/lib/useDanceSession";
import type { PredictionMsg } from "@/lib/types";

interface Props {
  src: string;
  report: SubmitControl;
  pred: PredictionMsg | null; // 播放頭所在時間的預測（含 CAM grid）
  showCam: boolean;
  onToggleCam: () => void;
  accent?: string | null; // 當前預測舞種色：邊框光暈跟著預測變色
  onTime: (t: number) => void;
  onEnded?: () => void;
  onPlay?: () => void;
}

export default function Mp4Panel({ src, report, pred, showCam, onToggleCam, accent, onTime, onEnded, onPlay }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [muted, setMuted] = useState(true);

  useEffect(() => {
    drawCamOverlay(
      canvasRef.current, videoRef.current,
      showCam ? pred?.cam_grid : undefined, pred?.cam_size,
    );
  }, [pred, showCam]);

  useMediaClock({
    getTime: () => {
      const v = videoRef.current;
      return v && v.readyState >= 1 ? v.currentTime : null;
    },
    getState: () => {
      const v = videoRef.current;
      if (!v) return -1;
      return v.paused ? 2 : v.ended ? 0 : 1;
    },
    report,
    onTime,
    onEnded,
    onPlay,
  });

  return (
    <div className="space-y-2">
      <div
        className="scanline relative w-full overflow-hidden rounded-2xl bg-black ring-1 ring-white/10 transition-[box-shadow] duration-500"
        style={{
          boxShadow: accent
            ? `0 0 0 1px ${accent}55, 0 12px 56px -12px ${accent}99`
            : undefined,
        }}
      >
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video
          ref={videoRef}
          src={src}
          className="w-full"
          autoPlay
          muted={muted}
          controls
          playsInline
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-10" />
      </div>
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <button
          aria-pressed={showCam}
          className={`glass rounded-full px-3 py-1 font-medium transition-colors hover:bg-white/10 ${
            showCam ? "bg-sky-500/20 text-sky-300" : "text-zinc-400"
          }`}
          onClick={onToggleCam}
        >
          CAM {showCam ? "on" : "off"}
        </button>
        <span>Direct MP4 fallback。熱區 = CAM 歸因：模型權重算出的高貢獻區域。</span>
      </div>
    </div>
  );
}
