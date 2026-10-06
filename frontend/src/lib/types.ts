export const CLASSES = [
  "ballet",
  "breakdance",
  "contemporary",
  "hiphop",
  "salsa",
  "unknown",
] as const;

export type Genre = (typeof CLASSES)[number];

export interface PredictionMsg {
  type: "prediction";
  media_time: number;
  t_start: number;
  t_end: number;
  genre: Genre;
  confidence: number;
  known_confidence: number;
  genre_raw: Genre;
  probs: Record<Genre, number>;
  topk: { genre: Genre; prob: number }[];
  latency_ms: number;
  /** CAM 歸因熱圖：H×W grid（row-major，[0,1]）；size = [H, W]。後端缺權重檔時缺省 */
  cam_grid?: number[];
  cam_size?: [number, number];
}

export interface StateMsg {
  type: "state";
  playing: boolean;
  media_time: number;
  reason?: string;
}

export interface ReadyMsg {
  type: "ready";
  classes: Genre[];
  stream: { fps: number; win_frames: number };
}

export type ServerMsg =
  | ReadyMsg
  | PredictionMsg
  | StateMsg
  | { type: "stream_started"; media_time: number }
  | { type: "stream_error"; detail: string; media_time: number }
  | { type: "fatal"; detail: string };

export interface SessionInfo {
  session_id: string;
  kind: "youtube" | "mp4" | "file";
  ws_url: string;
  classes: Genre[];
}
