"""FFmpeg 幀流：持有子進程，rawvideo rgb24 逐幀輸出，支援 seek 式重啟。

不落盤、不解碼到中間檔：ffmpeg stdout 直讀 uint8 幀。
restart(offset) 殺舊進程 → -ss offset 輸入定位（phase-0 量測首幀 ~0.7s）→ 幀計數歸零。
"""
import subprocess
import sys

from server.config import STREAM_FPS, STREAM_SIZE, find_ffmpeg

FRAME_BYTES = STREAM_SIZE * STREAM_SIZE * 3


class FFmpegStream:
    def __init__(self, playable_url: str):
        self.url = playable_url
        self.proc: subprocess.Popen | None = None
        self.offset_s = 0.0  # 當前串流的媒體時間起點

    def _spawn(self, offset_s: float) -> None:
        cmd = [
            find_ffmpeg(), "-hide_banner", "-loglevel", "error",
            # rw_timeout：網路讀寫超時（µs）。沒有它，來源卡住時 stdout.read 會無限阻塞，
            # worker 看似存活實則僵死（前端表現 = seek 後永遠轉圈）。
            "-rw_timeout", "10000000",
            "-ss", f"{max(0.0, offset_s):.3f}", "-i", self.url,
            "-vf", f"fps={STREAM_FPS},scale={STREAM_SIZE}:{STREAM_SIZE}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7
        )
        self.offset_s = max(0.0, offset_s)

    def restart(self, offset_s: float) -> None:
        self.stop()
        self._spawn(offset_s)

    def read_frame(self) -> bytes | None:
        """讀一幀；EOF/串流失敗回 None。"""
        if self.proc is None or self.proc.stdout is None:
            return None
        buf = self.proc.stdout.read(FRAME_BYTES)
        if not buf or len(buf) < FRAME_BYTES:
            return None
        return buf

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=3)  # 收屍：避免 Windows 上累積殭屍 handle
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.proc = None

    def __del__(self):
        self.stop()
