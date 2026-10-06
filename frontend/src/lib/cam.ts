/**
 * CAM 歸因熱圖繪製：後端給的 H×W grid（[0,1]）→ jet 色階 + 依強度透明
 * → 畫到離屏 canvas → 平滑放大疊在播放器全區。
 */
export function drawCamOverlay(
  canvas: HTMLCanvasElement | null,
  wrap: HTMLElement | null,
  grid: number[] | undefined,
  size: [number, number] | undefined,
): void {
  if (!canvas || !wrap) return;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  if (!grid || !size || grid.length !== size[0] * size[1]) return;

  const [rows, cols] = size;
  const off = document.createElement("canvas");
  off.width = cols;
  off.height = rows;
  const octx = off.getContext("2d");
  if (!octx) return;
  const img = octx.createImageData(cols, rows);
  for (let i = 0; i < grid.length; i++) {
    const v = Math.min(1, Math.max(0, grid[i]));
    const [r, g, b] = jet(v);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = Math.round(v * 165); // 峰值不透明度 ~0.65
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(off, 0, 0, w, h);
}

/** jet 色階：0=深藍 → 0.5=綠 → 1=紅 */
function jet(v: number): [number, number, number] {
  const f = (t: number) =>
    Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - t))));
  return [f(3), f(2), f(1)];
}
