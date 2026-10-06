# Dance Genre Classifier

Paste a YouTube URL (or a direct MP4 URL) → the video plays in the browser → the backend streams frames in real time → a spatiotemporal model classifies the dance genre **every second**, overlaid on the video with a **live CAM attribution heatmap** (where the model looks, toggleable) and an honest **Unknown** fallback. 👍/👎 feedback is persisted to PostgreSQL.

Five genres: Ballet / Breakdance / Salsa / Hiphop / Contemporary + **Unknown**.

> 中文說明文件：[README.zh.md](README.zh.md)
> Analysis & numbers (architecture rationale, dataset, training, benchmarks, scaling design): [docs/report.md](docs/report.md) · all docs: [docs/](docs/)

---

## Quick start (Docker)

Prerequisite: Docker Desktop running (WSL2 backend on Windows). GPU optional.

```bash
docker compose up --build
# → open http://localhost:8080
```

Five containers behind nginx :8080 — frontend (Next.js), backend (FastAPI + in-proc inference), redis (control messages), postgres (feedback). Model files are bind-mounted from `./models` — swapping a model needs no rebuild.

**GPU / CPU:** `gpus: all` in the backend service → CUDA EP. On a CPU-only host, delete that line — the same image falls back to CPU automatically (provider auto-detect: CUDA > DirectML > CPU).

**Verify:**

```bash
curl http://localhost:8080/healthz
# {"ok":true,"ffmpeg":true,"model":true,"db":true,"redis":true,"active_sessions":0}
```

## Using it

1. **Paste a YouTube dance URL** (or a direct MP4 — the deterministic fallback that YouTube anti-bot can't break)
2. Predictions stream onto the overlay at 1 Hz; the first one takes ~2–3 s (a 2 s window must fill first — physics, not lag)
3. **CAM on/off** toggles the attribution heatmap — red/yellow = where the model is looking for the currently displayed label
4. **Drag the progress bar** freely — the backend re-streams from the new position (debounced 500 ms)
5. **👍/👎** logs wrong predictions into PostgreSQL (even if the DB is down — the UI then says "stored locally"); when the video ends, a summary card tallies the whole-video vote

## Local development (without Docker)

- **Backend**: Python 3.12 venv (`requirements.txt`; Windows uses `onnxruntime-directml`), ffmpeg via `FFMPEG_BIN` env or unzipped under `tools/ffmpeg-<version>/bin/` (binaries not in this repo — download the essentials build from gyan.dev), then
  `.venv\Scripts\python.exe -m uvicorn server.main:app --port 8000`
- **Frontend**: `cd frontend && npm install && npm run dev` → http://localhost:3000 (talks to the backend at :8000 by default)
- **Feedback DB**: any Postgres reachable at `DATABASE_URL` — without one, everything still runs; `/feedback` returns 503 and the UI falls back to local storage

## Architecture

```
┌────────────────────────────── Browser ──────────────────────────────┐
│  YouTube IFrame player (react-youtube)      Next.js 16 + TS + Tailwind │
│  playhead every 250ms ──┐                   prediction buffer keyed   │
│  pause/seek events      │                   by media_time → Canvas    │
│                         ▼                   CAM heatmap + dashboard  │
└─────────────────────────┼──────────────────────────▲──────────────────┘
                          │ HTTP + WebSocket         │ predictions {media_time,
                          ▼                          │ genre, conf, CAM grid}
┌────────────────────── nginx :8080 ────────────────┴──────────────────┐
│        same-origin reverse proxy + WS upgrade (/ws/* → backend)      │
└──────────┬──────────────────────────────────────────────┬────────────┘
           ▼                                              ▼
┌─────────────────────────────┐              ┌────────────────────────┐
│ frontend (Next.js :3000)    │              │ backend (FastAPI :8000)│
└─────────────────────────────┘              │  POST /session         │
                                             │  WS   /ws/{sid}        │
                                             │  POST/GET /feedback    │
                                             │  InferenceWorker:      │
                                             │   yt-dlp → direct URL  │
                                             │   ffmpeg pipe (raw RGB │
                                             │   224², 8fps, no disk) │
                                             │   ONNX R(2+1)D-18 1Hz  │
                                             │   CAM heatmap / pred   │
                                             └───────┬───────┬────────┘
                                                     │       │
                                        control msgs │       │ feedback
                                        (in-proc,    ▼       ▼
                                         Redis-ready)  ┌──────────┐ ┌──────────┐
                                          ┌──────────┐ │postgres16│ │ redis:7  │
                                          │ models/  │ └──────────┘ └──────────┘
                                          │ (volume) │   👍/👎 rows    control bus
                                          └──────────┘
```

Why it's built this way (two streams + `media_time`, frames never cross the bus, CAM without backprop, seek semantics): [docs/report.md](docs/report.md) · decision-by-decision: [docs/decisions.md](docs/decisions.md)

## Results & documentation

Everything analytical lives in the docs — deliberately not duplicated here:

- **[docs/report.md](docs/report.md)** — technical report: F1 history (0.619 → 0.823 → 0.879, group-split held-out), per-class P/R, confusion matrices, latency measurements, dataset curation, training methodology, 1 → 100 scaling design
- **[docs/limitations.md](docs/limitations.md)** — known limitations, honestly quantified where measurable
- **`training/*.log`** — raw training logs · **`docs/*.json`** — metrics & E2E results

## Repo layout

```
frontend/          Next.js app (player, CAM overlay, dashboard, feedback UI)
server/            FastAPI backend + InferenceWorker + Dockerfile
nginx/             reverse proxy config (WS upgrade)
models/            ONNX models (dual-output: logits + features) + CAM FC weights (bind-mounted at runtime)
training/          train/export scripts + training logs
data/              dataset pipeline (manifest, window extraction, splits)
docs/              technical report, phase reports, decisions, limitations, metrics
spikes/            risk-spike scripts & E2E tests
```
