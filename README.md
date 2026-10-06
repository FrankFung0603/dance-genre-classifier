# Dance Genre Classifier

Paste a YouTube URL (or a direct MP4 URL) → the video plays in the browser → a streaming backend extracts frames in real time → a spatiotemporal model classifies the dance genre **every second** → the prediction is displayed as an overlay, time-aligned with playback. A **real-time CAM attribution heatmap** (where the model looks, user-toggleable) rides on every prediction, and a 👍/👎 feedback loop is persisted to PostgreSQL.

**Classes:** Ballet / Breakdance / Salsa / Hiphop / Contemporary + **Unknown** fallback

**Headline result (held-out test, group split by source video):** macro-F1 **0.619 → 0.823 → 0.879** across three training rounds (v3 vs v2 on the same test set). Unknown recall **1.000** — the system says "unknown" rather than guessing.

> 中文說明文件：[README.zh.md](README.zh.md) Technical report: docs/report.md 

---

## Architecture

```
┌────────────────────────────── Browser ──────────────────────────────┐
│  YouTube IFrame player (react-youtube)      Next.js 16 + TS + Tailwind │
│  playhead every 250ms ──┐                   prediction buffer keyed   │
│  pause/seek events      │                   by media_time → Canvas    │
│                         ▼                   overlay + dashboard      │
└─────────────────────────┼──────────────────────────▲──────────────────┘
                          │ HTTP + WebSocket         │ predictions {media_time,
                          ▼                          │ genre, confidence, topk}
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
                                             │   CAM heatmap / pred  │
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

Key design rules (enforced, not aspirational):

- **Two separate streams.** Playback (YouTube IFrame) and prediction (backend) are independent; predictions carry `media_time` and the frontend displays them only when playback reaches that time. This is the core sync mechanism — seek/pause/buffering all fall out of it.
- **The inference worker owns its ffmpeg process.** Raw frames never cross a message bus. Redis only carries control messages (session lifecycle, seek commands) — in this single-process demo the control contract runs over an in-proc queue with the exact message shapes of the future Redis pub/sub bus.
- **No video downloads to disk.** `yt-dlp` resolves a direct URL; ffmpeg pipes raw RGB frames (8 fps, decoded at 256², center-cropped to 224 — identical preprocessing to training) straight into the model window buffer.
- **Real attribution via CAM, not a stand-in.** R(2+1)D-18 ends in GAP+FC, so Grad-CAM weights collapse to the FC row: `cam = ReLU(Σ w_class·feat)` — no backprop needed. The ONNX export exposes the last conv features as a second output; every prediction carries a 14×14 attribution grid, rendered as a live heatmap over the video (user toggleable). The heatmap shows where the model looked for the *currently displayed label*; it is spatial attribution, not a concept-level explanation.
- **Direct MP4 URL is a first-class input** on the same pipeline — YouTube anti-bot measures or throttling can't blank out the demo.

## Quick start (Docker)

Prerequisites: Docker Desktop (with WSL2 backend on Windows). GPU optional.

```bash
docker compose up --build
# → open http://localhost:8080
```

That's it. The stack: `nginx :8080` → `frontend :3000` + `backend :8000` + `postgres 16` + `redis 7`. Model files are bind-mounted from `./models` (not baked into the image — swapping a model needs no rebuild).

**GPU:** the backend image ships `onnxruntime-gpu` (CUDA 12.4 + cuDNN 9 base). With `gpus: all` in `docker-compose.yml` (the default here) ONNX Runtime uses the CUDA EP. **On a CPU-only host, delete the `gpus: all` line in the `backend` service — the same image runs CPU EP, no other change needed** (provider selection is auto-detected: CUDA > DirectML > CPU). CPU single-clip latency is ~385 ms, comfortably inside the 1 Hz budget.

**Verify:**

```bash
curl http://localhost:8080/healthz
# {"ok":true,"ffmpeg":true,"model":true,"db":true,"redis":true,"active_sessions":0}
```

Then paste a YouTube dance URL, or use a direct MP4 for a deterministic demo, e.g.:
`https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4` (non-dance — should stay **Unknown** the whole way).

## Local development (without Docker)

- Backend: Python 3.12 venv (`requirements.txt`; Windows uses `onnxruntime-directml`), ffmpeg via `FFMPEG_BIN` env or unzipped under `tools/ffmpeg-<version>/bin/` (Windows dev: download the essentials build from gyan.dev — binaries are not shipped in this repo), `uvicorn server.main:app --port 8000`. Postgres via any instance reachable at `DATABASE_URL`.
- Frontend: `cd frontend && npm install && npm run dev` (set `NEXT_PUBLIC_API_BASE=http://localhost:8000` for cross-origin dev; defaults to it).
- Training/data pipeline: see `training/` and `data/` (not required to run the demo).

## Model

| | |
|---|---|
| Architecture | R(2+1)D-18, Kinetics-400 init, fine-tuned |
| Input | 16 frames @ 8 fps (2 s window), 224×224, 50% overlap → 1 Hz predictions |
| Deployment artifact | `models/dance_r2plus1d_fp16.onnx` (opset 18, FP16, static batch-1, 62.8 MB) |
| Smoothing | EMA α=0.4 → unknown threshold 0.45 → hysteresis 0.10 → min duration 2 updates |
| Training data | 176 self-collected source videos (weak labels) → pseudo-label filtering → group split **by source video** |

Test macro-F1 history (group-split held-out):

| Round | macro-F1 | Notes |
|---|---|---|
| v1 | 0.619 | 84 videos; hiphop/breakdance/contemporary confused (hiphop F1 0.08) |
| v2 | 0.823 | 152 videos + style-contradiction filtering; hiphop 0.79, breakdance 0.92 |
| v3 | **0.879** | +24 videos (contemporary/salsa focus), incremental training from v2 weights + 30% replay; v2 on the same test set = 0.852 |

Full per-class P/R, confusion matrices and training logs: [`docs/phase-1.md`](docs/phase-1.md), raw runs in `training/*.log`.

## Latency (measured)

| Stage | Value |
|---|---|
| Frame → prediction inside worker | mean ~118–140 ms (all source types, incl. CAM) |
| Single-clip inference | 85.9 ms (DML EP, Windows) / ~385 ms (CPU EP) |
| First prediction after `POST /session` | ~2–3 s — bounded below by the 2 s window fill + URL resolve (~1.6 s) + first frame (~0.7 s) |
| First prediction on WS connect (post prewarm) | MP4 ~4.2 s incl. player startup / YouTube ~0.5 s |
| Update rate | 1 Hz predictions, each carrying a 14×14 CAM attribution grid |

Definitions per `CLAUDE.md` §6: we do **not** promise 500 ms first frame (that's YouTube's player); "frame-to-overlay" target < 1 s is met by the 125 ms worker path plus network.

## Scaling (1 → 100 concurrent users)

The single-process demo was shaped so scaling is topological, not conceptual: session state moves to Redis, the in-proc control queue becomes Redis pub/sub (the message contract was designed as the bus contract from day one), and inference workers run as a pool of containers — one active session per worker, sticky-routed, auto-torn-down on WS close (session lifetime already equals connection lifetime). Raw frames still never cross the bus; each worker keeps its own ffmpeg. Capacity from measured numbers: one laptop-class GPU sustains ~11 sessions at 85 ms/clip, CPU tier as overflow; cross-session batching is projected at 2–3× but not measured. The real 100× bottlenecks are upstream: yt-dlp resolution (IP-bound — mitigated by a resolved-URL cache with ~6 h TTL and a proxy pool) and ffmpeg decode cost (roughly ~0.3 core/session, estimate). Details: [docs/report.md §5](docs/report.md).

## Repo layout

```
frontend/          Next.js app (player, overlay, dashboard, feedback UI)
server/            FastAPI backend + InferenceWorker + Dockerfile
nginx/             reverse proxy config (WS upgrade)
models/            ONNX models (dual-output: logits + features) + CAM FC weights (bind-mounted at runtime)
training/          train/export scripts + training logs
data/              dataset pipeline (manifest, window extraction, splits)
docs/              phase reports, decisions, limitations, metrics
spikes/            risk-spike scripts & E2E tests
```

## Known limitations

Honest list lives in [`docs/limitations.md`](docs/limitations.md). Highlights:

- **contemporary is the weakest class** (F1 0.756 in v3) — confusable with lyrical/modern and ballet; hiphop videos may alternate between hiphop/contemporary predictions.
- ~1/3 of predictions on unseen videos can oscillate during style transitions; hysteresis trades 1–2 s of switch delay for stability.
- yt-dlp direct URLs are IP-bound and expire (~6 h); the worker self-heals mid-session stream failures by re-resolving (up to 5 attempts before giving up), and MP4 fallback covers the rest.
- CAM attribution is spatially coarse (14×14 grid, one blob per ~32 px) and shows *where*, not *why conceptually*; the unknown-class heatmap is the least interpretable.
- Quantified edge cases: non-dance fallback (unknown recall 1.000), genre switching. **Not** quantified (no labeled set): multi-dancer accuracy, camera-motion stability.
