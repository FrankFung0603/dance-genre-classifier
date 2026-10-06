# Technical Report — Real-Time Dance Genre Classification

> **Word deliverable:** [`report.docx`](report.docx)（regenerate: `.venv/Scripts/python.exe tools/make_report_docx.py` — this markdown is the source of truth）

*Paste a YouTube URL, watch the video, and see the dance genre predicted every second, time-aligned with playback — five genres plus an honest "Unknown". Full measurements: `docs/`, `training/`.*

## 1. Architecture

```
Browser: YouTube IFrame player + overlay/dashboard (Next.js)
  │ playhead / pause / seek every 250ms      ▲ predictions {media_time, genre, conf}
  ▼                                         │ WebSocket (nginx :8080)
FastAPI backend ── InferenceWorker (1/session):
  yt-dlp → direct URL → ffmpeg pipe (8 fps, 256²→224² raw RGB, zero disk I/O)
  → R(2+1)D-18 ONNX FP16 @ 1 Hz → EMA/unknown smoothing → emit
  → CAM attribution grid per prediction (GAP+FC ⇒ Grad-CAM without backprop)
Redis (control msgs only) · PostgreSQL (👍/👎 feedback)
```

- **Two streams correlated by media time.** Predictions display only when playback reaches their timestamp; seek, pause and buffering correctness fall out of this single mechanism. Playback stays on YouTube's IFrame — no backend proxy.
- **Chunked streaming, never a download.** yt-dlp resolves a direct URL; ffmpeg pipes raw frames (byte-identical preprocessing to training) straight into the window buffer at 2.58× realtime, with a 10 s network timeout and self-healing re-resolve on stall (up to 5 attempts before giving up). Direct MP4 input rides the same pipeline, so YouTube anti-bot measures can't blank a demo.
- **Seek** is debounced 500 ms and restarts ffmpeg at `offset = target` — a −2 s warmup design proved permanently laggy on throttled sources (bracket simulation; documented deviation).
- **Inference optimizations**: ONNX opset 18, FP16 (62.8 MB), lightweight R(2+1)D-18, 8 fps smart sampling; execution provider auto-detects CUDA → DirectML → CPU from one image. Smoothing (EMA α=0.4, unknown threshold 0.45, hysteresis 0.10, min-duration 2) is why non-dance content reads **Unknown** instead of a hallucinated genre.
- **Explainability is real attribution**: the model ends in GAP+FC, so Grad-CAM weights collapse exactly to the FC row — `cam = ReLU(Σ w_class·feat)` needs no backprop. The ONNX export adds the last conv features (512×2×14×14) as a second output; every prediction carries a 14×14 attribution grid for the currently displayed label, rendered live as a heatmap over the video (user toggleable). Spatial attribution, honestly coarse — not a pose-skeleton stand-in.

## 2. Dataset curation

No public dance-genre dataset exists at this granularity, so the data is **176 self-collected public YouTube videos** with weak labels — 5 genres plus **unknown negatives** (cartoons, conferences, cooking). Three curation mechanisms mattered more than volume: (1) a Kinetics-400 scorer drops low-dance windows (~50 %); (2) a style-contradiction filter drops windows whose K400 top-1 names a *different specific genre* ≥0.35 — training only, val/test untouched; (3) **group split by source video**, since window-level splits leak near-duplicates and inflate F1.

| Round | Sources | Test n | macro-F1 | Change |
|---|---|---|---|---|
| v1 | 84 | 631 | 0.619 | baseline; label noise |
| v2 | 152 | 1,174 | 0.823 | filtering + more sources |
| v3 | 176 | 1,414 | **0.879** | contemporary/salsa focus, incremental |

## 3. Training methodology

R(2+1)D-18 fine-tuned from Kinetics-400 (never frozen). v1/v2 re-initialized when data strategy changed; **v3 was incremental** from v2 weights: +908 new windows with 30 % old-window replay against catastrophic forgetting (8 epochs, lr 1e-4 head / 1e-5 backbone). Incremental rounds are the operating model — new videos are a ~1 h round-trip, not a re-training. On an 8 GB laptop GPU: batch 4 × grad-accum 4, uint8 memmap window cache, `cudnn.benchmark` (epoch 26→6.6 min), atomic checkpoints + verified `--resume` (survived one native crash, zero loss). Evaluation discipline: group-split held-out only; rounds are compared on the identical test set.

## 4. Performance analysis

**Quality** (v2 vs v3 on the identical n=1,414 held-out test):

| class | v2 | v3 | Δ |
|---|---|---|---|
| ballet | 0.853 | 0.810 | −4.3 pt |
| breakdance | 0.918 | 0.929 | +1.1 |
| contemporary | 0.646 | **0.756** | **+11.0** |
| hiphop | 0.837 | 0.831 | −0.6 |
| salsa | 0.995 | 0.984 | −1.1 |
| unknown | 0.864 | **0.965** | +10.1 |
| **macro** | **0.852** | **0.879** | **+2.7** |

The ballet dip is the deliberate price of relaxing contemporary's contradiction threshold; it bought +11 contemporary and unknown precision 0.760→0.957. Full P/R and confusion matrices: `docs/phase-1.md`, `docs/phase1_confusion_matrix*.png`.

**Latency (measured)**:

| stage | value |
|---|---|
| frame → prediction (worker) | **mean 118–138 ms** (DML/CUDA EP, all source types, incl. CAM) |
| single-clip inference | 85.9 ms GPU / 611 ms CPU EP (still within 1 Hz) |
| first prediction | 2–3 s — bounded by the 2 s window fill + 1.6 s resolve + 0.7 s first frame |
| update rate | 1 Hz predictions, each with a 14×14 CAM grid |

The "< 1 s end-to-end" requirement is met on the frame-to-overlay path; the 2–3 s *first* prediction is physics — you cannot classify 2 seconds of video in less than 2 seconds of it.

**Edge cases**: non-dance stays Unknown (unknown recall **1.000**; Big Buck Bunny ≥80 % Unknown); abrupt switches cost 1–2 s via hysteresis; low resolution is in-distribution (training data is ≤480p); multi-dancer / camera motion are handled only implicitly — honestly **not quantified** (§6). The 👍/👎 feedback loop lands in PostgreSQL as correction samples for the next incremental round, and explainability is delivered as **live CAM attribution** — the model's own weight-computed attention, not a surrogate visualization.

## 5. Scaling from 1 to 100 concurrent users

**Design (not implemented in the demo)**. The demo's shape makes scaling topological, not conceptual: session state moves to Redis; the in-proc control queue becomes Redis pub/sub (the message contract was designed as the bus contract from day one); workers become a container pool, one active session each, sticky-routed and torn down on WS close (session lifetime already equals connection lifetime). Raw frames still never cross the bus. Capacity from measured numbers: ~11 sessions per laptop-class GPU at 85 ms/clip, CPU tier as overflow; cross-session batching is projected at 2–3× but **not measured**. The real 100× bottlenecks are upstream: yt-dlp resolution (IP-bound — mitigated by a ~6 h resolved-URL cache and a proxy pool) and ffmpeg decode cost (roughly ~0.3 core/session, estimate).

## 6. Limitations

- **Multi-dancer / camera-motion accuracy is not quantified**: no condition-labeled test set exists, and single-label ground truth is ill-defined for two dancers doing two genres. Minimal path: tag 20–30 test videos per window (multi-dancer y/n, camera static/pan/handheld) and stratify existing predictions — no retraining.
- **contemporary remains weakest** (F1 0.76); the fix is more data, which the incremental pipeline makes cheap.
- **First-frame latency is not promised**: playback is YouTube's player by design; everything downstream is measured and fast.
