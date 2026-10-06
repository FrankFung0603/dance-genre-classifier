"""Phase 2 端到端驗收（本機 dev，local file 來源走同一條 pipeline）。

流程：
    1. 起 uvicorn（子進程，port 8123）
    2. POST /session（local raw 影片；MP4/YouTube 同管線，僅 resolver 不同）
    3. WS 收 10s 訊息：驗證 ready / stream_started / prediction ≥6 且 1Hz /
       media_time 遞增 / cam_grid 到貨 / probs 6 類齊
    4. seek 到 30s → 驗證恢復且新預測 t ≈ 31±1.5
    5. pause → 無預測；resume → 恢復
    6. 輸出 PASS/FAIL 摘要（產出 docs/phase2_e2e_result.json）

Usage:
    python server/e2e_test.py [--video path]
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from server.config import ROOT  # noqa: E402

PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}"

results: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


async def drain_until(ws, want_types: set[str], timeout_s: float) -> list[dict]:
    """收訊息直到任一 want_types 出現或 timeout；回傳期間收到的全部訊息。"""
    got = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=max(0.05, timeout_s - (time.monotonic() - t0)))
        except TimeoutError:
            break
        m = json.loads(msg)
        got.append(m)
        if m.get("type") in want_types:
            break
    return got


async def collect(ws, seconds: float) -> list[dict]:
    got, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=max(0.05, seconds - (time.monotonic() - t0)))
            got.append(json.loads(msg))
        except TimeoutError:
            break
    return got


def pick_video() -> str:
    """挑一支 30s 處解碼得到的影片（本地 dev 來源）。"""
    from server.config import find_ffmpeg, STREAM_SIZE

    candidates = sorted((ROOT / "data" / "raw").glob("*/*.mp4"))[:8]
    if not candidates:
        raise SystemExit("no local mp4 under data/raw (dev test source)")
    fb = STREAM_SIZE * STREAM_SIZE * 3
    chosen = str(candidates[0].resolve())
    for c in candidates:
        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", "30",
               "-i", str(c), "-frames:v", "1",
               "-vf", f"scale={STREAM_SIZE}:{STREAM_SIZE}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True, timeout=30).stdout
        if len(out) >= fb:
            chosen = str(c.resolve())
            break
    return chosen


def http_json(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path, method=method,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


async def run_flow(video: str) -> None:
    import websockets

    sess = http_json("POST", "/session", {"url": video})
    print(f"session: {sess}", flush=True)
    check("session_created", bool(sess.get("session_id")))

    async with websockets.connect(f"{WS_URL}/ws/{sess['session_id']}", max_size=2**22) as ws:
        # ---- 1) 初始串流：10 秒 ----
        t0 = time.monotonic()
        msgs = await collect(ws, 12)
        wall = time.monotonic() - t0
        preds = [m for m in msgs if m["type"] == "prediction"]
        states = [m for m in msgs if m["type"] in ("ready", "stream_started")]

        check("ready_and_stream_started",
              any(m["type"] == "ready" for m in states)
              and any(m["type"] == "stream_started" for m in states),
              f"states={[m['type'] for m in states]}")
        check("predictions_flow", len(preds) >= 6, f"n={len(preds)} in {wall:.1f}s")
        if preds:
            rate = wall / max(1, len(preds))
            check("prediction_rate_1hz", 0.6 <= rate <= 2.5, f"{rate:.2f}s/pred")
            mt = [p["media_time"] for p in preds]
            check("media_time_monotonic", all(b > a for a, b in zip(mt, mt[1:])),
                  f"{mt[:4]}...")
            check("payload_schema",
                  all(set(p) >= {"media_time", "t_start", "t_end", "genre", "confidence",
                                 "genre_raw", "topk", "probs"} and len(p["probs"]) == 6
                      for p in preds),
                  f"genres={[p['genre'] for p in preds[:5]]}")
        cams = [p for p in preds
                if isinstance(p.get("cam_grid"), list) and p.get("cam_size")]
        if cams:
            ok = all(
                len(p["cam_grid"]) == p["cam_size"][0] * p["cam_size"][1]
                and all(0.0 <= v <= 1.0 for v in p["cam_grid"])
                for p in cams
            )
            check("cam_grid_valid", ok,
                  f"n={len(cams)}/{len(preds)} size={cams[0]['cam_size']}")
        else:
            check("cam_grid_valid", False, "no prediction carried cam_grid")
        lats = [p["latency_ms"] for p in preds if "latency_ms" in p]
        if lats:
            check("frame_to_prediction_lt_1s", max(lats) < 1000,
                  f"mean={sum(lats)/len(lats):.0f}ms max={max(lats):.0f}ms n={len(lats)}")

        # ---- 2) seek 到 30s ----
        # 注意：seek 有 500ms debounce，舊串流的殘留預測會先到。
        # 正確判斷：state(reason=seek) 之後的第一個 prediction 才是新串流的。
        await ws.send(json.dumps({"type": "seek", "media_time": 30.0}))
        msgs = await collect(ws, 12)
        seek_state_idx = next((i for i, m in enumerate(msgs)
                               if m["type"] == "state" and m.get("reason") == "seek"), None)
        check("seek_state_received", seek_state_idx is not None,
              f"types={[m['type'] for m in msgs][:10]}")
        if seek_state_idx is not None:
            post = [m for m in msgs[seek_state_idx:] if m["type"] == "prediction"]
            check("seek_recovered", len(post) >= 1, f"n={len(post)}")
            if post:
                mt = post[0]["media_time"]
                # 預測是 2s 窗中心：seek(30) → 窗 [29,31] → 中心 30.0；+1s 誤差容限
                check("seek_position_accurate", abs(mt - 30.0) <= 2.0,
                      f"first post-seek pred media_time={mt}")

        # ---- 3) pause / resume ----
        await ws.send(json.dumps({"type": "pause"}))
        await asyncio.sleep(0.3)
        msgs = await collect(ws, 3)
        check("pause_stops_predictions",
              all(m["type"] != "prediction" for m in msgs),
              f"msgs={[m['type'] for m in msgs]}")
        check("pause_state_msg", any(m.get("type") == "state" and m.get("playing") is False
                                     for m in msgs))
        await ws.send(json.dumps({"type": "resume"}))
        msgs = await drain_until(ws, {"prediction"}, 12)
        check("resume_restarts_stream",
              any(m["type"] == "prediction" for m in msgs),
              f"types={[m['type'] for m in msgs][:8]}")

    # ---- 4) 斷線後 session 應註銷 ----
    await asyncio.sleep(1.0)
    h = http_json("GET", "/healthz")
    check("session_cleaned_on_disconnect", h["active_sessions"] == 0,
          f"active_sessions={h['active_sessions']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="本機檔案路徑（dev 來源）")
    ap.add_argument("--url", default=None, help="直接指定來源（YouTube URL = 驗證 yt-dlp 路徑）")
    args = ap.parse_args()
    video = args.url or args.video or pick_video()
    print(f"e2e source: {video}", flush=True)

    srv = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            try:
                http_json("GET", "/healthz")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            raise SystemExit("server did not come up")
        asyncio.run(run_flow(video))
    finally:
        srv.terminate()

    n_fail = sum(1 for r in results.values() if not r["ok"])
    print(f"\n=== E2E {'PASS' if n_fail == 0 else 'FAIL'} "
          f"({len(results) - n_fail}/{len(results)} checks) ===")
    out = ROOT / "docs" / "phase2_e2e_result.json"
    out.write_text(json.dumps({"passed": n_fail == 0, "checks": results}, indent=2))
    print(f"-> {out}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
