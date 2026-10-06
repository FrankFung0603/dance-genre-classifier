"""E2E WS 測試：首預測延遲（預熱後）+ seek 恢復 + seek 超出片長的錯誤語義。

用法：先起後端，再
    .venv/Scripts/python.exe spikes/test_ws_latency.py <mp4_or_youtube_url> [seek_target]
"""
import asyncio
import json
import sys
import time

import httpx
import websockets

API = "http://localhost:8000"


async def drain(ws: websockets.WebSocketClientProtocol, results: dict,
                stop_event: asyncio.Event) -> None:
    async for raw in ws:
        m = json.loads(raw)
        t = m.get("type")
        if t == "prediction" and "first_pred_s" not in results:
            results["first_pred_s"] = time.monotonic() - results["t0"]
            results["first_pred"] = m
        elif t in ("fatal",):
            results["fatal"] = m.get("detail")
            stop_event.set()
        elif t == "state" and m.get("reason") in ("ended",):
            results["ended"] = m
            stop_event.set()
        if t == "prediction":
            results["last_pred"] = m
            results["n_preds"] = results.get("n_preds", 0) + 1


async def main(url: str, seek_target: float | None) -> int:
    results: dict = {}
    stop_event = asyncio.Event()
    async with httpx.AsyncClient(timeout=60) as hc:
        t0 = time.monotonic()
        r = await hc.post(f"{API}/session", json={"url": url, })
        r.raise_for_status()
        info = r.json()
        results["t0"] = time.monotonic() - t0
        print(f"session resolve: {results['t0']:.2f}s  sid={info['session_id']} kind={info['kind']}")

    async with websockets.connect(f"ws://localhost:8000{info['ws_url']}") as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready"
        t_ws = time.monotonic()
        results["t0"] = t_ws
        task = asyncio.create_task(drain(ws, results, stop_event))
        # 模擬播放頭回報（pacing 需要）
        async def playhead():
            t = 0.0
            while not stop_event.is_set():
                try:
                    await ws.send(json.dumps({"type": "playhead", "media_time": t}))
                except websockets.ConnectionClosed:
                    return
                t += 0.25
                await asyncio.sleep(0.25)
        ph = asyncio.create_task(playhead())

        if seek_target is not None:
            await asyncio.sleep(8)  # 讓前面先出幾個預測
            n_before = results.get("n_preds", 0)
            t_seek = time.monotonic()
            await ws.send(json.dumps({"type": "seek", "media_time": seek_target}))
            results["seek_at"] = t_seek
            results["n_before_seek"] = n_before

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=40)
        except asyncio.TimeoutError:
            print("[timeout] 40s 內沒等到結束條件（觀察期結束）")
        ph.cancel()
        task.cancel()

    print(json.dumps({k: v for k, v in results.items() if k != "t0"} | {
        "session_resolve_s": round(results["t0"], 2),
    }, ensure_ascii=False, default=str))
    if "first_pred_s" in results:
        fp = results["first_pred"]
        print(f"first prediction: {results['first_pred_s']:.2f}s after WS ready "
              f"→ {fp['genre']} {fp['confidence']:.2f} @{fp['media_time']}s")
    if "seek_at" in results:
        lp = results.get("last_pred", {})
        after = results.get("n_preds", 0) - results["n_before_seek"]
        gap = lp.get("media_time", "?")
        print(f"after seek({seek_target}s): {after} predictions, last @ {gap}s")
    if results.get("fatal"):
        print(f"FATAL: {results['fatal']}")
        return 1
    return 0


if __name__ == "__main__":
    url = sys.argv[1]
    seek = float(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(asyncio.run(main(url, seek)))
