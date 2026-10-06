"""重現「seek 後 current genre 不顯示，但 last 10s 有更新」：
仿真播放頭（1× 前進、seek 立即跳躍），每秒檢查
  - bracket: 是否存在預測 p 使 t ∈ [p.t_start, p.t_end)（getPredictionAt 的主路徑）
  - nearest: |p.media_time - t| < 0.7（fallback）
持續 miss 而 buffer 內有預測 = 重現成功。
"""
import asyncio
import json
import sys
import time

import httpx
import websockets

API = "http://localhost:8000"


async def main(url: str) -> int:
    buf: list[dict] = []          # (t_start, t_end, media_time)
    player = {"t": 0.0}
    events: list[str] = []
    stop = asyncio.Event()

    async with httpx.AsyncClient(timeout=60) as hc:
        r = await hc.post(f"{API}/session", json={"url": url, "url_only_placeholder": None})
        r.raise_for_status()
        info = r.json()
    print(f"session {info['session_id']} kind={info['kind']}")

    async with websockets.connect(f"ws://localhost:8000{info['ws_url']}") as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready"

        async def receiver():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "prediction":
                    buf.append({"s": m["t_start"], "e": m["t_end"], "mt": m["media_time"]})
                    if len(buf) > 400:
                        del buf[:100]
                elif m.get("type") == "fatal":
                    events.append(f"FATAL {m.get('detail')}")
                    stop.set()

        async def player_sim():
            """仿真播放器：1× 前進；在 10s/25s/40s 時 seek 跳躍（含前後向）。"""
            t = 0.0
            last = time.monotonic()
            t0 = time.monotonic()
            seeks = [(10, 45.0), (25, 20.0), (40, 70.0)]  # (wall, target)
            i = 0
            while not stop.is_set():
                await asyncio.sleep(0.25)
                now = time.monotonic()
                t += now - last          # 1× 前進
                last = now
                if i < len(seeks) and now - t0 >= seeks[i][0]:
                    t = seeks[i][1]      # 播放頭立即跳
                    await ws.send(json.dumps({"type": "seek", "media_time": t}))
                    events.append(f"SEEK->{t:.0f}s @wall {now - t0:.0f}")
                    i += 1
                player["t"] = t
                try:
                    await ws.send(json.dumps({"type": "playhead", "media_time": t}))
                except websockets.ConnectionClosed:
                    return

        recv = asyncio.create_task(receiver())
        sim = asyncio.create_task(player_sim())

        # 觀察 55s，每秒檢查 bracket 覆蓋
        for wall in range(55):
            await asyncio.sleep(1)
            t = player["t"]
            recent = [p for p in buf if p["mt"] >= t - 10 and p["mt"] <= t + 1]
            bracket = any(p["s"] <= t < p["e"] for p in buf)
            nearest = min((abs(p["mt"] - t) for p in buf), default=99)
            mark = ""
            if not bracket and nearest > 0.7:
                mark = "  << MISS（儀表板會顯示 —）"
            if recent:
                print(f"wall {wall:2d}  player {t:6.1f}  newest_mt {buf[-1]['mt']:6.1f}  "
                      f"recent {len(recent):2d}  nearest {nearest:5.2f}{mark}")
            for e in events:
                if e:
                    print(f"        [{e}]")
                    events = []
        stop.set()
        recv.cancel()
        sim.cancel()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1])))
