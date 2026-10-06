"""Phase 5 端到端驗收：對 docker-compose stack（nginx :8080）打。

與 server/e2e_test.py（Phase 2，自起 uvicorn + 本機檔案）不同：本腳本不起任何服務，
目標是「一鍵部署後的完整 demo 路徑」——nginx 反代、WS upgrade、postgres 入庫、
yt-dlp/direct MP4 兩種來源，全部走對外入口。

流程：
    1. GET  /healthz                → ok/model/db/redis 全 true
    2. POST /session（direct MP4）  → WS：ready、預測 1Hz、media_time 遞增、latency<1s
    3. seek 恢復（同 Phase 2 語意）
    4. POST /feedback ×2 → GET /feedback 讀回（nginx → backend → postgres 全鏈路）
    5. POST /session（YouTube 舞蹈 URL）→ WS：預測流 + pose 流
    6. 非舞蹈 MP4 應以 unknown 為主（BBB）

Usage:
    python spikes/test_compose_e2e.py [--base http://localhost:8080]
        [--yt <youtube url>] [--mp4 <direct mp4 url>]
"""
import argparse
import asyncio
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

results: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def http_json(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path, method=method,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


async def collect(ws, seconds: float) -> list[dict]:
    got, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=max(0.05, seconds - (time.monotonic() - t0)))
            got.append(json.loads(msg))
        except TimeoutError:
            break
    return got


async def ws_flow(url: str, yt: str | None) -> None:
    import websockets

    # ---- 1) healthz：全綠（nginx → backend → postgres/redis） ----
    code, h = http_json("GET", "/healthz")
    check("healthz_via_nginx", code == 200 and h.get("ok"), f"{h}")
    check("db_connected", h.get("db") is True, f"db={h.get('db')}")
    check("redis_connected", h.get("redis") is True, f"redis={h.get('redis')}")

    # ---- 2) direct MP4 來源：預測流 + seek ----
    code, sess = http_json("POST", "/session", {"url": url})
    check("session_created_mp4", code == 200 and bool(sess.get("session_id")),
          f"kind={sess.get('kind')}")
    async with websockets.connect(
        BASE.replace("http", "ws", 1) + sess["ws_url"], max_size=2**22
    ) as ws:
        t0 = time.monotonic()
        msgs = await collect(ws, 14)
        wall = time.monotonic() - t0
        preds = [m for m in msgs if m["type"] == "prediction"]
        check("ws_ready_msg", any(m["type"] == "ready" for m in msgs),
              f"types={[m['type'] for m in msgs][:6]}")
        check("predictions_flow_mp4", len(preds) >= 6, f"n={len(preds)} in {wall:.1f}s")
        if preds:
            rate = wall / max(1, len(preds))
            check("prediction_rate_1hz", 0.6 <= rate <= 2.5, f"{rate:.2f}s/pred")
            mt = [p["media_time"] for p in preds]
            check("media_time_monotonic", all(b > a for a, b in zip(mt, mt[1:])))
            lats = [p["latency_ms"] for p in preds if "latency_ms" in p]
            if lats:
                check("frame_to_prediction_lt_1s", max(lats) < 1000,
                      f"mean={sum(lats)/len(lats):.0f}ms max={max(lats):.0f}ms")
            genres = [p["genre"] for p in preds]
            if "BigBuckBunny" in url or "big_buck_bunny" in url:
                check("nondance_stays_unknown", genres.count("unknown") >= len(genres) * 0.8,
                      f"genres={set(genres)}")

        # ---- 3) seek（debounce 500ms 後 state(seek) → 新預測） ----
        await ws.send(json.dumps({"type": "seek", "media_time": 30.0}))
        msgs = await collect(ws, 12)
        idx = next((i for i, m in enumerate(msgs)
                    if m["type"] == "state" and m.get("reason") == "seek"), None)
        check("seek_state_received", idx is not None)
        if idx is not None:
            post = [m for m in msgs[idx:] if m["type"] == "prediction"]
            check("seek_recovered", len(post) >= 1, f"n={len(post)}")
            if post:
                check("seek_position_accurate", abs(post[0]["media_time"] - 30.0) <= 2.0,
                      f"first={post[0]['media_time']}")

    # ---- 4) 反饋閉環（nginx → backend → postgres） ----
    sid = sess["session_id"]
    code, r1 = http_json("POST", "/feedback", {
        "session_id": sid, "url": url, "kind": sess.get("kind", "mp4"),
        "media_time": 12.3, "predicted": "unknown", "confidence": 0.5,
        "correct": None, "verdict": "up"})
    code2, r2 = http_json("POST", "/feedback", {
        "session_id": sid, "url": url, "kind": sess.get("kind", "mp4"),
        "media_time": 45.6, "predicted": "hiphop", "confidence": 0.8,
        "correct": "breakdance", "verdict": "down"})
    code3, bad = http_json("POST", "/feedback", {
        "session_id": sid, "url": url, "kind": "mp4", "media_time": 1.0,
        "predicted": "not-a-genre", "confidence": 0.5, "verdict": "up"})
    check("feedback_post_roundtrip", r1.get("stored") and r2.get("stored"),
          f"up={r1.get('id')} down={r2.get('id')}")
    check("feedback_invalid_genre_400", code3 == 400, f"code={code3}")
    code4, lst = http_json("GET", f"/feedback?session_id={sid}")
    check("feedback_get_readback",
          code4 == 200 and lst.get("count") == 2
          and {i["verdict"] for i in lst["items"]} == {"up", "down"}
          and any(i["correct"] == "breakdance" for i in lst["items"]),
          f"count={lst.get('count')}")

    # ---- 5) YouTube 來源：yt-dlp 解析 + 預測 + CAM ----
    if yt:
        code, sess2 = http_json("POST", "/session", {"url": yt})
        check("session_created_youtube", code == 200 and sess2.get("kind") == "youtube",
              f"kind={sess2.get('kind')}")
        if code == 200:
            async with websockets.connect(
                BASE.replace("http", "ws", 1) + sess2["ws_url"], max_size=2**22
            ) as ws:
                msgs = await collect(ws, 14)
                preds = [m for m in msgs if m["type"] == "prediction"]
                check("predictions_flow_youtube", len(preds) >= 5, f"n={len(preds)}")
                cams = [p for p in preds
                        if isinstance(p.get("cam_grid"), list) and p.get("cam_size")]
                check("cam_streaming", len(cams) >= 1,
                      f"n={len(cams)}/{len(preds)}（缺 fc_weights.npy 時無 cam）")
                if preds:
                    check("youtube_payload_schema",
                          all(set(p) >= {"media_time", "genre", "confidence"} for p in preds),
                          f"genres={[p['genre'] for p in preds[:5]]}")


def main() -> None:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--mp4", default="https://archive.org/download/BigBuckBunny_124/"
                                     "Content/big_buck_bunny_720p_surround.mp4")
    ap.add_argument("--yt", default=None, help="YouTube 舞蹈影片 URL（可省略）")
    args = ap.parse_args()
    BASE = args.base.rstrip("/")
    print(f"compose E2E target: {BASE}", flush=True)
    asyncio.run(ws_flow(args.mp4, args.yt))

    n_fail = sum(1 for r in results.values() if not r["ok"])
    print(f"\n=== Compose E2E {'PASS' if n_fail == 0 else 'FAIL'} "
          f"({len(results) - n_fail}/{len(results)} checks) ===")
    out = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "phase5_e2e_result.json"
    out.write_text(json.dumps({"target": BASE, "passed": n_fail == 0, "checks": results},
                              indent=2, ensure_ascii=False))
    print(f"-> {out}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
