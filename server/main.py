"""FastAPI：POST /session（解析 URL）→ WS /ws/{sid}（預測 + CAM 歸因 + 控制）。

Session 生命週期 = WS 連線（單客戶端 demo 模型）：
    POST /session 解析並驗證 URL（錯誤走 HTTP status）
    → WS 連上才啟動 worker、斷線即停、session 註銷
控制訊息（pause/seek/playhead）走 WS 文字訊息；本機 dev 用 in-proc queue 承接控制，
Redis 控制匯流排留待 Phase 5 docker-compose（多進程拆分）時接入——raw 幀一律不過匯流排。
"""
import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server import resolver
from server.config import (
    CLASSES,
    SEEK_DEBOUNCE_S,
    STREAM_FPS,
    WIN_FRAMES,
    MODEL_ONNX,
    find_ffmpeg,
)
from server.db import Feedback, SessionLocal, init_db
from server.worker import InferenceWorker, prewarm

sessions: dict[str, dict] = {}


class SessionIn(BaseModel):
    url: str = Field(min_length=4)


class SessionOut(BaseModel):
    session_id: str
    kind: str
    ws_url: str
    classes: list[str]


class FeedbackIn(BaseModel):
    session_id: str = Field(min_length=1)
    url: str = Field(min_length=4)
    kind: str = "unknown"
    media_time: float = Field(ge=0)
    predicted: str
    confidence: float = Field(ge=0, le=1)
    correct: str | None = None   # 👍 = None；👎 = 正確類別
    verdict: str = Field(pattern="^(up|down)$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    find_ffmpeg()  # 早失敗：ffmpeg 不在就別起服務
    if not MODEL_ONNX.exists():
        raise RuntimeError(f"model missing: {MODEL_ONNX}")
    # 預熱 ONNX(DML)+dummy 推理 + CAM 權重載入 —— 原本這些
    # lazy 發生在用戶貼第一支影片的當下（可達 10s+），移到啟動時一次性付清。
    warm = prewarm(str(MODEL_ONNX))
    print(f"[prewarm] model={warm['model_s']}s", flush=True)
    app.state.db_ok = init_db()  # 反饋入庫；DB 不在服務照起（demo 降級）
    yield


app = FastAPI(title="dance-classifier streaming backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _redis_ok() -> bool | None:
    """REDIS_URL 有設才檢查（compose）；本機 dev 未設 → None（不適用）。"""
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis  # 延遲載入：本機 dev venv 未裝 redis 套件也不影響
        return bool(redis.Redis.from_url(url, socket_connect_timeout=1).ping())
    except Exception:
        return False


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "ffmpeg": bool(find_ffmpeg()),
        "model": MODEL_ONNX.exists(),
        "db": bool(getattr(app.state, "db_ok", False)),
        "redis": _redis_ok(),
        "active_sessions": len(sessions),
    }


@app.post("/session", response_model=SessionOut)
def create_session(body: SessionIn) -> SessionOut:
    kind, playable = resolver.classify(body.url)
    if kind == "youtube":
        # 立刻解析：失敗走 HTTP 錯誤，別等 WS 才爆
        resolver.resolve_youtube(playable)
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {
        "input": body.url,
        "kind": kind,
        "worker": None,
    }
    return SessionOut(
        session_id=sid, kind=kind, ws_url=f"/ws/{sid}", classes=CLASSES,
    )


@app.delete("/session/{sid}")
def delete_session(sid: str) -> dict:
    s = sessions.pop(sid, None)
    if s and s.get("worker") is not None:
        s["worker"].stop_and_join()
    return {"deleted": sid in sessions or s is not None}


# ---- Phase 4：反饋閉環（PostgreSQL） ----

@app.post("/feedback", status_code=201)
def create_feedback(body: FeedbackIn) -> dict:
    if not getattr(app.state, "db_ok", False):
        raise HTTPException(status_code=503, detail="database unavailable")
    if body.predicted not in CLASSES or (body.correct is not None and body.correct not in CLASSES):
        raise HTTPException(status_code=400, detail="unknown genre")
    with SessionLocal() as db:
        row = Feedback(
            session_id=body.session_id, url=body.url, kind=body.kind,
            media_time=body.media_time, predicted=body.predicted,
            confidence=body.confidence, correct=body.correct, verdict=body.verdict,
        )
        db.add(row)
        db.commit()
        return {"id": row.id, "stored": True}


@app.get("/feedback")
def list_feedback(session_id: str | None = None, limit: int = 50) -> dict:
    if not getattr(app.state, "db_ok", False):
        raise HTTPException(status_code=503, detail="database unavailable")
    limit = max(1, min(limit, 500))
    with SessionLocal() as db:
        q = db.query(Feedback).order_by(Feedback.id.desc())
        if session_id:
            q = q.filter(Feedback.session_id == session_id)
        rows = q.limit(limit).all()
        return {
            "count": len(rows),
            "items": [
                {
                    "id": r.id, "session_id": r.session_id, "url": r.url, "kind": r.kind,
                    "media_time": r.media_time, "predicted": r.predicted,
                    "confidence": r.confidence, "correct": r.correct,
                    "verdict": r.verdict, "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
        }


def _source_getter(s: dict):
    def get(force: bool = False) -> tuple[str, str]:
        kind, playable = resolver.classify(s["input"])
        if kind == "youtube":
            return "youtube", resolver.resolve_youtube(playable, force=force)
        return kind, playable
    return get


@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str) -> None:
    await ws.accept()
    s = sessions.get(sid)
    if s is None:
        await ws.close(code=4404)
        return
    if s.get("worker") is not None:
        await ws.close(code=4409)  # 單客戶端：已有連線
        return

    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue(maxsize=512)

    def sink(msg: dict) -> None:
        def put():
            if out_q.full():
                out_q.get_nowait()  # 丟最舊，保即時性
            out_q.put_nowait(msg)
        loop.call_soon_threadsafe(put)

    worker = InferenceWorker(str(MODEL_ONNX), _source_getter(s), sink)
    s["worker"] = worker
    worker.start()

    # ---- seek debounce（~500ms 內只吃最後一個 seek） ----
    pending_seek: dict = {"value": None, "seq": 0, "event": asyncio.Event()}

    async def debounced_seeker() -> None:
        while True:
            await pending_seek["event"].wait()
            pending_seek["event"].clear()
            await asyncio.sleep(SEEK_DEBOUNCE_S)
            v = pending_seek["value"]
            if v is not None and worker.is_alive():
                worker.control({"type": "seek", "media_time": v})
            pending_seek["value"] = None

    seeker = asyncio.create_task(debounced_seeker())

    async def sender() -> None:
        await ws.send_json({
            "type": "ready", "classes": CLASSES,
            "stream": {"fps": STREAM_FPS, "win_frames": WIN_FRAMES},
        })
        while True:
            msg = await out_q.get()
            await ws.send_json(msg)

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "ping":
                await ws.send_json({"type": "pong"})
            elif mtype == "playhead":
                worker.control({"type": "playhead",
                                "media_time": float(msg["media_time"])})
            elif mtype == "pause":
                worker.control({"type": "pause"})
            elif mtype == "resume":
                # media_time 必須轉發：暫停中拖進度條後恢復，target=當前播放頭；
                # 丟掉的話 worker 退回 paused_at（舊暫停點）→ 預測流永遠落後
                ctl: dict = {"type": "resume"}
                if msg.get("media_time") is not None:
                    ctl["media_time"] = float(msg["media_time"])
                worker.control(ctl)
            elif mtype == "seek":
                pending_seek["value"] = float(msg["media_time"])
                pending_seek["seq"] += 1
                pending_seek["event"].set()
    except WebSocketDisconnect:
        pass
    finally:
        seeker.cancel()
        send_task.cancel()
        worker.stop_and_join()
        sessions.pop(sid, None)
