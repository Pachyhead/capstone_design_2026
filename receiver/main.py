from contextlib import asynccontextmanager
from threading import Lock
from pathlib import Path
import json

from receiver import Receiver
from config import PROJECT_ROOT

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STORAGE = PROJECT_ROOT / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)

FRONTEND_DIST = PROJECT_ROOT.parent / "tone" / "dist"

@asynccontextmanager
async def lifespan(app: FastAPI):
    with Receiver(
        storage=STORAGE,
        user_id=1,
        sender_id=1,
    ) as receiver:
        app.state.receiver = receiver
        app.state.receiver_lock = Lock()
        app.state.last_audio_file = None
        yield

app = FastAPI(lifespan=lifespan)

app.mount("/storage", StaticFiles(directory=STORAGE), name="storage")

@app.post("/set_my_id")
def set_my_id(my_id: int | None = None):
    if my_id is None: raise HTTPException(status_code=400, detail="your id is required. range is [0, 3]")
    receiver: Receiver = app.state.receiver
    receiver.user_id = my_id

    result = receiver.get_pending_messages()

    return result

@app.post("/play_voice")
def play_voice(message_id: str | None = None):
    if message_id is None: raise HTTPException(status_code=400, detail="message id is required.")
    receiver: Receiver = app.state.receiver

    duration = receiver.play_voice(message_id)
    return {"duration": duration}

@app.get("/")
def _spa_root():
    return FileResponse(FRONTEND_DIST / "index.html")


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def _spa_fallback(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")