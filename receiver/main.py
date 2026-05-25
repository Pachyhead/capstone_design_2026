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
        server_ip="127.0.0.1:8000",
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
    print(result)

    return {
        "message": "user_id updated",
        "user_id": my_id,
        "get_pending_messages": result,
    }

@app.post("/set_sender_id")
def set_sender_id(sender_id: int | None = None):
    if sender_id is None: raise HTTPException(status_code=400, detail="sender id is required. range is [0, 3]")
    receiver: Receiver = app.state.receiver
    receiver.peer_id = sender_id

    receiver.get_pending_messages()
    
    return {
        "message": "sender_id updated",
        "sender_id": sender_id
    }

@app.post("/play_voice")
def play_voice(message_id: int | None = None):
    if message_id is None: raise HTTPException(status_code=400, detail="message id is required.")
    receiver: Receiver = app.state.receiver
    
    receiver.play_voice(message_id)
    return f"successfully play voice"

@app.post("/get_message")
def get_message(message_id: int):
    receiver: Receiver = app.state.receiver

    with open(receiver.storage / "000001.json", "r", encoding="utf-8") as f:
        result = json.load(f)

    return {
        "message_id": result["message_id"],
        "sender_id": result["sender_id"],
        "message": result["message"],
        "emo_type": result["emo_type"],
        "send_time": result["send_time"]
    }


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