from contextlib import asynccontextmanager
from threading import Lock
from pathlib import Path

from receiver import Receiver
from config import PROJECT_ROOT

from fastapi import FastAPI, HTTPException

@asynccontextmanager
async def lifespan(app: FastAPI):
    with Receiver(
        storage=PROJECT_ROOT / "storage",
        user_id=1,
        sender_id=1,
        server_ip="127.0.0.1:8000",
    ) as receiver:
        app.state.receiver = receiver
        app.state.receiver_lock = Lock()
        app.state.last_audio_file = None
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/set_my_id")
def set_user_id(my_id: int | None = None):
    if my_id is None: raise HTTPException(status_code=400, detail="your id is required. range is [0, 3]")
    receiver: Receiver = app.state.receiver
    receiver.user_id = my_id
    
    result = receiver.get_pending_messages()

    return {
        "message": "user_id updated",
        "user_id": my_id,
        "get_pending_messages": {result}
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
