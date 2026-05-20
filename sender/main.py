from contextlib import asynccontextmanager
from threading import Lock
from pathlib import Path

from sender import Sender
from config import PROJECT_ROOT

from scipy.io.wavfile import read
import numpy as np
from fastapi import FastAPI, HTTPException

@asynccontextmanager
async def lifespan(app: FastAPI):
    with Sender(
        storage=PROJECT_ROOT / "storage",
        user_id=1,
        receiver_id=1,
        server_ip="127.0.0.1:8000",
        fsq_path=str(PROJECT_ROOT / "sender_models" / "skip_kl_8d_8L_kl05_1e-4.pt")
    ) as sender:
        app.state.sender = sender
        app.state.sender_lock = Lock()
        app.state.last_audio_file = None
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/set_my_id")
def set_user_id(value: int | None = None):
    if value is None: raise HTTPException(status_code=400, detail="value is required. range is [0, 3]")
    sender: Sender = app.state.sender
    sender.user_id = value
    
    return {
        "message": "user_id updated",
        "user_id": value
    }

@app.post("/set_receiver_id")
def set_target_id(value: int | None = None):
    if value is None: raise HTTPException(status_code=400, detail="value is required. range is [0, 3]")
    sender: Sender = app.state.sender
    sender.peer_id = value
    
    return {
        "message": "receiver_id updated",
        "receiver_id": value
    }

@app.post("/record")
def record(duration: int = 10):
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    with sender_lock:
        app.state.last_audio_file = sender.record(duration=duration)

@app.post("/record_reference")
def record_reference():
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    with sender_lock:
        app.state.last_audio_file = sender.record()
    
    # 송신: self.user_id와 wav 파일 보내기
    return f"successfully send reference voice"

@app.post("/send")
def send():
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    with sender_lock:
        sample_rate, audio = read(app.state.last_audio_file)

        if not isinstance(audio, np.ndarray):
            raise TypeError(f"recorded audio is not ndarray: {type(audio)}")
        if audio.dtype != np.float32:
            raise TypeError(f"recorded audio type is not float32: {audio.dtype}")
        if audio.ndim != 1:
            raise ValueError(f"not mono audio: {audio.ndim}")

        result = sender.encoder.encode(audio)

        sender.send(
            result.text,
            result.emotion_label,
            result.emotion_indices
        )
    
    return f"successfully send yout message"