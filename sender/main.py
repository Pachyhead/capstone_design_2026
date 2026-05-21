from contextlib import asynccontextmanager
from threading import Lock

from sender import Sender
from config import PROJECT_ROOT

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

STORAGE = PROJECT_ROOT / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with Sender(
        storage=STORAGE,
        user_id=1,
        receiver_id=1,
        server_ip="127.0.0.1:8000",
        fsq_path=str(PROJECT_ROOT / "sender_models" / "skip_kl_8d_8L_kl05_1e-4.pt"),
    ) as sender:
        app.state.sender = sender
        app.state.sender_lock = Lock()
        app.state.last_audio_file = None
        yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory=STORAGE), name="storage")


@app.post("/set_my_id")
def set_user_id(value: int | None = None):
    if value is None:
        raise HTTPException(status_code=400, detail="value is required. range is [0, 3]")
    sender: Sender = app.state.sender
    sender.user_id = value
    return {"message": "user_id updated", "user_id": value}


@app.post("/set_receiver_id")
def set_target_id(value: int | None = None):
    if value is None:
        raise HTTPException(status_code=400, detail="value is required. range is [0, 3]")
    sender: Sender = app.state.sender
    sender.peer_id = value
    return {"message": "receiver_id updated", "receiver_id": value}


@app.post("/record")
def record(duration: int = 10):
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    with sender_lock:
        result = sender.record(duration=duration)

    return {
        "text": result.text,
        "emotion": result.emotion_label,
        "duration": duration,
    }


@app.post("/send")
def send(message: str | None = None):
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock
    if message is None:
        message = sender.temp_result.text

    with sender_lock:
        message = sender.send(message)

    print(f"successfully send yout message: {message}")
    return {"message": "sent", "text": message}


@app.post("/send_ref")
def send_voice(duration: int = 5):
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    with sender_lock:
        filepath = sender.send_voice(duration)

    print(f"successfully sent wav file: {filepath}")
    return {"message": "reference sent", "file": str(filepath)}


@app.post("/get_emotion_label")
def get_emotion_label():
    sender: Sender = app.state.sender
    encoded_result = sender.temp_result
    if not encoded_result:
        raise ValueError("Encoded Result not found")

    return {
        "emotion_label": encoded_result.emotion_label,
        "emotion_score": encoded_result.emotion_score,
    }
