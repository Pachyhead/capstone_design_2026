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
def set_my_id(value: int | None = None):
    if value is None:
        raise HTTPException(status_code=400, detail="value is required. range is [0, 3]")
    sender: Sender = app.state.sender
    sender.user_id = value
    return {"message": "user_id updated", "user_id": value}


@app.post("/set_receiver_id")
def set_receiver_id(value: int | None = None):
    if value is None:
        raise HTTPException(status_code=400, detail="value is required. range is [0, 3]")
    sender: Sender = app.state.sender
    sender.peer_id = value
    return {"message": "receiver_id updated", "receiver_id": value}


@app.post("/start_recording")
def start_recording():
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    with sender_lock:
        sender.recoder.start_recording()

    return {"status": "recording start"}


@app.post("/stop_recording")
def stop_recording():
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    with sender_lock:
        result, duration, file_path = sender.recoder.stop_recording(encording=True)
        sender.temp_result = result
        app.state.last_audio_file = str(file_path)

    if not result:
        raise HTTPException(status_code=500, detail="encoding failed")

    return {
        "text": result.text,
        "emotion": result.emotion_label.name.lower(),
        "duration": round(duration, 1),
        "audio_url": f"/storage/{file_path.name}",
    }


@app.post("/send")
def send(message: str | None = None):
    sender: Sender = app.state.sender
    sender_lock: Lock = app.state.sender_lock

    if message is None:
        if sender.temp_result is None:
            raise HTTPException(status_code=400, detail="no encoded message available")
        message = sender.temp_result.text

    with sender_lock:
        sent_message = sender.send(message)

    print(f"successfully sent your message: {sent_message}")
    return {"message": "sent", "text": sent_message}


@app.post("/send_ref")
def send_ref(duration: int = 5):
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
        raise HTTPException(status_code=400, detail="no encoded result found")

    return {
        "emotion_label": encoded_result.emotion_label.name.lower(),
        "emotion_score": encoded_result.emotion_score,
    }
