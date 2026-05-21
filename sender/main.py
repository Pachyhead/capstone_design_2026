from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock

import sounddevice as sd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
STORAGE = Path(__file__).resolve().parent / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)


def _record_audio(duration: int) -> Path:
    filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
    file_path = STORAGE / filename
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    write(file_path, SAMPLE_RATE, audio)
    return file_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lock = Lock()
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
        raise HTTPException(status_code=400, detail="value is required")
    return {"message": "user_id updated", "user_id": value}


@app.post("/set_receiver_id")
def set_receiver_id(value: int | None = None):
    if value is None:
        raise HTTPException(status_code=400, detail="value is required")
    return {"message": "receiver_id updated", "receiver_id": value}


@app.post("/record")
def record(duration: int = 10):
    with app.state.lock:
        file_path = _record_audio(duration)
        app.state.last_audio_file = str(file_path)

    return {
        "text": "안녕 오늘 점심 같이 먹을래?",
        "emotion": "happy",
        "duration": duration,
        "audio_url": f"/storage/{file_path.name}",
    }


@app.post("/record_reference")
def record_reference():
    with app.state.lock:
        file_path = _record_audio(5)
        app.state.last_audio_file = str(file_path)
    return {"message": "reference recorded", "file": str(file_path)}


@app.post("/send")
def send():
    if app.state.last_audio_file is None:
        raise HTTPException(status_code=400, detail="no recording available. call /record first")
    return {"message": "sent", "file": app.state.last_audio_file}
