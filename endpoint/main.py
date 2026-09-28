"""Serve the sender and receiver features from one endpoint application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import PROJECT_ROOT
from grpc_getting_started.server_communicate_sender import SendVoice
from receiver import Receiver
from sender import Sender

_STORAGE = PROJECT_ROOT / "storage"
_STORAGE.mkdir(parents=True, exist_ok=True)
_FRONTEND_DIST = PROJECT_ROOT.parent / "tone" / "dist"


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    with (
        Sender(
            storage=_STORAGE,
            user_id=1,
            receiver_id=2,
            fsq_path=str(PROJECT_ROOT / "sender_models" / "skip_kl_8d_8L_kl05_1e-4.pt"),
        ) as sender,
        Receiver(storage=_STORAGE, user_id=1, sender_id=1) as receiver,
    ):
        application.state.sender = sender
        application.state.receiver = receiver
        application.state.sender_lock = Lock()
        application.state.last_audio_file = None
        yield


app = FastAPI(lifespan=_lifespan)
app.mount("/storage", StaticFiles(directory=_STORAGE), name="storage")


@app.get("/")
def _spa_root() -> FileResponse:
    return FileResponse(_FRONTEND_DIST / "index.html")


if _FRONTEND_DIST.exists():
    app.mount(
        "/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets"
    )

    @app.get("/{full_path:path}")
    def _spa_fallback(full_path: str) -> FileResponse:
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")


@app.post("/set_my_id")
def set_my_id(
    value: int | None = None, my_id: int | None = None
) -> list[list[dict[str, object]]]:
    """Set the local user ID for both roles and return pending messages."""
    selected_id = value if value is not None else my_id
    if selected_id is None:
        raise HTTPException(
            status_code=400, detail="value is required. range is [0, 3]"
        )

    sender: Sender = app.state.sender
    receiver: Receiver = app.state.receiver
    with app.state.sender_lock:
        sender.user_id = selected_id
        receiver.user_id = selected_id
        return receiver.get_pending_messages()


@app.post("/set_receiver_id")
def set_receiver_id(value: int | None = None) -> dict[str, str | int]:
    """Select the peer to receive the next message."""
    if value is None:
        raise HTTPException(
            status_code=400, detail="value is required. range is [0, 3]"
        )
    sender: Sender = app.state.sender
    with app.state.sender_lock:
        if value == sender.user_id:
            raise ValueError("Sender ID and receiver ID cannot be the same")
        sender.peer_id = value
    return {"message": "receiver_id updated", "receiver_id": value}


@app.post("/start_recording")
def start_recording() -> dict[str, str]:
    """Start recording through the endpoint's microphone."""
    sender: Sender = app.state.sender
    with app.state.sender_lock:
        sender.recoder.start_recording()
    return {"status": "recording start"}


@app.post("/stop_recording")
def stop_recording() -> dict[str, str | float]:
    """Stop recording and retain the encoded message for sending."""
    sender: Sender = app.state.sender
    with app.state.sender_lock:
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
def send(message: str | None = None) -> dict[str, str]:
    """Send the encoded message to the selected peer."""
    sender: Sender = app.state.sender
    with app.state.sender_lock:
        if message is None:
            if sender.temp_result is None:
                raise HTTPException(
                    status_code=400, detail="no encoded message available"
                )
            message = sender.temp_result.text
        sent_message = sender.send(message)
    return {"message": "sent", "text": sent_message}


@app.post("/send_ref")
def send_ref() -> dict[str, str]:
    """Upload the just-recorded reference voice to the relay."""
    sender: Sender = app.state.sender
    with app.state.sender_lock:
        _result, _duration, filepath = sender.recoder.stop_recording(encording=False)
        sender_id = str(sender.user_id)
    SendVoice(sender_id, filepath)
    return {"message": "reference sent", "file": str(filepath)}


@app.post("/get_emotion_label")
def get_emotion_label() -> dict[str, str | float]:
    """Return the label and score of the last encoded recording."""
    sender: Sender = app.state.sender
    encoded_result = sender.temp_result
    if not encoded_result:
        raise HTTPException(status_code=400, detail="no encoded result found")
    return {
        "emotion_label": encoded_result.emotion_label.name.lower(),
        "emotion_score": encoded_result.emotion_score,
    }


@app.post("/play_voice")
def play_voice(message_id: str | None = None) -> bool:
    """Fetch and play a received message through the endpoint's speaker."""
    if message_id is None:
        raise HTTPException(status_code=400, detail="message id is required.")
    receiver: Receiver = app.state.receiver
    return receiver.play_voice(message_id)
