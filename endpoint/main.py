"""Serve the sender and receiver features from one endpoint application."""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from threading import Event, Lock

import grpc
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import PROJECT_ROOT
from grpc_getting_started.server_communicate_connect import close_connection
from grpc_getting_started.server_communicate_sender import SendVoice
from inbox_stream import Broadcaster, InboxSubscriber
from receiver import Receiver
from sender import Sender
from utils import measure_audio_codec_sizes
from grpc_getting_started import server_communicate_pb2

_STORAGE = PROJECT_ROOT / "storage"
_STORAGE.mkdir(parents=True, exist_ok=True)
_FRONTEND_DIST = PROJECT_ROOT.parent / "tone" / "dist"

# 종료 신호(Ctrl+C)를 받으면 설정된다. uvicorn은 열린 연결이 끝나야 종료하므로
# SSE(/events) 응답이 이 값을 보고 스스로 끝나게 한다.
_SHUTTING_DOWN = Event()
_SSE_PING_SECONDS = 15  # 브라우저 연결 확인용 주석 이벤트 주기

try:
    from uvicorn.server import Server as _UvicornServer

    _uvicorn_handle_exit = _UvicornServer.handle_exit

    def _handle_exit(self, sig, frame):  # noqa: ANN001
        _SHUTTING_DOWN.set()
        _uvicorn_handle_exit(self, sig, frame)

    _UvicornServer.handle_exit = _handle_exit
except ImportError:  # uvicorn 없이 import하는 경우 (테스트 등)
    pass


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
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
        # 화면이 /set_my_id, /set_receiver_id를 보냈는지. 엔드포인트를 재시작하면
        # 위의 기본값(사용자 1, 수신자 2)으로 돌아가므로, 화면이 다시 알려 주기 전에는
        # 녹음·전송을 받지 않는다 (_require_ids).
        application.state.user_set = False
        application.state.peer_set = False

        # 새 메시지 실시간 수신: 서버 구독 스트림 -> 브라우저 SSE (/events)
        broadcaster = Broadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        subscriber = InboxSubscriber(broadcaster)
        subscriber.start()  # /set_my_id로 사용자가 정해지면 구독을 시작한다
        application.state.broadcaster = broadcaster
        application.state.inbox_subscriber = subscriber
        try:
            yield
        finally:
            _SHUTTING_DOWN.set()
            subscriber.stop()  # 구독 스트림을 먼저 끊고
            close_connection()  # 공유 gRPC 채널 종료


app = FastAPI(lifespan=_lifespan)
app.mount("/storage", StaticFiles(directory=_STORAGE), name="storage")

# 서버에 연결할 수 없거나 시간 초과면 503, 그 밖의 gRPC 오류는 502로 응답한다
_RELAY_UNREACHABLE = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}


@app.exception_handler(grpc.RpcError)
async def _grpc_error_handler(request: Request, exc: grpc.RpcError) -> JSONResponse:
    code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.UNKNOWN
    details = exc.details() if hasattr(exc, "details") else ""
    if code in _RELAY_UNREACHABLE:
        return JSONResponse(
            status_code=503,
            content={"detail": f"relay server unavailable ({code.name})", "grpc_details": details},
        )
    return JSONResponse(
        status_code=502,
        content={"detail": f"relay server error ({code.name})", "grpc_details": details},
    )


def _require_ids(*, peer: bool = False) -> None:
    """사용자 id(필요하면 수신자 id)를 아직 받지 못했으면 409로 거부한다.

    화면은 409를 받으면 id를 다시 보낸다 (tone/src/lib/api.ts).
    """
    if not app.state.user_set:
        raise HTTPException(status_code=409, detail="user id not set. call /set_my_id first")
    if peer and not app.state.peer_set:
        raise HTTPException(
            status_code=409, detail="receiver id not set. call /set_receiver_id first"
        )


@app.get("/events")
async def events(request: Request) -> StreamingResponse:
    """새 메시지를 Server-Sent Events로 보낸다 (event: message | resync).

    아래 SPA fallback(GET /{full_path:path})보다 먼저 등록해야 한다.
    """
    broadcaster: Broadcaster = app.state.broadcaster
    queue = broadcaster.connect()

    async def stream() -> AsyncGenerator[str, None]:
        try:
            # 끊기면 브라우저가 3초 뒤 다시 접속한다. 다시 접속한 화면은 스스로
            # /set_my_id, /set_receiver_id를 다시 보내므로 여기서 resync는 보내지 않는다.
            yield "retry: 3000\n\n"
            idle = 0.0
            while not _SHUTTING_DOWN.is_set():
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    idle += 1.0
                    if await request.is_disconnected():
                        break
                    if idle >= _SSE_PING_SECONDS:
                        idle = 0.0
                        yield ": ping\n\n"
                    continue
                idle = 0.0
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            broadcaster.disconnect(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
def set_my_id(value: int | None = None) -> list[list[dict[str, object]]]:
    """Set the local user ID for both roles and return pending messages."""
    if value is None:
        raise HTTPException(
            status_code=400, detail="value is required. range is [0, 3]"
        )

    sender: Sender = app.state.sender
    receiver: Receiver = app.state.receiver
    with app.state.sender_lock:
        sender.user_id = value
        receiver.user_id = value
        app.state.user_set = True
        # 목록 조회가 실패해도 실시간 수신은 새 사용자로 바꿔 둔다 (재접속하면서 resync)
        app.state.inbox_subscriber.switch_user(value)
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
        # 사용자 id를 받기 전의 user_id는 기본값이라 비교하지 않는다
        # (화면을 새로 열면 /set_receiver_id가 /set_my_id보다 먼저 올 수 있음)
        if app.state.user_set and value == sender.user_id:
            raise ValueError("Sender ID and receiver ID cannot be the same")
        sender.peer_id = value
        app.state.peer_set = True
    return {"message": "receiver_id updated", "receiver_id": value}


@app.post("/start_recording")
def start_recording() -> dict[str, str]:
    """Start recording through the endpoint's microphone."""
    _require_ids()
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

    payload = server_communicate_pb2.SpeechUploadRequest(
        sender_id=str(sender.user_id),
        receiver_id=str(sender.peer_id),
        message=result.text,
        emo_type=result.emotion_label_idx,
        emotion_vector=result.emotion_indices
    )

    return {
        "text": result.text,
        "emotion": result.emotion_label.name.lower(),
        "duration": round(duration, 1),
        "audio_url": f"/storage/{file_path.name}",
        "payload_bytes": payload.ByteSize()
    }


@app.post("/send")
def send(message: str | None = None) -> dict[str, str]:
    """Send the encoded message to the selected peer."""
    _require_ids(peer=True)
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
    _require_ids()
    sender: Sender = app.state.sender
    with app.state.sender_lock:
        _result, _duration, filepath = sender.recoder.stop_recording(encording=False)
        sender_id = str(sender.user_id)
    if not SendVoice(sender_id, filepath):
        raise HTTPException(status_code=500, detail="not found ref audio file")

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
    receiver.play_voice(message_id)
        
    return True


@app.post("/measure_audio_codec_sizes")
def measure_latest_audio_codec_sizes() -> dict[str, object]:
    """Measure codec sizes for the newest WAV file in endpoint storage."""
    latest_wav = max(
        _STORAGE.glob("*.wav"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        default=None,
    )
    if latest_wav is None:
        raise HTTPException(status_code=404, detail="No WAV file found in storage")
    return measure_audio_codec_sizes(str(latest_wav))
