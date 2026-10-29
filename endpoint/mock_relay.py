"""Run a fake SpeechRelay server so the endpoint can be tested without the GPU server.

Everything lives in memory: no MySQL, no TTS, no FSQ decoder. GetVoice streams a
WAV in 64 KB frames, the same way the real relay forwards the TTS response.

    cd endpoint
    uv run python mock_relay.py              # 2 s, 440 Hz tone
    uv run python mock_relay.py --wav a.wav  # stream this file instead

The port comes from SERV_PORT in endpoint/.env (default 50051), so the endpoint
connects to it with SERV_IP=127.0.0.1.
"""

import argparse
import os
import struct
import uuid
from concurrent import futures
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import grpc
import numpy as np
from dotenv import load_dotenv
from google.protobuf.timestamp_pb2 import Timestamp

from grpc_getting_started import server_communicate_pb2 as pb2
from grpc_getting_started import server_communicate_pb2_grpc as pb2_grpc

_USER_IDS = range(4)
_VALID_IDS = {str(i) for i in _USER_IDS}
_CHUNK_SIZE = 1024 * 64  # 실제 relay의 iter_content(chunk_size=1024 * 64)와 같다


def _tone_wav(seconds: float = 2.0, sample_rate: int = 24000) -> bytes:
    """Build a streaming WAV shaped like the TTS /stream response."""
    # TTS의 _streaming_wav_header처럼 RIFF/data 크기를 0xFFFFFFFF로 둔다
    header = (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    pcm = (np.sin(2 * np.pi * 440.0 * t) * 8000).astype("<i2").tobytes()
    return header + pcm


class MockRelay(pb2_grpc.SpeechRelayServicer):
    """Answer the endpoint's RPCs from in-memory state."""

    def __init__(self, wav: bytes) -> None:
        self.wav = wav
        self.lock = Lock()
        self.messages: list[dict] = []
        self.ref_sizes: dict[str, int] = {}

        # 어떤 프로필을 골라도 목록과 재생을 바로 시험할 수 있도록 쌍마다 하나씩
        start = datetime.now(timezone.utc) - timedelta(minutes=10)
        pairs = [(a, b) for a in _USER_IDS for b in _USER_IDS if a < b]
        for i, (a, b) in enumerate(pairs):
            text = f"{a}번이 {b}번에게 보낸 테스트 메시지"
            self._add(a, b, text, i % 8, start + timedelta(minutes=i))

    def _add(
        self, sender: int, receiver: int, text: str, emo_type: int, send_time: datetime
    ) -> str:
        message_id = str(uuid.uuid4())
        self.messages.append(
            {
                "message_id": message_id,
                "sender_id": sender,
                "receiver_id": receiver,
                "message": text,
                "emo_type": emo_type,
                "send_time": send_time,
            }
        )
        return message_id

    def SendVoice(self, request, context):
        with self.lock:
            self.ref_sizes[request.sender_id] = len(request.audio_content)
        print(f"[SendVoice] sender={request.sender_id} {len(request.audio_content):,} bytes")
        return pb2.UploadStatus(accepted=True)

    def Send(self, request, context):
        indices = list(request.emotion_vector)
        print(
            f"[Send] {request.sender_id} -> {request.receiver_id} "
            f"emo_type={request.emo_type} indices={indices} text={request.message!r}"
        )
        # 실제 relay에서 ID 변환이나 FSQ 디코딩이 실패하면 False가 된다
        ids_ok = {request.sender_id, request.receiver_id} <= _VALID_IDS
        if not ids_ok or len(indices) != 8 or not all(0 <= v <= 7 for v in indices):
            print("[Send] rejected")
            return pb2.UploadStatus(accepted=False)

        with self.lock:
            message_id = self._add(
                int(request.sender_id),
                int(request.receiver_id),
                request.message,
                request.emo_type,
                datetime.now(timezone.utc),
            )
        print(f"[Send] saved as {message_id}")
        return pb2.UploadStatus(accepted=True)

    def GetPendingMessages(self, request, context):
        if request.user_id not in _VALID_IDS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id must be in range(0, 4)")
        user = int(request.user_id)
        with self.lock:
            messages = sorted(self.messages, key=lambda m: m["send_time"])

        # DBManager.get_chats_by_user_id와 같은 모양: 상대 ID 순서로 4칸, 자기 자리는 빈 칸
        lists = []
        for peer in _USER_IDS:
            items = []
            if peer != user:
                items = [
                    self._item(m)
                    for m in messages
                    if {m["sender_id"], m["receiver_id"]} == {user, peer}
                ]
            lists.append(pb2.MetadataList(items=items))
        return pb2.MetadataResponse(lists=lists)

    def GetVoice(self, request, context):
        with self.lock:
            known = any(m["message_id"] == request.message_id for m in self.messages)
        if not known:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown message_id {request.message_id}")

        print(f"[GetVoice] {request.message_id}: {len(self.wav):,} bytes")
        for offset in range(0, len(self.wav), _CHUNK_SIZE):
            yield pb2.AudioFrame(
                audio_content=self.wav[offset : offset + _CHUNK_SIZE],
                message_id=request.message_id,
                is_final=False,
            )
        yield pb2.AudioFrame(audio_content=b"", message_id=request.message_id, is_final=True)

    @staticmethod
    def _item(message: dict) -> pb2.MetadataItem:
        send_time = Timestamp()
        send_time.FromDatetime(message["send_time"])
        return pb2.MetadataItem(
            message_id=message["message_id"],
            sender_id=str(message["sender_id"]),
            receiver_id=str(message["receiver_id"]),
            message=message["message"],
            emo_type=message["emo_type"],
            send_time=send_time,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake SpeechRelay server for endpoint tests.")
    parser.add_argument("--wav", type=Path, help="WAV file for GetVoice (default: 2 s tone)")
    args = parser.parse_args()
    wav = args.wav.read_bytes() if args.wav else _tone_wav()

    load_dotenv()
    port = os.environ.get("SERV_PORT") or "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_SpeechRelayServicer_to_server(MockRelay(wav), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    print(f"Mock relay listening on 127.0.0.1:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
