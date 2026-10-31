"""Encode and send messages from a local endpoint."""

from pathlib import Path

import config  # noqa: F401 - 저장소 루트를 sys.path에 추가
from grpc_getting_started.server_communicate_sender import SendPacket
from recorder import AudioRecorder
from src.utils.bitpacking import encode_packet
from tone_core.config import SenderConfig
from tone_core.sender import EncodeResult, SenderEncode
from user import User


class Sender(User):
    """Hold recording state and transmit encoded messages to the relay."""

    def __init__(
        self, storage: Path, user_id: int, receiver_id: int, fsq_path: str
    ) -> None:
        super().__init__(storage, user_id, receiver_id)
        self.recoder = AudioRecorder(
            storage=self.storage,
            encoder=SenderEncode.from_config(SenderConfig(fsq_path)),
        )
        self.temp_result: EncodeResult | None = None

    def build_packet(self, message: str) -> bytes:
        """현재 인코딩 결과와 메시지를 비트패킹 패킷으로 만든다."""
        if not self.temp_result:
            raise ValueError("recorded audio is not found")
        return encode_packet(
            send_id=self.user_id,
            receiver_id=self.peer_id,
            emo_type=int(self.temp_result.emotion_label),
            emotion_indices=list(self.temp_result.emotion_indices),
            message=message,
        )

    def send(self, message: str) -> str:
        """Send the current encoding as a bit-packed packet and clear it after a successful upload."""
        result: bool = SendPacket(self.build_packet(message))
        if not result:
            raise RuntimeError("send failed")

        # 메시지 중복 전송 방지
        self.temp_result = None

        return message
