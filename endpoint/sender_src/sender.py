"""Encode and send messages from a local endpoint."""

from pathlib import Path

from grpc_getting_started.server_communicate_sender import Send
from recoder import AudioRecorder
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

    def send(self, message: str) -> str:
        """Send the current encoding and clear it after a successful upload."""
        if not self.temp_result:
            raise ValueError("recorded audio is not found")
        result: bool = Send(
            str(self.user_id),
            str(self.peer_id),
            message,
            int(self.temp_result.emotion_label),
            self.temp_result.emotion_indices,
        )
        if not result:
            raise RuntimeError("send failed")

        # 메시지 중복 전송 방지
        self.temp_result = None

        return message
