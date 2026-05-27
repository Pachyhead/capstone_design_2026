from pathlib import Path
import time

from user import User
from recoder import AudioRecorder
from tone_core.sender import SenderEncode, EncodeResult
from tone_core.config import SenderConfig

from grpc_getting_started.server_communicate_sender import Send

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

class Sender(User):
    def __init__(self, storage: Path, user_id: int, receiver_id: int, fsq_path: str):
        super().__init__(storage, user_id, receiver_id)
        self.recoder = AudioRecorder(
            storage=self.storage, 
            encoder=SenderEncode.from_config(SenderConfig(fsq_path))
        )
        self.temp_result: EncodeResult | None = None

    def send(self, message: str):
        if not self.temp_result: raise ValueError("recorded audio is not found")
        result: bool = Send(str(self.user_id), str(self.peer_id), message, int(self.temp_result.emotion_label), self.temp_result.emotion_indices)
        if not result:
            raise Exception(f"send is not work")

        # 메시지 중복 전송 방지
        self.temp_result = None

        return message