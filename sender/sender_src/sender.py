from pathlib import Path
import time

from user import User
from recoder import AudioRecorder
from tone_core.sender import SenderEncode, EncodeResult
from tone_core.config import SenderConfig

from grpc_getting_started.server_communicate_sender import Send, SendVoice

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

class Sender(User):
    def __init__(self, storage: Path, user_id: int, receiver_id: int, server_ip: str, fsq_path: str):
        super().__init__(storage, user_id, receiver_id, server_ip)
        self.message_id: int = 0
        self.recoder = AudioRecorder(
            storage=self.storage, 
            encoder=SenderEncode.from_config(SenderConfig(fsq_path))
        )
        self.temp_result: EncodeResult | None = None

    def send(self, message: str):
        if not self.temp_result: raise ValueError("recorded audio is not found")
        Send(str(self.user_id), str(self.peer_id), message, int(self.temp_result.emotion_label), self.temp_result.emotion_indices)
        self.message_id = self.message_id + 1

        # 메시지 중복 전송 방지
        self.temp_result = None

        return message
    
    def send_voice(self, duration: int = 5):
        self.recoder.start_recording()
        time.sleep(duration)
        _result, _duration, filepath = self.recoder.stop_recording(encording=False)
        SendVoice(str(self.user_id), filepath)

        return filepath