from pathlib import Path
from datetime import datetime

from user import User
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
        self.encoder = SenderEncode.from_config(SenderConfig(fsq_path))
        self.temp_result: EncodeResult | None = None

    def record(self, duration=5, sample_rate=16000, channels=1):
        # 파일 이름 생성
        filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
        file_path = self.storage / filename

        print("Recording started...")

        # 녹음
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32"
        )

        # 녹음이 끝날 때까지 대기
        sd.wait()

        audio_data = audio_data.flatten()
        if not isinstance(audio_data, np.ndarray):
            raise TypeError(f"recorded audio is not ndarray: {type(audio_data)}")
        if audio_data.dtype != np.float32:
            raise TypeError(f"recorded audio type is not float32: {audio_data.dtype}")
        if audio_data.ndim != 1:
            raise ValueError(f"not mono audio: {audio_data.ndim}")

        result = self.encoder.encode(audio_data)
        self.temp_result = result

        # WAV 파일로 저장
        write(file_path, sample_rate, audio_data)
        print(f"Recording saved: {file_path}")

        return result

    def send(self, message: str):
        if not self.temp_result: raise ValueError("recorded audio is not found")
        Send(str(self.user_id), str(self.peer_id), message, int(self.temp_result.emotion_label), self.temp_result.emotion_indices)
        self.message_id = self.message_id + 1

        # 메시지 중복 전송 방지
        self.temp_result = None

        return message
    
    def send_voice(self, duration: int):
        filepath = self.record(duration=duration)
        SendVoice(str(self.user_id), filepath)

        self.temp_result = None

        return filepath