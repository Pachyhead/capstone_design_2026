from pathlib import Path
from datetime import datetime

from user import User
from config import PROJECT_ROOT
from tone_core.sender import SenderEncode
from tone_core.config import SenderConfig

from grpc_getting_started.server_communicate_sender import Send, SendVoice

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from fastapi import FastAPI

class Sender(User):
    def __init__(self, storage: Path, user_id: int, receiver_id: int, server_ip: str, fsq_path: str):
        super().__init__(storage, user_id, receiver_id, server_ip)
        self.message_id: int = 0
        self.encoder = SenderEncode.from_config(SenderConfig(fsq_path))

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

        # WAV 파일로 저장
        write(file_path, sample_rate, audio_data)
        print(f"Recording saved: {file_path}")

        return file_path

    def send(self, message: str, emotion_type, emotion_vector: tuple[int, ...]):
        Send(str(self.user_id), str(self.peer_id), message, int(emotion_type), emotion_vector)
        self.message_id = self.message_id + 1
        return message
        # response = requests.post(f"http://{self.server_ip}/send", json=packet)
        # self.logger.info(f"response status: {response.status_code}")
        # self.logger.info(f"response text: {response.text}")
        # self.logger.info(f"Message Sent. Message_id: {self.message_id}")

    def send_voice(self, duration: int):
        filepath = self.record(duration=duration)
        SendVoice(str(self.user_id), filepath)

        return filepath