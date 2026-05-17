from pathlib import Path
from datetime import datetime

from user import User
from config import PROJECT_ROOT
from tone_core.sender import SenderEncode
from tone_core.config import SenderConfig

import requests
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read

class Sender(User):
    def __init__(self, storage: Path, user_id: int, receiver_id: int, server_ip: str, fsq_path: str):
        super().__init__(storage, user_id, receiver_id, server_ip)
        self.message_id: int = 0
        self.encoder = SenderEncode.from_config(SenderConfig(fsq_path))

    def _make_packet(self, message: str, emotion_vector: tuple[int, ...]) -> dict[str, str | int | tuple[int, ...]]:
        packet: dict[str, str | int | tuple[int, ...]] = {}

        packet["sender_Id"] = str(self.user_id)
        packet["receiver_Id"] = str(self.peer_id)
        packet["message_Id"] = str(self.message_id)
        packet["emo_type"] = 000000
        packet["message"] = message
        packet["emotion_vector"] = emotion_vector

        return packet

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

    def send(self, message: str, emotion_vector: tuple[int, ...]):
        packet = self._make_packet(message, emotion_vector)
        print(packet)
        # response = requests.post(f"http://{self.server_ip}/send", json=packet)
        # self.logger.info(f"response status: {response.status_code}")
        # self.logger.info(f"response text: {response.text}")
        # self.logger.info(f"Message Sent. Message_id: {self.message_id}")
        # self.message_id = self.message_id + 1

if __name__ == "__main__":
    with Sender(
        storage=PROJECT_ROOT / "storage",
        user_id=1,
        receiver_id=1,
        server_ip="127.0.0.1:8000",
        fsq_path=str(PROJECT_ROOT / "sender_models" / "skip_kl_8d_8L_kl05_1e-4.pt")
    ) as sender:
        audio_file = sender.record(duration=10)

        sample_rate, audio = read(audio_file)
        
        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
            raise TypeError(f"recorded audio type is not float 32: {type(audio)}")

        if not audio.ndim == 1:
            raise ValueError(f"not mono audio: {audio.ndim}")

        result = sender.encoder.encode(audio)

        packet = sender.send(result.text, result.emotion_indices)