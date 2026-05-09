from argparse import ArgumentParser
from datetime import datetime

from sender.config import SERVER_IP, SERVER_PORT, SENDER_ROOT, MESSEGE_ID, USER_ID
from sender.inference import speech2emovec, speech2text

import sounddevice as sd
from scipy.io.wavfile import write
import requests

audio_storage = SENDER_ROOT / "storage"
audio_storage.mkdir(parents=True, exist_ok=True)

def _parse_args():
    parser = ArgumentParser()
    parser.add_argument()

    return parser.parse_args()

def _make_packet(user_id: str, message_id: str, message: str, emo_embed: str) -> dict[str, str]:
    # 추후 user_id와 message_id 검증 로직 필요.
    packet: dict[str, str] = {}

    packet["User_Id"] = user_id
    packet["Message_Id"] = message_id
    packet["Message"] = message
    packet["emo_embed"] = emo_embed

    return packet

def _record(save_dir=audio_storage, duration=5, sample_rate=44100, channels=1):
    # 파일 이름 생성
    filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
    file_path = save_dir / filename

    print("Recording started...")

    # 녹음
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="int16"
    )

    # 녹음이 끝날 때까지 대기
    sd.wait()

    # WAV 파일로 저장
    write(file_path, sample_rate, audio_data)
    print(f"Recording saved: {file_path}")

    return file_path

def _send(packet: dict[str, str]):
    response = requests.post(
    f"{SERVER_IP}:{SERVER_PORT}/send",
    json=packet
    )
    print(response.status_code)
    print(response.text)

def process():
    audio_file = _record(duration=10)

    if not (text := speech2text(str(audio_file))):
        print("STT에서 에러 발생")
        exit(1)
    if not (emovec := speech2emovec(str(audio_file))):
        print("emo2vec에서 에러 발생")
        exit(1)
    # emovec 압축 로직 필요

    packet = _make_packet(USER_ID, MESSEGE_ID, text, emovec)

    _send(packet=packet)

    return text, emovec


if __name__ == "__main__":
    process()
