from argparse import ArgumentParser

from sender.config import SERVER_IP, SERVER_PORT, SENDER_ROOT, MESSEGE_ID, USER_ID
from sender.inference import speech2emovec, speech2text

import requests
import torch

audio_storage = SENDER_ROOT / "temp_storage"

def _parse_args():
    parser = ArgumentParser()

    parser.add_argument()

    return parser.parse_args

def _make_packet(user_id: str, message_id: str, message: str, emo_embed: str) -> dict[str, str]:
    # 추후 user_id와 message_id 검증 로직 필요.
    packet: dict[str, str] = {}

    packet["User_Id"] = user_id
    packet["Message_Id"] = message_id
    packet["Message"] = message
    packet["emo_embed"] = emo_embed

    return packet

def _record():
    text = speech2text()
    emovec = speech2emovec()
    return text, emovec

def send():
    text, emovec = _record()
    print("successfully recorded")
    packet = _make_packet(USER_ID, MESSEGE_ID, text, emovec)
    
    response = requests.post(
    f"{SERVER_IP}:{SERVER_PORT}/send",
    json=packet
    )
    print(response.status_code)
    print(response.text)

if __name__ == "__main__":
    args = _parse_args()

    send()