from argparse import ArgumentParser

from sender.config import SERVER_IP, SERVER_PORT, SENDER_ROOT
from sender.inference import speech2emovec, speech2text

import torch

audio_storage = SENDER_ROOT / "storage"

def _parse_args():
    parser = ArgumentParser()

    parser.add_argument()

    return parser.parse_args

def _make_packet(message: str, emovec: torch.Tensor) -> dict[str, str]:
    packet: dict[str, str] = {}

    return packet

def _record():
    text = speech2text()
    emovec = speech2emovec()
    return text, emovec

def send():
    text, emovec = _record()
    print("준비 완료")
    packet = _make_packet(text, emovec)
    pass

if __name__ == "__main__":
    args = _parse_args()

    send()