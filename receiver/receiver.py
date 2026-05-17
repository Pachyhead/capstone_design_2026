from pathlib import Path
from datetime import datetime
import json

from user import User
from config import PROJECT_ROOT

import requests

class Receiver(User):
    def __init__(self, storage: Path, user_id: int, receiver_id: int, server_ip: str):
        super().__init__(storage, user_id, receiver_id, server_ip)
        self.message_id: int = self.get_pending_messages()

    def update_sender(self, sender_id: int):
        self.peer_id = sender_id
    
    def get_pending_messages(self) -> int:
        response = requests.get(f"http://{self.server_ip}/get_pending_messages", timeout=5)
        response.raise_for_status()

        datas: list[dict] = response.json()["data"]

        cur_message_id = -1
        for data in datas:
            cur_message_id = cur_message_id + 1
            file_name = f"message_{cur_message_id}.json"
            with open(self.storage / file_name, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        return cur_message_id

    def _get_voice(self, message_id: int) -> Path:
        if message_id < 0: raise ValueError("message_id must be non-negative")
        if message_id > self.message_id: raise ValueError(f"such message is not received yet")
        response = requests.get(
            f"http://{self.server_ip}/get_voice",
            params={"message_id": message_id},
            timeout=5
        )
        response.raise_for_status()

        file_path = self.storage / f"voice_{message_id}.wav"
        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path
    
    def play_voice(self, message_id: int):
        import subprocess
        file = self._get_voice(message_id=message_id)
        subprocess.run(["afplay", f"{file}"])

if __name__ == "__main__":
    with Receiver(
        storage=PROJECT_ROOT / "storage",
        user_id=1,
        receiver_id=1,
        server_ip="127.0.0.1:8000",
    ) as receiver:

        receiver.play_voice(message_id=5)