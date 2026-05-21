from pathlib import Path
from datetime import datetime
import json

from user import User
from config import PROJECT_ROOT
from grpc_getting_started.server_communicate_receiver import GetPendingMessages, GetVoice, merge_wav_byte, save_message_to_json

class Receiver(User):
    def __init__(self, storage: Path, user_id: int, sender_id: int, server_ip: str):
        super().__init__(storage, user_id, sender_id, server_ip)
    
    def get_pending_messages(self) -> str:
        messages = GetPendingMessages(str(self.user_id))
        save_message_to_json(messages)
        self.message_id = len(messages)
            
        return f"Get Pending Message Success"

    def _get_voice(self, message_id: int) -> str:
        stream = GetVoice(f"{message_id:06d}", str(self.peer_id), str(self.user_id))
        merge_wav_byte(stream)

        return f"successfully received wav file"
    
    def play_voice(self, message_id: int):
        import subprocess
        file = self._get_voice(message_id=message_id)
        subprocess.run(["afplay", f"{file}"])