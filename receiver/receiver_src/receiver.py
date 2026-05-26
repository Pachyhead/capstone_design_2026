from pathlib import Path
from datetime import datetime
import json

from user import User
from config import PROJECT_ROOT
from grpc_getting_started.server_communicate_receiver import GetPendingMessages, GetVoice, merge_wav_byte, save_message_to_json
from speaker import AudioSpeaker

class Receiver(User):
    def __init__(self, storage: Path, user_id: int, sender_id: int):
        super().__init__(storage, user_id, sender_id)
        self.speaker = AudioSpeaker()
    
    def get_pending_messages(self) -> list[list[dict]]:
        messages: list[list[dict]] = GetPendingMessages(str(self.user_id))
        
        return messages

    def _get_voice(self, message_id: str) -> Path:
        stream, message_id = GetVoice(message_id)
        wav_path = merge_wav_byte(stream, self.storage, message_id)
        if not wav_path: raise ValueError(f"wav path is empty")

        return wav_path
    
    def play_voice(self, message_id: str) -> float:
        """Returns duration in seconds (0.0 on failure)."""
        file = self._get_voice(message_id)
        try:
            return self.speaker.play_wav(file)
        except Exception as e:
            print(f"Error during playback: {e}")
            self.speaker.stop_speaker()
            return 0.0