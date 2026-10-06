"""Receive relay messages and play their synthesized voices."""

from pathlib import Path

from grpc_getting_started.server_communicate_receiver import (
    GetPendingMessages,
    GetVoice,
    merge_wav_byte,
)
from speaker import AudioSpeaker
from user import User


class Receiver(User):
    """Handle the receiving role on a local endpoint."""

    _cleanup_pattern: str = "*.json"

    def __init__(self, storage: Path, user_id: int, sender_id: int) -> None:
        super().__init__(storage, user_id, sender_id)
        self.speaker = AudioSpeaker()

    def get_pending_messages(self) -> list[list[dict[str, object]]]:
        """Return message buckets for the active local user."""
        messages: list[list[dict[str, object]]] = GetPendingMessages(str(self.user_id))

        return messages

    def _get_voice(self, message_id: str) -> Path:
        stream, message_id = GetVoice(message_id)
        wav_path = merge_wav_byte(stream, self.storage, message_id)
        if not wav_path:
            raise ValueError("wav path is empty")

        return wav_path

    def play_voice(self, message_id: str) -> bool:
        """
        Fetch and play a voice message on the endpoint's speaker.
        - 스트리밍 재생 + 파일 저장 병렬 진행
        """
        # file = self._get_voice(message_id)
        try:
            audio_frames, _ = GetVoice(message_id)
            fpath = self.storage / f"{message_id}.wav"

            # self.speaker.play_wav(file)

            # 청크 도착 즉시 재생
            self.speaker.play_wav_streaming(audio_frames, fpath)
            
            return True
        except Exception as error:  # noqa: BLE001 - preserve the receiver's False result
            self.logger.error("Error during playback: %s", error)
            self.speaker.stop_speaker()
            return False
