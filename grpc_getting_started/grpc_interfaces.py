# 공통 인터페이스
from abc import ABC, abstractmethod
from collections.abc import Generator

class AbstractSpeechHandler(ABC):
    @abstractmethod
    def save_incoming_reference(self, sender_id: str, audio_content: bytes) -> bool:
        """송신측의 레퍼런스 음성 처리"""
        pass

    @abstractmethod
    def save_incoming_speech(self, sender_id: str, receiver_id: str, message: str, emo_type: int, emotion_vector: list) -> str | None:
        """송신측의 데이터 처리. 저장한 메시지의 message_id를 반환하고, 실패하면 None"""
        pass

    @abstractmethod
    def get_message_metadata(self, message_id: str) -> dict | None:
        """message_id 하나의 메타데이터 (get_pending_metadata 항목과 같은 형식). 없으면 None"""
        pass

    @abstractmethod
    def get_pending_metadata(self, user_id: str) -> list:
        """수신측의 대기 메시지 전달"""
        pass

    @abstractmethod
    def generate_voice_stream(self, message_id: str) -> Generator[bytes, None, None]: # Generator[바이트를 생성, 입력 안 받음, 반환 안함]
        """메시지 ID에 해당하는 음성 바이트 스트림 반환"""
        pass