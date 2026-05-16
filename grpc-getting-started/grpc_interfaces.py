# 공통 인터페이스
from abc import ABC, abstractmethod

class AbstractSpeechHandler(ABC):
    @abstractmethod
    def save_incoming_speech(self, sender_id: str, receiver_id: str, message: str, emo_type: int, emotion_vector: list) -> bool:
        """송신측의 데이터 처리 (현재는 json으로 저장 -> 추후 AudoEncoder/DB 연동)"""
        pass
    
    @abstractmethod
    def get_pending_metadata(self, user_id: str) -> list:
        """수신측의 대기 메시지 전달"""
        pass

    @abstractmethod
    def generate_voice_stream(self, message_id: str):
        """메시지 ID에 해당하는 음성 바이트 스트림 반환"""
        pass