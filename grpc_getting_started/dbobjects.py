from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, BigInteger, ForeignKey, TIMESTAMP, func, DateTime, text

Base = declarative_base()

"""
사용자 정보 테이블 매핑 클래스
"""
class UserTable(Base):
    __tablename__ = 'USER_TABLE'

    id = Column(BigInteger, primary_key=True, nullable=False)
    user_name = Column(String(255), nullable=False)
    user_ref_audio_path = Column(String(255), nullable=False)


"""
채팅 테이블 매핑 클래스
"""
class ChatTable(Base):
    __tablename__ = 'CHAT_TABLE'

    massage_id = Column(String(255), primary_key=True, nullable=False)
    
    # 외래키(Foreign Key) 설정 및 CASCADE 연동
    send_user_id = Column(
        BigInteger, 
        ForeignKey('USER_TABLE.id', ondelete='CASCADE'), 
        nullable=False
    )
    rec_user_id = Column(
        BigInteger, 
        ForeignKey('USER_TABLE.id', ondelete='CASCADE'), 
        nullable=False
    )
    
    massage = Column(String(255), nullable=False)
    emotion_path = Column(String(255), nullable=False)
    emotion = Column(Integer, nullable=False)

    updated_at = Column(
        DateTime, 
        server_default=text('CURRENT_TIMESTAMP'), 
        onupdate=text('CURRENT_TIMESTAMP'), 
        nullable=False
    )

    def to_dict(self) -> dict:
        """
        Convert this model instance into a dictionary structure.
        """
        return {
            "message_id": self.massage_id,
            "sender_id": self.send_user_id,
            "receiver_id": self.rec_user_id,
            "message": self.massage,
            "emotion_path": self.emotion_path,
            "emo_type": self.emotion,
            "send_time": self.updated_at.strftime("%Y-%m-%d %H:%M:%S.%f") if getattr(self, 'updated_at', None) else None
        }