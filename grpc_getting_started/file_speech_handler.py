import os
from pathlib import Path
import hashlib
import uuid
from datetime import datetime
from collections.abc import Generator
import requests

from src.tone_core.receiver import ReceiverDecode
from src.tone_core.config import ReceiverConfig

from .grpc_interfaces import AbstractSpeechHandler
from .dbobjects import Base, UserTable, ChatTable
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv()

users: dict[int, str] = {
    0: "철수",
    1: "영희",
    2: "민수",
    3: "준호"
}

class DBManager:
    def __init__(self):
        db_user = "root"
        db_password = os.environ.get("DB_PASSWORD")
        if not db_password: raise ValueError(f"db password not found")
        db_host = "localhost"
        db_port = "8080"
        db_name = "db"
        
        db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(db_url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

    """
    1. 새로운 유저를 등록하는 함수
    """
    def create_user(self, user_id: int, name: str, audio_path: str):
        with self.Session() as session:
            new_user = UserTable(id=user_id, user_name=name, user_ref_audio_path=audio_path)
            session.add(new_user)
            session.commit()

    """
    2. 채팅 메시지를 저장하는 함수
    """
    def save_chat(self, message_id: str, sender: int, receiver: int, msg: str, emo_path: str, emo_type: int):
        """
        Generates a unique UUID for the message.
        Saves the chat data and returns the ID.
        """
        with self.Session() as session:
            
            new_chat = ChatTable(
                massage_id=message_id,
                send_user_id=sender, 
                rec_user_id=receiver,
                massage=msg, 
                emotion_path=emo_path, 
                emotion=emo_type
            )
            
            session.add(new_chat)
            session.commit()

    """
    3. 특정 유저가 보낸 모든 채팅 메시지를 가져오는 함수
    """
    def get_chats_by_sender(self, sender_id: int):
        with self.Session() as session:
            return session.query(ChatTable).filter(ChatTable.send_user_id == sender_id).all()
        
    def get_chat_by_id(self, message_id: str) -> ChatTable | None:
        with self.Session() as session:
            # .filter()로 조건을 걸고, .first()를 사용해 매칭되는 첫 번째 row 하나만 객체로 가져옵니다.
            return session.query(ChatTable).filter(ChatTable.massage_id == message_id).first()
        
    def get_user_by_id(self, user_id: int) -> UserTable | None:
        with self.Session() as session:
            # .filter()로 id 조건을 걸고, .first()를 통해 객체 하나만 가져옵니다.
            return session.query(UserTable).filter(UserTable.id == user_id).first()

class FileSpeechHandler(AbstractSpeechHandler):
    def __init__(self):
        ref_folder = PROJECT_ROOT / "DataBase" / "ref_audio"
        ref_folder.mkdir(exist_ok=True, parents=True)
        self.ref_folder = ref_folder
        
        emotion_folder = PROJECT_ROOT / "DataBase" / "emotion_vectors"
        emotion_folder.mkdir(exist_ok=True, parents=True)
        self.emotion_folder = emotion_folder
        
        self.dbmanager = DBManager()

        fsq_path = "/home/cap/data/models/skip_kl_8d_8L_kl05_1e-4.pt"
        self.decoder = ReceiverDecode.from_config(ReceiverConfig(fsq_path))

    # wav로 저장
    def save_incoming_reference(self, sender_id, audio_content) -> bool:
        try:
            fpath = self.ref_folder / f"{sender_id}_ref.wav"

            with open(fpath, "wb") as f:
                f.write(audio_content)

            int_sender_id = int(sender_id)

            self.dbmanager.create_user(
                int_sender_id, 
                users[int_sender_id], 
                str(fpath)
            )
            return True
        except OSError as e:
            raise OSError(f"File save error: {e}")

    # 송신측이 보낸 데이터를 수신자id.json으로 서버에 저장
    def save_incoming_speech(self, sender_id, receiver_id, message, emo_type, emotion_vector) -> bool:
        try:
            int_sender_id = int(sender_id)
            int_receiver_id = int(receiver_id)
            unique_msg_id = str(uuid.uuid4())
            fpath = self.emotion_folder / f"{unique_msg_id}.txt"

            decoded_emotion_vector = self.decoder.decode(emotion_vector)

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(decoded_emotion_vector)

            self.dbmanager.save_chat(
                message_id=unique_msg_id,
                sender=int_sender_id,
                receiver=int_receiver_id,
                msg=message,
                emo_path=str(fpath),
                emo_type=emo_type,
            )
            return True
        except (OSError, Exception) as e:
            return False
    
    def get_pending_metadata(self, user_id: str) -> list[dict]:
        int_sender_id = int(user_id)
        
        # 1. DBManager를 통해 특정 유저의 ChatTable 객체 리스트를 가져옵니다.
        rows: list[ChatTable] = self.dbmanager.get_chats_by_sender(int_sender_id)
        
        # 데이터가 없으면 ValueError를 발생시킵니다.
        if not rows: 
            return []
            
        # 2. ChatTable 객체 리스트를 순수한 list[dict] 형태로 변환합니다.
        result_list = []
        for row in rows:
            row_data = {
                "message_id": row.massage_id,  # 외부 JSON 규격과 맞추기 위해 message_id로 매핑
                "sender_id": row.send_user_id,
                "receiver_id": row.rec_user_id,
                "message": row.massage,
                "emotion_path": row.emotion_path,
                "emo_type": row.emotion,
                # datetime 객체는 나중에 gRPC나 JSON 통신 시 에러 방지를 위해 문자열로 미리 변환합니다.
                "send_time": row.updated_at.strftime("%Y-%m-%d %H:%M:%S.%f") if row.updated_at else None
            }
            result_list.append(row_data)
                    
        # 3. 딕셔너리들이 가득 담긴 파이썬 리스트를 최종 리턴합니다.
        return result_list
            
    # message_id에 해당하는 wav를 찾아 바이트 형태로 반환
    def generate_voice_stream(self, message_id: str) -> Generator[bytes, None, None]:
        chat_row: ChatTable | None = self.dbmanager.get_chat_by_id(message_id)
        if not chat_row: raise ValueError(f"Such Message ID Not Found")

        sender_id = chat_row.send_user_id
        user_row = self.dbmanager.get_user_by_id(sender_id) # type: ignore
        if not user_row: raise ValueError(f"Such User ID Not Found")

        API_URL = "http://localhost:8080/stream"

        payload = {
            "target_text" : chat_row.massage,
            "ref_audio" : user_row.user_ref_audio_path,
            "ref_text" : "어쨌든 우리한테 와서 건강하게 지금도 잘 자라고 있으니까",
            "use_emotion" : True,
            "emotion_npy_path" : chat_row.emotion_path,
        }

        with requests.post(API_URL, json=payload, stream=True) as r:
            # HTTP 상태 코드가 200(정상)이 아니면 에러를 발생
            r.raise_for_status()
            
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    yield chunk