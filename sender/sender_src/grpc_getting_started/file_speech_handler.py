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
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.mysql import insert
import numpy as np
from sqlalchemy import or_, and_

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()

users: dict[int, str] = {
    0: "종찬",
    1: "재웅",
    2: "경택",
    3: "태원"
}

class DBManager:
    def __init__(self):
        db_user = "root"
        db_password = os.environ.get("DB_PASSWORD")
        if not db_password: raise ValueError(f"db password not found")
        db_host = "localhost"
        db_port = "3306"
        db_name = "db"
        
        db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(db_url, pool_pre_ping=True, echo=True)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

    def _convert_path(self, path: str) -> str:
        base_path = "/home/cap/capstone_design_2026"

        relative_path = os.path.relpath(path, base_path)

        return "./" + relative_path

    """
    1. 새로운 유저를 등록하는 함수
    """
    def create_user(self, user_id: int, name: str, audio_path: str):
        with self.Session() as session:
            try:
                # 1. MySQL 전용 insert 구문을 생성합니다.
                stmt = insert(UserTable).values(
                    id=user_id,
                    user_name=name,
                    user_ref_audio_path=self._convert_path(audio_path)
                )

                # 2. 중복(Primary Key 또는 Unique Key 충돌) 발생 시 업데이트할 값을 지정합니다.
                stmt = stmt.on_duplicate_key_update(
                    user_name=stmt.inserted.user_name,
                    user_ref_audio_path=stmt.inserted.user_ref_audio_path
                )

                # 3. 쿼리를 실행하고 커밋합니다.
                session.execute(stmt)
                session.commit()
                print("✨ 유저 등록/업데이트 성공!")
            except SQLAlchemyError as e:
                """
                Rollback the session to clear the failed transaction.
                """
                session.rollback()
                print(f"❌ 유저 저장 중 에러 발생: {e}")
                raise e

    def save_chat(self, message_id: str, sender: int, receiver: int, msg: str, emo_path: str, emo_type: int):
        """
        Saves the chat data and explicitly raises any database errors.
        """
        with self.Session() as session:
            try:
                new_chat = ChatTable(
                    massage_id=message_id,
                    send_user_id=sender, 
                    rec_user_id=receiver,
                    massage=msg, 
                    emotion_path=self._convert_path(emo_path), 
                    emotion=emo_type
                )
                
                session.add(new_chat)
                session.commit()
                print("✨ DB commit 성공!")  # 정상 작동 확인용 임시 출력
                
            except SQLAlchemyError as e:
                """
                Rollback the session to clear the failed transaction
                and raise the error to see the full trace.
                """
                session.rollback()
                print(f"❌ DB 저장 중 에러 발생: {e}")
                raise e  # 상위 코드로 에러를 강제로 던져서 프로그램을 멈추고 트레이스백을 띄웁니다.

    def get_chats_by_user_id(self, user_id: int) -> list[list[dict]]:
        """
        Get all chat history between sender_id and i.
        """
        if not user_id in range(0, 4): raise ValueError(f"sender id must be in range(0, 4)")
        with self.Session() as session:
            chatrooms: list[list[dict]] = []
            
            for i in range(4):
                # 1. (sender_id, sender_id)인 경우는 제외함
                if i == user_id:
                    chatrooms.append([])
                    continue
                    
                # 2. sender_id와 i가 주고받은 대화 내역을 전부 가져오기
                chats: list[ChatTable] = (
                    session.query(ChatTable)
                    .filter(
                        or_(
                            (ChatTable.send_user_id == user_id) & (ChatTable.rec_user_id == i),
                            (ChatTable.send_user_id == i) & (ChatTable.rec_user_id == user_id)
                        )
                    )
                    .order_by(ChatTable.updated_at.asc())
                    .all()
                )

                """
                조회된 ChatTable 객체들을 to_dict()를 활용해 딕셔너리로 변환합니다.
                """
                dict_chats = [chat.to_dict() for chat in chats]
                
                """
                최종 리스트(2차원)에 변환된 딕셔너리 리스트를 추가합니다.
                """
                chatrooms.append(dict_chats)
                
            return chatrooms
        
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
            fpath = self.emotion_folder / f"{unique_msg_id}.npy"

            decoded_emotion_vector = self.decoder.decode(emotion_vector)

            np.save(fpath, decoded_emotion_vector)

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
            print(f"{e}")
            return False
    
    def get_pending_metadata(self, user_id: str) -> list[list[dict]]:
        int_user_id = int(user_id)
        
        try:
        # 1. DBManager를 통해 특정 유저의 ChatTable 객체 리스트를 가져옵니다.
            chatrooms: list[list[dict]] = self.dbmanager.get_chats_by_user_id(int_user_id)
        except Exception as e:
            raise Exception(f"db logic: {e}")
        
        return chatrooms
            
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
            "emotion_npy_path" : chat_row.emotion_path
        }

        with requests.post(API_URL, json=payload, stream=True) as r:
            # HTTP 상태 코드가 200(정상)이 아니면 에러를 발생
            try :
                r.raise_for_status()
            except :
                print(r.content)
            else :
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        yield chunk