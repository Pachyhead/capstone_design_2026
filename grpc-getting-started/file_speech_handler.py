import json
import os
from datetime import datetime
from grpc_interfaces import AbstractSpeechHandler

class FileSpeechHandler(AbstractSpeechHandler):
    # 송신측이 보낸 데이터를 수신자id.json으로 서버에 저장
    def save_incoming_speech(self, sender_id, receiver_id, message, emo_type, emotion_vector) -> bool:
        try:
            fname = f"{receiver_id}.json"

            if os.path.exists(fname): # 파일이 존재하는 경우
                with open(fname, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f) # 기존 데이터 읽어옴
                    except json.JSONDecodeError:
                        data = [] # 파일이 비어있거나 에러 나면 초기화
            else: # 파일이 존재하지 않는 경우
                data = [] # 빈 리스트로 시작
            
            new_message = { # 받은 내용 message로 만듦
                "message_id": "000001",
                "sender_id": sender_id,
                "message": message,
                "emo_type": emo_type,
                "send_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            }

            data.append(new_message) # 기존 데이터에 새 데이터 추가

            with open(fname, "w", encoding="utf-8") as f: # JSON으로 저장
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True
        except (OSError, Exception) as e:
            print(f"Error in file_speech_handler: {e}")
            return False
    
    # 수신자id.json에서 메시지를 가져옴
    def get_pending_metadata(self, user_id: str) -> list:
        fname = f"{user_id}.json"

        if not os.path.exists(fname):
            return []
        
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # message_id에 해당하는 wav를 찾아 바이트 형태로 반환
    def generate_voice_stream(self, message_id: str):
        fname = f"../{message_id}.wav"

        if not os.path.exists(fname):
            return
        
        chunk_size = 1024 * 64 # 64KB씩 쪼개서 전송
        with open(fname, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
