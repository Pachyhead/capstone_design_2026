import json
from datetime import datetime

def doSomething(request): # 받은 SpeechUploadRequest 내용을 수신자id.json으로 서버에 저장
    try:
        fname = f"{request.receiver_id}.json"

        with open(fname, "w", encoding="utf-8") as f:
            f.write("[\n")
            f.write(f"\t{{\n")
            f.write(f"\t\t\"message_id\": \"000001\",\n")
            f.write(f"\t\t\"sender_id\": \"{request.sender_id}\",\n")
            f.write(f"\t\t\"message\": \"{request.message}\",\n")
            f.write(f"\t\t\"emo_type\": {request.emo_type},\n")
            f.write(f"\t\t\"send_time\": \"{datetime.now()}\"\n")
            f.write(f"\t}}\n")
            f.write("]")
            return True
    except OSError as e:
        print(f"File write error: {e}")
        return False

def getVoice(request): # 받은 MessageIdentifier 정보에 해당하는 wav를 찾아 바이트 형태로 반환
    fname = f"../{request.message_id}.wav"

    with open(fname, "rb") as f:
        wav_bytes = f.read()

    yield wav_bytes

def getMetadataList(user_id): # 받은 UserIdentifier 정보에 해당하는 데이터를 수신자id.json에서 가져옴
    fname = f"{user_id}.json"

    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data