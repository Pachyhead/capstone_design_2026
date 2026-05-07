import json

def doSomething(request): # 받은 SpeechUploadRequest 내용을 수신자id.json으로 서버에 저장
    try:
        fname = f"{request.receiver_id}.json"

        with open(fname, "w", encoding="utf-8") as f:
            f.write("{\n")
            f.write(f"\t\"sender_id\": \"{request.sender_id}\",\n")
            f.write(f"\t\"receiver_id\": \"{request.receiver_id}\",\n")
            f.write(f"\t\"msg_id\": \"{request.msg_id}\",\n")
            f.write(f"\t\"text\": \"{request.text}\",\n")
            f.write(f"\t\"emotion_vector\": {request.emotion_vector}\n")
            f.write("}")
            return True
    except OSError as e:
        print(f"File write error: {e}")
        return False

def getVoice(request): # 받은 MessageIdentifier 정보에 해당하는 wav를 찾아 바이트 형태로 반환
    fname = f"../{request.message_id}.wav"

    with open(fname, "rb") as f:
        wav_bytes = f.read()

    return wav_bytes

def getMetadata(user_id):
    fname = f"{user_id}.json"

    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [data['msg_id'], data['sender_id'], data['text']]