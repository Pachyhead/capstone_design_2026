import json

def doSomething(request):
    try:
        fname = f"{request.receiver_id}.json"

        with open(fname, "w", encoding="utf-8") as f:
            f.write("{\n")
            f.write(f"\t\"sender_id\": \"{request.sender_id}\",\n")
            f.write(f"\t\"receiver_id\": \"{request.receiver_id}\",\n")
            f.write(f"\t\"text\": \"{request.text}\",\n")
            f.write(f"\t\"emotion_vector\": {request.emotion_vector}\n")
            f.write("}")
            return True
    except OSError as e:
        print(f"File write error: {e}")
        return False

def get_pending_message(user_id):
    fname = f"../{user_id}.wav"

    with open(fname, "rb") as f:
        wav_bytes = f.read()

    return wav_bytes