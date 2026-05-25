import requests

API_URL = "http://localhost:8080/stream"

payload = {
    "target_text" : "안녕하세요",
    "ref_audio" : "./000024.wav",
    "ref_text" : "참 기쁘고 기특하고 좋아",
    "use_emotion" : True,
    "emotion_npy_path" : "./000001.npy",
}


with requests.post(API_URL, json=payload, stream=True) as r:
    with open("output.wav", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("💾 output.wav 파일로 저장 완료!")