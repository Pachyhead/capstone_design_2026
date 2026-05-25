import requests

# API 설정 및 오디오 스펙
# 1. 오디오 스트리밍 및 PyAudio 설정
CHANNELS = 1
RATE = 24000  # API의 샘플 레이트와 반드시 맞춰야 합니다.

# 2. 로컬 API URL 설정
API_URL = "http://localhost:8080/stream" 

payload = {
    "target_text": "안녕하세요.",
    "ref_audio": "./000024.wav",
    "ref_text" : "참 기쁘고 기특하고 좋아.",
    "emotion_npy_path" : "./000001.npy",
    "speed": 1.0,
    "pitch": 0.0
}
    
with requests.post(API_URL, json=payload, stream=True) as r:
    with open("output.wav", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("💾 output.wav 파일로 저장 완료!")