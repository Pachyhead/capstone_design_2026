import requests

API_URL = "http://localhost:8080/stream"

payload = {
    "target_text" : "어쨌든 우리한테 와서 건강하게 지금도 잘 자라고 있으니까",
    "ref_audio" : "./000023.wav",
    "ref_text" : "어쨌든 우리한테 와서 건강하게 지금도 잘 자라고 있으니까",
    "use_emotion" : True,
    "emotion_npy_path" : "./000022.npy",
}


with requests.post(API_URL, json=payload, stream=True) as r:
    with open("output.wav", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("💾 output.wav 파일로 저장 완료!")