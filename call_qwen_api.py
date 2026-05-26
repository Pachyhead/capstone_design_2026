import requests

API_URL = "http://localhost:8080/stream"

payload = {
    "target_text" : "호출이 이상하게 되는 이유를 알아보자",
    "ref_audio" : "./000023.wav",
    "ref_text" : "어쨌든 우리한테 와서 건강하게 지금도 잘 자라고 있으니까",
    "use_emotion" : True,
    "emotion_npy_path" : "./100324.npy",
    "temperature" : 0.5,
    "top_k" : 50,
    "top_p" : 0.8
}


with requests.post(API_URL, json=payload, stream=True) as r:
    if r.status_code != 200 :
        print(r.content)
    else :
        with open("output.wav", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
print("💾 output.wav 파일로 저장 완료!")