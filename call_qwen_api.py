import requests

API_URL = "http://localhost:8080/stream"

payload = {
    "target_text" : "와 진짜 태원이 너무 대단한데..?",
    "ref_audio" : "./DataBase/ref_audio/0_ref.wav",
    "ref_text" : "안녕하세요, 오늘 날씨가 참 좋아서 산책하기 딱 좋은 날인 것 같네요. 주말에는 보통 집에서 책을 읽거나 가까운 카페에 가서 시간을 보냅니다.",
    "use_emotion" : True,
    "emotion_npy_path" : "./DataBase/emotion_vectors/06018423-a74e-47f0-ba29-512bd61605bf.npy",
    "temperature" : 0.9,
    "top_k" : 50,
    "top_p" : 1.0
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