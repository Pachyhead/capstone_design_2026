from dataclasses import dataclass

@dataclass
class Config:
    # 확장자
    extensions: tuple = ("wav", "WAV")      

    # VAD 설정 default
    sample_rate: int = 16000                # VAD 모델이 기대하는 샘플링 레이트
    vad_threshold: float = 0.5              # 음성 감지 임계값
    min_speech_duration_ms: int = 250       # 최소 발화 길이 (ms)
    min_silence_duration_ms: int = 200      # 발화 사이 최소 침묵 (ms)
    speech_pad_ms: int = 100                # 발화 앞뒤 패딩 (ms)
    
    # GPU 할당
    whisper_device: str = "cuda"            # faster-Whisper는 deviee:"cuda", device_index:0 으로 분리해서 받음
    whisper_device_index: int = 0
    emotion_device: str = "cuda:1"
    
    # Whisper 설정
    whisper_model_size: str = "large-v3"    # "tiny", "base", "small", "medium", "large-v2", "large-v3" 중 가장 정확한 모델 default
    whisper_language: str = "ko"
    whisper_beam_size: int = 5              # 우선 5가 적절해 보임, 정확도 확인해서 나중에 1로 바꾸면 될거 같음
    whisper_compute_type: str = "float16"   

    # emotion2vec 감정 라벨 순서
    EMOTION_LABELS: tuple = (
        "angry", "disgusted", "fearful", "happy", 
        "neutral", "other", "sad", "surprised", "unk"
    )