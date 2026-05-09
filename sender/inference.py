from faster_whisper import WhisperModel

def speech2emovec(audio_path: str = "") -> str | None:
    return "test"

def speech2text(audio_path: str = "") -> str | None:
    if audio_path == "":
        print("audio path is empty")
        return None
    
    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="int8_float16"
    )

    segments, info = model.transcribe(
        audio_path,
        language="ko",
        vad_filter=True
    )
    
    texts = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        texts.append(segment.text)

    return " ".join(texts).strip()

if __name__ == "__main__":
    speech2text("/home/cap/capstone_design_2026/sender/storage/000001.wav")