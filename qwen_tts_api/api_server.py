import struct
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import torch

from qwen_tts.inference.emotion_loader import load_emotion_projector
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.lora import set_lora_enabled

## import torch # 또는 사용하시는 라이브러리
print(f"Loading Model (no emotion projector) ...")
qwen3tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
)
print(f"Loading Model with projector ...")
load_emotion_projector(
    qwen3tts.model,
    "app/qwen_tts_api/chk_1",
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)


def _load_emotion_npy(path: str) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return torch.from_numpy(arr.astype(np.float32))


def _pcm16_bytes(wav: np.ndarray) -> bytes:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


def _streaming_wav_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    # 길이를 알 수 없는 streaming WAV: RIFF/data size를 0xFFFFFFFF로 두면
    # 대부분의 디코더(ffmpeg, modern browser <audio>)가 EOF까지 읽어준다.
    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


app = FastAPI()


class Query(BaseModel):
    target_text: str
    ref_audio: str
    ref_text: Optional[str] = None
    emotion_npy_path: Optional[str] = None
    use_emotion: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 4096
    chunk_ms: int = 320  # streaming packet size (12Hz 토크나이저 default 4 timestep)


def _synthesize(req: Query):
    """모델 호출 한 번. (wav: np.ndarray[float32], sr: int)"""
    use_emo = req.use_emotion and req.emotion_npy_path is not None
    emotion = _load_emotion_npy(req.emotion_npy_path) if use_emo else None
    # baseline 비교용으로 emotion 안 쓸 때는 LoRA도 꺼서 진짜 upstream 모델 동작
    set_lora_enabled(qwen3tts.model, use_emo)

    wavs, sr = qwen3tts.generate_voice_clone(
        text=req.target_text,
        language="Korean",
        ref_audio=req.ref_audio,
        ref_text=req.ref_text,
        emotion_vec=emotion,
        do_sample=True,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_new_tokens=req.max_new_tokens,
    )
    return wavs[0], sr


@app.post("/")
async def predict(req: Query):
    try:
        wav, sr = _synthesize(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")
    return {"status": "success", "sample_rate": sr, "wav": wav.tolist()}


@app.post("/stream")
async def stream(req: Query):
    """WAV streaming endpoint.

    주의: 현재 model.generate()는 전체 시퀀스를 한 번에 반환하므로 TTFB(첫 패킷까지)는
    full-synthesis 비용이 든다. 본 엔드포인트는 (1) 합성을 동기 수행한 뒤
    (2) 생성된 PCM을 chunk_ms 간격으로 흘려보내는 'chunked WAV streaming' 방식.
    클라이언트는 점진 재생이 가능해 UX적으로는 스트리밍과 동일하게 동작.
    저지연 실시간 streaming을 원하면 talker.generate를 자체 자기회귀 루프로 풀고
    codec_decoder를 packet 단위로 호출하도록 별도 구현 필요.
    """
    try:
        wav, sr = _synthesize(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    samples_per_chunk = max(1, int(sr * req.chunk_ms / 1000))

    def gen():
        yield _streaming_wav_header(sample_rate=sr)
        for start in range(0, wav.shape[0], samples_per_chunk):
            chunk = wav[start:start + samples_per_chunk]
            if chunk.size == 0:
                break
            yield _pcm16_bytes(chunk)

    return StreamingResponse(
        gen(),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sr),
            "X-Chunk-Ms": str(req.chunk_ms),
            "Cache-Control": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn

    # 지정한 포트로 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8080)
