import numpy as np
import torch

from preprocess_pipeline.stt import WhisperSTT
from preprocess_pipeline.emotion import EmotionExtractor
from fsq_ae import load

from .types import EncodeResult, EmotionLabel
from .config import SenderConfig


class SenderEncode:
    def __init__(self, stt, emotion_encoder, fsq_model, fsq_norm, device):
        self.stt = stt
        self.emotion = emotion_encoder
        self.fsq = fsq_model
        self.norm = fsq_norm
        self.device = device
        
    @classmethod
    def from_config(cls, cfg: SenderConfig) -> "SenderEncode":
        fsq_model, _, fsq_norm = load(cfg.fsq_ckpt)
        stt = WhisperSTT(cfg.preprocess_cfg)
        emotion_encoder = EmotionExtractor(cfg.preprocess_cfg)
        return cls(stt, emotion_encoder, fsq_model, fsq_norm, cfg.device)
    # 1. 각 모델별 device 수정해야됨
    # 2. 송신단 서버 활용 가능한지?
    
    @torch.inference_mode()
    def encode(self, audio: np.ndarray[np.float32]) -> EncodeResult:
        text = self.stt.transcribe(audio)
        emo_vec_1024d, el, es = self.emotion.extract(audio)
        
        idx = int(np.argmax(es))
        emo_label = EmotionLabel[idx]
        # emo_label = el[idx]
        emo_score = float(es[idx])
        
        emo_vec_8d = self.fsq.codes_to_indices(self.fsq(emo_vec_1024d)[2])
        
        return EncodeResult(
            text=text,
            emotion_indices=emo_vec_8d,
            emotion_label=emo_label,
            emotion_score=emo_score,
        )

def main():
    sample_rate = 16000
    duration = 1.0
    arr = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    sender = SenderEncode.from_config(fsq_ckpt = "경로")
    sender.encode(arr)
    

if __name__ == "__main__":
    main()