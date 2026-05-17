import numpy as np
import torch

# from preprocess_pipeline.stt import WhisperSTT
import mlx_whisper
from preprocess_pipeline.emotion import EmotionExtractor
from fsq_ae.load import load_fsq_ae

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
        cfg.preprocess_cfg.emotion_device = cfg.device
        cfg.preprocess_cfg.whisper_device = cfg.device
        
        fsq_model, _, fsq_norm = load_fsq_ae(cfg.fsq_ckpt, cfg.device)
        # stt = WhisperSTT(cfg.preprocess_cfg)
        stt = cfg.preprocess_cfg.whisper_model_name
        emotion_encoder = EmotionExtractor(cfg.preprocess_cfg)
        return cls(stt, emotion_encoder, fsq_model, fsq_norm, cfg.device)
    
    @torch.inference_mode()
    def encode(self, audio: np.ndarray[np.float32], sample_rate = 16000) -> EncodeResult:
        # text = self.stt.transcribe(audio)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.stt,
            language="ko",      # 한국어 음성이면 지정
            verbose=False,
        )
        text = result["text"]
        emo_vec_1024d, _, es = self.emotion.extract(audio, sample_rate)
        
        x = torch.as_tensor(emo_vec_1024d, dtype=torch.float32, device=self.device)
        x = (x - self.norm["mean"]) / self.norm["std"]
        
        idx = int(np.argmax(es))
        emo_label = EmotionLabel(idx)
        emo_score = float(es[idx])
        emo_vec_8d = tuple(self.fsq.encode_to_indices(x).cpu().tolist())
        
        return EncodeResult(
            text=text,
            emotion_indices=emo_vec_8d,
            emotion_label=emo_label,
            emotion_label_idx=idx,
            emotion_score=emo_score,
        )