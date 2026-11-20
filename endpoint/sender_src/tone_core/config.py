from dataclasses import dataclass, field

import torch

from preprocess_pipeline.config import Config as PreprocessConfig
from fsq_ae.config import FSQAEConfig as FSQConfig


def default_device() -> str:
    """macOS 는 mps, CUDA 가 있으면 cuda, 둘 다 없으면 cpu"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class SenderConfig:
    fsq_ckpt: str
    whisper_model: str = "large-v3"
    emotion2vec_model: str = "iic/emotion2vec_plus_large"
    preprocess_cfg: PreprocessConfig = PreprocessConfig
    device: str = field(default_factory=default_device)


@dataclass
class ReceiverConfig:
    fsq_ckpt: str
    # fsq_cfg: FSQConfig
    device: str = "cuda"
