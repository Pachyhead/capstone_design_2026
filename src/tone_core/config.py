from dataclasses import dataclass
from preprocess_pipeline.config import Config as PreprocessConfig
from fsq_ae.config import FSQAEConfig as FSQConfig

@dataclass
class SenderConfig:
    fsq_ckpt: str 
    whisper_model: str = "large-v3"
    emotion2vec_model: str = "iic/emotion2vec_plus_large"
    preprocess_cfg: PreprocessConfig = PreprocessConfig
    device: str = "cuda"
    
@dataclass
class ReceiverConfig:
    fsq_ckpt: str
    # fsq_cfg: FSQConfig
    device: str = "cuda"