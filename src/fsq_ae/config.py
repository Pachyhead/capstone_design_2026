from dataclasses import dataclass, field
from typing import List
 
 
@dataclass
class FSQAEConfig:
    # 모델 구조
    input_dim: int = 1024                                                   # emotion2vec 출력
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])      # MLP
    hidden_dims_skip: List[int] = field(default_factory=lambda: [512, 512, 512, 512])

    # FSQ
    levels: List[int] = field(default_factory=lambda: [5, 5, 5, 5, 5, 5, 5, 5, 5])
 
    # 데이터
    data_path: str = "../data/processed/emotion2vec_dataset.pt"
    val_speaker_ratio: float = 0.1                         
 
    # 학습
    batch_size: int = 256
    learning_rate: float = 3e-4
    num_epochs: int = 200
    grad_clip: float = 1.0
    num_workers: int = 2
    
    # Loss
    emotion_kl_weight: float = 0.1
 
    # 환경
    seed: int = 66
    save_dir: str = "../data/checkpoints/fsq_ae"
    log_interval: int = 100
 