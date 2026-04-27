from typing import Tuple
 
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
 
 
class EmotionEmbedDataset(Dataset):
    """
    utterance-level emotion2vec 임베딩 (정규화 자동 적용)
    """
 
    def __init__(self, data_path: str):
        obj = torch.load(data_path, map_location="cpu")
 
        self.embeddings = obj["embeddings"].float()        # (N, D)
        self.ids = obj["ids"]                              # 원본 청크 ID
        self.speakers = obj["speakers"]                    # 화자 split용
        self.emotions = obj["emotions"]                    # 2단계 검증용
        self.mean = obj["mean"].float()                    # 정규화 통계
        self.std = obj["std"].float()
 
        print(
            f"[Dataset] N={len(self):,}, D={self.embeddings.shape[1]}, "
            f"화자 {len(set(self.speakers)):,}명"
        )
 
    def __len__(self):
        return self.embeddings.shape[0]
 
    def __getitem__(self, idx):
        # z-score 정규화
        return (self.embeddings[idx] - self.mean) / self.std
 
    def speaker_split(self, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
        """
        화자 단위로 train/val 분리. 같은 화자가 양쪽에 들어가지 않음
        """
        rng = np.random.default_rng(seed)
        unique_speakers = sorted(set(self.speakers))
        rng.shuffle(unique_speakers)
 
        n_val = max(1, int(len(unique_speakers) * val_ratio))
        val_speakers = set(unique_speakers[:n_val])
 
        is_val = np.isin(self.speakers, list(val_speakers))
        train_idx = np.where(~is_val)[0].tolist()
        val_idx = np.where(is_val)[0].tolist()
 
        print(
            f"[Split] 화자 train {len(unique_speakers) - n_val} / val {n_val} | "
            f"샘플 train {len(train_idx):,} / val {len(val_idx):,}"
        )
        return Subset(self, train_idx), Subset(self, val_idx)