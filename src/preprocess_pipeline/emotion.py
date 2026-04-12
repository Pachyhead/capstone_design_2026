import torch
import numpy as np
from funasr import AutoModel
from preprocess_pipeline.config import Config

class EmotionExtractor:
    def __init__(self, config: Config):
        self.model = AutoModel(
            model="iic/emotion2vec_plus_large",
            device=config.emotion_device,
            disable_update=True,
            #hub="ms",
        )
 
    def extract(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, list[str], list[float]]:
        """
        numpy 오디오 to 768d 감정 임베딩
        
        Args:
        - audio (np.ndarray): 1D numpy 오디오 데이터
        - sr (int): 샘플링 레이트 (Hz)

        Returns:
        - tuple[np.ndarray, list[str], list[float]]: (감정 임베딩, 감정 레이블, 감정 점수)
        """
        
        # emotion2vec은 16kHz 기대
        result = self.model.generate(
            input=audio,
            output_dir=None,
            granularity="utterance",
            extract_embedding=True,
            sampling_rate=sr,
            disable_pbar=True,
        )
        
        labels = result[0]["labels"]  # 감정 레이블
        scores = result[0]["scores"]  # 각 감정에 대한 확률 점수
        
        # 768d embedding 추출
        embedding = result[0]["feats"]  # shape: (768,)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        return embedding, labels, scores