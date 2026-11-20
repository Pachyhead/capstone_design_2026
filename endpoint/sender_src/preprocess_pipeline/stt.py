import numpy as np
from preprocess_pipeline.config import Config
from preprocess_pipeline.osAdapter import create_stt


class WhisperSTT:
    def __init__(self, config: Config):
        self.config = config
        self.backend = create_stt(config)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        numpy 오디오 to 텍스트

        Args:
            audio (np.ndarray): 1D numpy 오디오 데이터

        Returns:
            str: STT 결과 텍스트

        """
        return self.backend.transcribe(audio)
