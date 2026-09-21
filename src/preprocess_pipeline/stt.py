import numpy as np
from preprocess_pipeline.config import Config
from preprocess_pipeline.osAdapter import create_stt


class WhisperSTT:
    """
    플랫폼에 맞는 Whisper 백엔드(faster-whisper / mlx-whisper)에 위임한다.
    백엔드 구현과 선택 로직은 osAdapter.py 참고.
    """

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
