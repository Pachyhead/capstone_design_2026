import numpy as np
from faster_whisper import WhisperModel
from .config import Config


class WhisperSTT:
    def __init__(self, config: Config):
        self.model = WhisperModel(
            config.whisper_model_size,
            device=config.whisper_device,
            device_index=config.whisper_device_index,
            compute_type=config.whisper_compute_type,
        )
        self.config = config
 
    def transcribe(self, audio: np.ndarray) -> str:
        """
        numpy 오디오 to 텍스트
        
        Args:
            audio (np.ndarray): 1D numpy 오디오 데이터
        
        Returns:
            str: STT 결과 텍스트
        
        """
        segments, _ = self.model.transcribe(
            audio,
            language=self.config.whisper_language,
            beam_size=self.config.whisper_beam_size,
            vad_filter=False,
        )
        text = "".join([seg.text for seg in segments]).strip()
        
        return text