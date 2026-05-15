import numpy as np
from faster_whisper import WhisperModel
from preprocess_pipeline.config import Config


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
            vad_filter=False,
            
            # 환각 방지 핵심
            condition_on_previous_text=False,
            
            # 임계값
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.4,
            
            # 디코딩
            beam_size=self.config.whisper_beam_size,
            temperature=[0.0, 0.2],
            
            # 반복 차단 (필요시)
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,       
        )
        text = "".join([seg.text for seg in segments]).strip()
        
        return text