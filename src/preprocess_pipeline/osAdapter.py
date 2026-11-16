"""
플랫폼별 Whisper 백엔드 선택

- macOS : mlx-whisper
- 그 외 (Linux / Windows) : faster-whisper

"""
import sys
from abc import ABC, abstractmethod

import numpy as np

from preprocess_pipeline.config import Config


class STTBackend(ABC):
    """모든 Whisper 백엔드의 공통 인터페이스"""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """
        numpy 오디오 to 텍스트

        Args:
            audio (np.ndarray): 1D numpy 오디오 데이터 (16kHz, float32)

        Returns:
            str: STT 결과 텍스트
        """


class FasterWhisperBackend(STTBackend):
    """Linux / Windows (CUDA) 용"""

    def __init__(self, config: Config):
        from faster_whisper import WhisperModel

        self.config = config
        self.model = WhisperModel(
            config.whisper_model_size,
            device=config.whisper_device,
            device_index=config.whisper_device_index,
            compute_type=config.whisper_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
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
        return "".join([seg.text for seg in segments]).strip()


class MLXWhisperBackend(STTBackend):
    """macOS 용"""

    def __init__(self, config: Config):
        import mlx_whisper

        self.config = config
        self._mlx_whisper = mlx_whisper
        self.repo = config.whisper_model_name

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=self.config.whisper_language,
            verbose=False,
        )
        return result["text"].strip()


_BACKENDS = {
    "faster_whisper": FasterWhisperBackend,
    "mlx": MLXWhisperBackend,
}


def detect_backend() -> str:
    """현재 플랫폼에 맞는 백엔드 이름"""
    if sys.platform == "darwin":
        return "mlx"
    return "faster_whisper"


def create_stt(config: Config) -> STTBackend:
    """config.whisper_backend 가 "auto" 면 플랫폼으로 판단, 아니면 지정한 백엔드를 강제한다."""
    name = config.whisper_backend
    if name == "auto":
        name = detect_backend()

    if name not in _BACKENDS:
        raise ValueError(f"알 수 없는 whisper_backend")

    try:
        return _BACKENDS[name](config)
    except ImportError as e:
        pkg = "mlx-whisper" if name == "mlx" else "faster-whisper"
        raise ImportError(
            f"whisper_backend에 필요한 패키지 {pkg} 가 설치되어 있지 않음"
        ) from e
