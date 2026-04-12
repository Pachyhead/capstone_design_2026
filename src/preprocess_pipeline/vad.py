import torch
import torchaudio
from preprocess_pipeline.config import Config

class VadSegmenter:
    def __init__(self, config: Config):
        self.config = config
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        #self.model = self.model.to("cuda:0")
        self.get_speech_timestamps = self.utils[0]
 
    def segment(self, audio_path: str) -> list[dict]:
        """
        오디오 파일에서 발화 구간을 감지하고 청크로 반환

        Args:
        - audio_path (str): 오디오 파일 경로

        Returns:
        - list[dict]: 각 발화 청크 정보 리스트
            - audio: 청크 오디오 데이터 (numpy.ndarray)
            - start_sec: 발화 시작 시점 (초)
            - end_sec: 발화 종료 시점 (초)
            - sr: 샘플링 레이트 (Hz)
        """
        
        wav, sr = torchaudio.load(audio_path)
 
        # 1D 모노 변환
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
 
        # 리샘플링 silero vad (sample_rate = 16000) 필요
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            wav = resampler(wav)
            sr = self.config.sample_rate
 
        # VAD 실행
        speech_timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            threshold=self.config.vad_threshold,
            sampling_rate=sr,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
        )
 
        # 청크 추출
        chunks = []
        for ts in speech_timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            chunk_audio = wav[start_sample:end_sample].numpy()
            chunks.append({
                "audio": chunk_audio,
                "start_sec": round(start_sample / sr, 3),
                "end_sec": round(end_sample / sr, 3),
                "sr": sr,
            })
 
        return chunks