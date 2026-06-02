from datetime import datetime
from pathlib import Path
from typing import Any
import time

from tone_core.sender import SenderEncode, EncodeResult

from scipy.io.wavfile import write
from scipy.signal import resample
import numpy as np
import sounddevice as sd

class AudioRecorder:
    def __init__(self, storage: Path, encoder: SenderEncode) -> None:
        """ Initialize class properties """
        self.storage: Path = storage
        self.encoder: SenderEncode = encoder
        self.stream: sd.InputStream | None = None
        self.audio_buffer: list[np.ndarray] = []
        self.hardware_sample_rate: int = 16000 
        self.target_sample_rate: int = 16000
        self.channels: int = 1
        self.temp_result: EncodeResult | None = None
        self.cur_tick: float | None = None

    def start_recording(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_buffer = []

        try:
            device_info = sd.query_devices(kind='input')
            self.hardware_sample_rate = int(device_info['default_samplerate'])
            print(f"마이크 하드웨어 샘플링 레이트 감지: {self.hardware_sample_rate}Hz")
        except Exception as e:
            print(f"장치 정보를 가져오지 못해 기본값 사용: {e}")
            self.hardware_sample_rate = sample_rate

        def callback(indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
            if status:
                print(status)
            self.audio_buffer.append(indata.copy())

        print("Recording started...")

        self.stream = sd.InputStream(
            samplerate=self.hardware_sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self.stream.start()
        self.cur_tick = time.perf_counter()

    def stop_recording(self, encording: bool=True) -> tuple[EncodeResult | None, float, Path]:
        if not (self.stream and self.cur_tick):
            raise ValueError("No active recording found.")

        print("Recording stopped. Processing data...")

        self.stream.stop()
        self.stream.close()
        self.stream = None

        if not self.audio_buffer:
            raise ValueError("No audio data captured.")

        audio_data: np.ndarray = np.concatenate(self.audio_buffer, axis=0).flatten()

        if self.hardware_sample_rate != self.target_sample_rate:
            print(f"리샘플링 중: {self.hardware_sample_rate}Hz -> {self.target_sample_rate}Hz")
            
            # 원래 샘플 수 대비 변경될 샘플 수 계산
            num_samples = int(len(audio_data) * self.target_sample_rate / self.hardware_sample_rate)
            # 리샘플링 실행 후 float32 타입 유지
            audio_data = resample(audio_data, num_samples).astype(np.float32)

        if not isinstance(audio_data, np.ndarray):
            raise TypeError(
                f"recorded audio is not ndarray: {type(audio_data)}"
            )
        if audio_data.dtype != np.float32:
            raise TypeError(
                f"recorded audio type is not float32: {audio_data.dtype}"
            )
        if audio_data.ndim != 1:
            raise ValueError(f"not mono audio: {audio_data.ndim}")

        if encording:
            self.temp_result = self.encoder.encode(audio_data)
        # reference audio 녹음 과정
        else:
            self.temp_result = None

        filename: str = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
        file_path: Path = self.storage / filename

        write(file_path, self.target_sample_rate, audio_data)
        print(f"Recording saved: {file_path}")

        prev_tick = self.cur_tick
        cur_tick = time.perf_counter()
        self.cur_tick = None

        return (self.temp_result, cur_tick - prev_tick, file_path)