"""Play received WAV files on the local endpoint."""

import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

import io # RAM 가상 파일을 활용하기 위해 임포트
from typing import Generator # Generator 타입 힌팅을 위해 임포트


class AudioSpeaker:
    """Control playback through the endpoint's speaker."""

    def play_wav(self, file_path: Path | str) -> None:
        """Play a WAV file through the default speaker."""
        print(f"Opening file: {file_path}")

        with wave.open(str(file_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()

            if sampwidth == 2:
                np_dtype = np.int16
            elif sampwidth == 4:
                np_dtype = np.int32
            else:
                np_dtype = np.int16

            total_frames = wav_file.getnframes()
            raw_bytes = wav_file.readframes(total_frames)

            audio_array = np.frombuffer(raw_bytes, dtype=np_dtype)

            audio_array = audio_array.reshape(-1, channels)

            print(f"Playing audio... ({sample_rate}Hz, {channels}Ch)")

            sd.play(audio_array, sample_rate)

            sd.wait()

            print("Playback finished.")

    def play_wav_streaming(self, audio_frames: Generator, file_path: Path | str = None) -> None:
        """
        - 첫 청크(WAV 헤더) 수신 시 sample_rate 추출 후 stream 초기화
        - 이후 청크: 즉시 스피커 출력 + 동시 파일 저장
        """
        mem_file = io.BytesIO()
        stream = None
        is_first_chunk = True

        try:
            for audio_frame in audio_frames:
                if not audio_frame.audio_content:
                    if audio_frame.is_final:
                        break
                    continue

                # 첫 청크: WAV 헤더 파싱
                if is_first_chunk:
                    wav_header = audio_frame.audio_content
                    sample_rate, channels, _ = self._parse_wav_header(wav_header)

                    stream = sd.RawOutputStream(
                        channels=channels,
                        samplerate=sample_rate,
                        dtype='int16'
                    )
                    stream.start()
                    is_first_chunk = False
                    continue  # 헤더는 버림

                # 이후 청크: 즉시 재생 + 동시 저장
                stream.write(audio_frame.audio_content)
                mem_file.write(audio_frame.audio_content)

                if audio_frame.is_final:
                    print("Received final chunk from server.")
                    break

        finally:
            if stream:
                stream.stop()
                stream.close()

            if file_path:
                with open(str(file_path), "wb") as f:
                    f.write(mem_file.getvalue())

            mem_file.close()


    def stop_speaker(self) -> None:
        """Stop playback on the default speaker."""
        print("Stopping audio playback...")
        sd.stop()
    
    def _parse_wav_header(self, wav_header: bytes) -> tuple:
        """WAV 헤더에서 sample_rate, channels 추출"""
        import struct

        channels = struct.unpack('<H', wav_header[22:24])[0]
        sample_rate = struct.unpack('<I', wav_header[24:28])[0]
        bits_per_sample = struct.unpack('<H', wav_header[34:36])[0]
        
        return sample_rate, channels, bits_per_sample