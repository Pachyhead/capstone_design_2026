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
        - 청크 도착 시 즉시 스피커 출력
        - 동시에 파일 저장
        """
        mem_file = io.BytesIO() # RAM에 가상 파일 생성(속도 빠름)
        stream = None

        try:
            for audio_frame in audio_frames:
                if not audio_frame.audio_content:
                    continue
                
                # 첫 청크인 경우, WAV 헤더를 파싱하여 해당 포맷(channels, samplerate, dtype)에 맞게 stream 시작
                if stream is None:
                    wav_header = audio_frame.audio_content
                    sample_rate, channels, sampwidth = self._parse_wav_header(wav_header)

                    if sampwidth == 16:
                        np_dtype = np.int16
                    elif sampwidth == 32:
                        np_dtype = np.int32
                    else:
                        np_dtype = np.int16

                    # RAM 파일에 헤더 저장
                    mem_file.write(wav_header)

                    # RawStream 시작(즉시 재생 시작)
                    stream = sd.RawStream(
                        channels=channels,
                        samplerate=sample_rate,
                        dtype=np.int16
                    )
                    stream.start()

                else:
                    # 이후 청크인 경우, 즉시 재생하며 동시에 파일 저장
                    stream.write(audio_frame.audio_content) # 지연시간 낮춤(TTFB 개선)
                    mem_file.write(audio_frame.audio_content)
                
                if audio_frame.is_final:
                    print("Received final chunk from server.")
                    break
        
        finally:
            if stream:
                stream.stop()
                stream.close()
            
            # 스트리밍 끝난 후 디스크로 딱 한번 내보냄
            if file_path:
                with open(str(file_path), "wb") as f:
                    f.write(mem_file.getvalue()) # RAM에 있던 데이터를 실제 디스크 파일로 저장
            
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