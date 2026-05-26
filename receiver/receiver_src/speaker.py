import wave
import threading
from pathlib import Path
import numpy as np
import sounddevice as sd


class AudioSpeaker:
    def play_wav(self, file_path: Path | str) -> float:
        """Plays the wav in a background thread and returns its duration in seconds
        so the HTTP caller can respond immediately."""
        print(f"Opening file: {file_path}")

        with wave.open(str(file_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()
            raw_bytes = wav_file.readframes(total_frames)

        if sampwidth == 2:
            np_dtype = np.int16
        elif sampwidth == 4:
            np_dtype = np.int32
        else:
            np_dtype = np.int16

        audio_array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(-1, channels)
        duration = total_frames / sample_rate if sample_rate else 0.0

        print(f"Playing audio... ({sample_rate}Hz, {channels}Ch, {duration:.2f}s)")

        def _play():
            sd.play(audio_array, sample_rate)
            sd.wait()
            print("Playback finished.")

        threading.Thread(target=_play, daemon=True).start()

        return duration

    def stop_speaker(self) -> None:
        print("Stopping audio playback...")
        sd.stop()
