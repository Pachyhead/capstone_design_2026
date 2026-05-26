import wave
from pathlib import Path
import numpy as np
import sounddevice as sd


class AudioSpeaker:
    def play_wav(self, file_path: Path | str) -> None:
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

    def stop_speaker(self) -> None:
        print("Stopping audio playback...")
        sd.stop()