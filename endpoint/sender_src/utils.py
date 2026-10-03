"""Compare compressed audio sizes at a target ViSQOL quality score."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from visqol import VisqolApi


def measure_audio_codec_sizes(
    wav_path: str,
) -> dict[str, object]:
    """Find the smallest MP3, AAC, and Opus files meeting the quality target."""
    input_path = Path(wav_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed.")

    # 비교할 코덱
    codecs = [
        {
            "name": "mp3",
            "encoder": "libmp3lame",
            "extension": "mp3",
        },
        {
            "name": "aac",
            "encoder": "aac",
            "extension": "m4a",
        },
        {
            "name": "opus",
            "encoder": "libopus",
            "extension": "opus",
        },
    ]

    # 각 코덱에서 시험할 bitrate
    bitrates = [
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
    ]

    # 목표 ViSQOL 점수
    target_quality = 4.5

    results: dict[str, object] = {}
    visqol_api = VisqolApi()
    visqol_api.create(mode="audio")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # ViSQOL Audio Mode용 48 kHz reference WAV
        reference_wav = temp_dir / "reference.wav"

        _run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-ar",
                "48000",
                "-c:a",
                "pcm_s16le",
                str(reference_wav),
            ]
        )

        for codec in codecs:
            candidates = []

            for bitrate in bitrates:
                compressed_path = (
                    temp_dir / f"{codec['name']}_{bitrate}.{codec['extension']}"
                )

                decoded_wav = temp_dir / f"{codec['name']}_{bitrate}_decoded.wav"

                # 1. WAV -> 압축 코덱
                _run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(input_path),
                        "-c:a",
                        codec["encoder"],
                        "-b:a",
                        f"{bitrate}k",
                        str(compressed_path),
                    ]
                )

                size = compressed_path.stat().st_size

                # 2. 압축 파일 -> WAV
                _run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(compressed_path),
                        "-ar",
                        "48000",
                        "-c:a",
                        "pcm_s16le",
                        str(decoded_wav),
                    ]
                )

                # 3. 원본 WAV와 decoded WAV의 음질 비교
                quality = float(
                    visqol_api.measure(str(reference_wav), str(decoded_wav)).moslqo
                )

                candidates.append(
                    {
                        "bitrate_kbps": bitrate,
                        "size_bytes": size,
                        "quality": quality,
                    }
                )

            # 목표 음질을 만족하는 후보만 남긴다.
            valid_candidates = [
                candidate
                for candidate in candidates
                if candidate["quality"] >= target_quality
            ]

            if valid_candidates:
                # 실제 파일 크기가 가장 작은 후보 선택
                best = min(
                    valid_candidates,
                    key=lambda candidate: candidate["size_bytes"],
                )

                results[codec["name"]] = best
            else:
                results[codec["name"]] = None

    return results


def _run(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed:\n{' '.join(command)}\n\n{e.stderr}") from e
