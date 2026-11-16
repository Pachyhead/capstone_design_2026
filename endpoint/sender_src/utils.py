"""Estimate speech payload sizes at representative service bitrates."""

from pathlib import Path

from scipy.io import wavfile

# 음성 서비스에서 쓰이는 대표 비트레이트 (kbps)
# - opus: RFC 7587 광대역 음성 권장 범위 16~20 kbps의 하한
# - volte: AMR-WB 12.65 kbps 모드 (GSMA HD Voice 필수 모드 세트)
# - codec2: Codec 2 700C 모드
SERVICE_BITRATES_KBPS = {
    "opus": 16,
    "volte": 12.65,
    "codec2": 0.7,
}


def measure_audio_codec_sizes(
    wav_path: str,
) -> dict[str, object]:
    """Return the WAV size and each codec's size at its service bitrate."""
    input_path = Path(wav_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")

    rate, samples = wavfile.read(input_path)
    seconds = len(samples) / rate

    results: dict[str, object] = {
        "wav": {"size_bytes": input_path.stat().st_size},
    }

    # 고정 비트레이트 코덱: 크기 = 비트레이트 × 길이 / 8
    for name, bitrate_kbps in SERVICE_BITRATES_KBPS.items():
        results[name] = {
            "bitrate_kbps": bitrate_kbps,
            "size_bytes": round(bitrate_kbps * 1000 * seconds / 8),
        }

    return results
