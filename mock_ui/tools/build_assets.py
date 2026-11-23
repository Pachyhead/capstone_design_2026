"""Make the demo's audio browser-friendly and report its length and size.

The audio lives in mock_ui/public/audio/<backend id>/ as reference.wav,
source.wav and output.wav (0 종찬, 1 재웅, 2 경택). Drop new recordings in
under those names, then run:

    endpoint/.venv/Scripts/python.exe mock_ui/tools/build_assets.py

Files that are not 16-bit PCM (the recorder and the TTS both write 32-bit
float WAV, which some players reject) are converted in place. Stereo is mixed
down to mono. The printed source.wav length and size go into
src/scenario.ts (sourceDuration, sourceBytes), and the reference.wav length
into referenceDuration. Rebuild with `npm run build` afterwards.

Needs numpy and soundfile (both are in endpoint/.venv).
"""

from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO = Path(__file__).resolve().parent.parent / "public" / "audio"
ROLES = ("reference", "source", "output")


def main() -> None:
    for folder in sorted(p for p in AUDIO.iterdir() if p.is_dir()):
        for role in ROLES:
            path = folder / f"{role}.wav"
            if not path.is_file():
                print(f"{folder.name}/{role}.wav  MISSING")
                continue

            info = sf.info(path)
            note = ""
            if info.subtype != "PCM_16" or info.channels != 1:
                samples, rate = sf.read(path, dtype="float32")
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                sf.write(path, np.clip(samples, -1.0, 1.0), rate, subtype="PCM_16")
                note = f"  (converted from {info.subtype}, {info.channels}ch)"
                info = sf.info(path)

            print(
                f"{folder.name}/{role}.wav  {info.duration:.2f}s  {info.samplerate}Hz  "
                f"{path.stat().st_size} bytes{note}"
            )


if __name__ == "__main__":
    main()
