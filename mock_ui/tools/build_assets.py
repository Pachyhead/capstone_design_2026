"""Prepare the offline demo's audio.

Reads the example sets (one folder per emotion, each holding reference.wav,
source.wav and generate_emotion.wav) and writes browser-friendly copies into
mock_ui/public/audio/<set>/{reference,source,output}.wav.

    python mock_ui/tools/build_assets.py [path/to/example_sets]

Needs numpy and soundfile (both are in endpoint/.venv). It prints each file's
length and size; copy the source.wav numbers into src/scenario.ts
(sourceDuration, sourceBytes), then rebuild with `npm run build`.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MOCK_UI = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = MOCK_UI.parent / "mock_ui_example_wav" / "mock_ui_example_wav"
FILES = {"reference": "reference.wav", "source": "source.wav", "output": "generate_emotion.wav"}


def main() -> None:
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source_root.is_dir():
        sys.exit(f"example sets not found: {source_root}")

    for set_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for role, filename in FILES.items():
            src = set_dir / filename
            if not src.is_file():
                sys.exit(f"missing file: {src}")
            samples, rate = sf.read(src, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            # 16-bit PCM plays in every browser; the synthesized files arrive as
            # 32-bit float WAV, which some players reject.
            dest = MOCK_UI / "public" / "audio" / set_dir.name / f"{role}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, np.clip(samples, -1.0, 1.0), rate, subtype="PCM_16")
            print(
                f"{set_dir.name}/{role}.wav  {len(samples) / rate:.2f}s  "
                f"{rate}Hz  {dest.stat().st_size} bytes"
            )


if __name__ == "__main__":
    main()
