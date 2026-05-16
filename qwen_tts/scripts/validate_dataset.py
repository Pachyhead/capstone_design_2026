# coding=utf-8
"""Pre-training integrity check on a manifest jsonl.

Verifies (without materializing the whole training set):
  - wav files exist and load via soundfile
  - emo_vec files exist, load, and share a consistent trailing dim
  - audio_codes is present (post-add_audio_codes step)
  - every entry has a non-empty neutral_pool
  - sample rates encountered (Qwen3-TTS uses 24kHz speaker encoder; mismatched rates require resample at dataset time)

Use --full to walk every entry; default is a 200-sample spot-check that scales
to large manifests.

Usage:
    python -m qwen_tts.scripts.validate_dataset \\
        --manifest manifests/manifest_train.codes.jsonl \\
        --data_root /path/to/dataset_root [--full]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def validate(manifest_path: str, data_root: str, *, full: bool = False, spot_n: int = 200, seed: int = 42) -> dict:
    data_root = Path(data_root)
    items = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Validating {manifest_path}: {len(items)} entries")

    issues = {
        "missing_wav": [],
        "missing_emo": [],
        "wav_load_fail": [],
        "emo_load_fail": [],
        "missing_audio_codes": [],
        "empty_neutral_pool": [],
        "emo_dim_mismatch": [],
    }
    sample_rates = Counter()
    expected_dim = None

    if full or len(items) <= spot_n:
        check_set = items
    else:
        random.seed(seed)
        check_set = random.sample(items, spot_n)
    print(f"  Spot-checking files: {len(check_set)} entries")

    try:
        import soundfile as sf  # noqa: F401
    except ImportError:
        sf = None
        print("  (soundfile not available -- skipping sample-rate check)")

    for item in check_set:
        # wav
        wav_rel = item.get("wav")
        if not wav_rel:
            issues["missing_wav"].append(item.get("id"))
            continue
        wav_path = data_root / wav_rel
        if not wav_path.exists():
            issues["missing_wav"].append(item.get("id"))
        elif sf is not None:
            try:
                info = sf.info(str(wav_path))
                sample_rates[info.samplerate] += 1
            except Exception:
                issues["wav_load_fail"].append(item.get("id"))

        # emo
        emo_rel = item.get("emo_vec")
        if not emo_rel:
            issues["missing_emo"].append(item.get("id"))
            continue
        emo_path = data_root / emo_rel
        if not emo_path.exists():
            issues["missing_emo"].append(item.get("id"))
            continue
        try:
            arr = np.load(emo_path)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            if expected_dim is None:
                expected_dim = int(arr.shape[0])
            elif int(arr.shape[0]) != expected_dim:
                issues["emo_dim_mismatch"].append(item.get("id"))
        except Exception:
            issues["emo_load_fail"].append(item.get("id"))

    # Cheap full-pass checks
    for item in items:
        if "audio_codes" not in item or not item["audio_codes"]:
            issues["missing_audio_codes"].append(item.get("id"))
        if not item.get("neutral_pool"):
            issues["empty_neutral_pool"].append(item.get("id"))

    print(f"\n--- Validation Report ---")
    print(f"Detected emotion_dim: {expected_dim}")
    if sample_rates:
        print(f"Sample rates seen   : {dict(sample_rates)}")
        if 24000 not in sample_rates:
            print("  WARNING: no 24kHz files detected. Dataset will resample but speaker_encoder mel must be 24kHz.")
    for key, ids in issues.items():
        n = len(ids)
        if n:
            print(f"  {key:25s}: {n}  (first few: {ids[:5]})")
        else:
            print(f"  {key:25s}: OK")
    return {"issues": {k: len(v) for k, v in issues.items()}, "emotion_dim": expected_dim, "sample_rates": dict(sample_rates)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--full", action="store_true", help="Check every entry's files (slow on large manifests)")
    parser.add_argument("--spot_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validate(args.manifest, args.data_root, full=args.full, spot_n=args.spot_n, seed=args.seed)


if __name__ == "__main__":
    main()
