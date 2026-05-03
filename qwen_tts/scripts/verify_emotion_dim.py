# coding=utf-8
"""Verify Emotion2Vec output dimensionality across the dataset.

Run this BEFORE building manifests so the rest of the pipeline can pin
emotion_dim to the verified value.

Usage:
    python -m qwen_tts.scripts.verify_emotion_dim \\
        --input_jsonl path/to/raw_metadata.jsonl \\
        --data_root path/to/dataset_root \\
        --n_samples 100
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="Raw jsonl from src/preprocess_pipeline")
    parser.add_argument("--data_root", required=True, help="Directory that contains 'embeddings/...' npy files referenced by emo_vec")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)

    items = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    n = min(args.n_samples, len(items))
    sampled = random.sample(items, n)

    shapes = Counter()
    missing = 0
    load_fail = 0
    granularity = Counter()  # "utterance" if 1D, "frame" if 2D

    for it in sampled:
        if "emo_vec" not in it:
            load_fail += 1
            continue
        path = data_root / it["emo_vec"]
        if not path.exists():
            missing += 1
            continue
        try:
            arr = np.load(path)
        except Exception:
            load_fail += 1
            continue
        shapes[tuple(arr.shape)] += 1
        if arr.ndim == 1:
            granularity["utterance"] += 1
        elif arr.ndim == 2:
            granularity["frame"] += 1
        else:
            granularity[f"unknown(ndim={arr.ndim})"] += 1

    print(f"Sampled: {n} files")
    if missing:
        print(f"  Missing on disk: {missing}")
    if load_fail:
        print(f"  Failed to load: {load_fail}")
    print(f"  Granularity: {dict(granularity)}")

    if not shapes:
        print("ERROR: no valid emotion vectors loaded.")
        return 1

    print(f"  Distinct shapes: {len(shapes)}")
    for shape, cnt in shapes.most_common():
        print(f"    {shape}: {cnt}")

    if len(shapes) == 1:
        shape = next(iter(shapes))
        if len(shape) == 1:
            dim = shape[0]
            print(f"\nDetected emotion dimension: {dim} (utterance-level [D])")
        elif len(shape) == 2:
            dim = shape[1]
            print(f"\nDetected emotion dimension: {dim} (frame-level [T, D]; collapse via mean(0) at dataset time)")
        else:
            print(f"\nUnsupported tensor rank {len(shape)} -- aborting.")
            return 1
    else:
        # Frame-level lengths can vary; the trailing dim should still be consistent.
        last_dims = {s[-1] for s in shapes if len(s) >= 1}
        if len(last_dims) == 1:
            print(f"\nDetected emotion dimension: {next(iter(last_dims))} (consistent trailing dim across {len(shapes)} shape signatures)")
        else:
            print(f"\nINCONSISTENT trailing dims: {last_dims}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
