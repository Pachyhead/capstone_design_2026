# coding=utf-8
"""Build training/validation manifests for Qwen3-TTS emotion fine-tuning.

Pipeline:
  1. Read raw jsonl (one entry per utterance) produced by src/preprocess_pipeline.
  2. Apply filters: non_verbal, duration bounds, empty text, missing files on disk.
  3. Derive speaker_id from source_file ('{rec}_person_{pos}' -> '{rec}_{pos}').
  4. Build per-speaker neutral pool (label == 'neutral' AND duration >= min_neutral_duration).
  5. Drop speakers without a usable neutral pool (or, with --allow_neutral_fallback,
     keep them by promoting the highest-neutral-score sample).
  6. Speaker-level train/val split (val_speaker_ratio fraction of speakers go to val).
  7. Attach 'neutral_pool' (list of neutral sample ids of the same speaker).
  8. Write manifest_train.jsonl, manifest_val.jsonl, manifest_stats.json.

The audio_codes field is NOT computed here -- run scripts/add_audio_codes.py
afterwards (it batches Qwen3TTSTokenizer.encode over each manifest).

Usage:
    python -m qwen_tts.scripts.build_manifest \\
        --input_jsonl raw_metadata.jsonl \\
        --data_root /path/to/dataset_root \\
        --output_dir /path/to/manifests
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

NEUTRAL_LABEL = "neutral"
EMOTION_CLASSES = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")


def parse_speaker_id(source_file: str) -> str:
    parts = source_file.split("_person_")
    if len(parts) != 2:
        return source_file
    return f"{parts[0]}_{parts[1]}"


def neutral_score_index() -> int:
    return EMOTION_CLASSES.index(NEUTRAL_LABEL)


def _filter(items, data_root: Path, args, counts: Counter):
    out = []
    for item in items:
        if item.get("non_verbal", False):
            counts["filter_non_verbal"] += 1
            continue
        text = (item.get("text") or "").strip()
        if not text:
            counts["filter_empty_text"] += 1
            continue
        dur = float(item.get("duration_sec", 0.0))
        if dur < args.min_duration:
            counts["filter_too_short"] += 1
            continue
        if dur > args.max_duration:
            counts["filter_too_long"] += 1
            continue
        if "wav" not in item or not (data_root / item["wav"]).exists():
            counts["filter_missing_wav"] += 1
            continue
        if "emo_vec" not in item or not (data_root / item["emo_vec"]).exists():
            counts["filter_missing_emo"] += 1
            continue
        item["speaker_id"] = parse_speaker_id(item.get("source_file", ""))
        out.append(item)
    return out


def _build_pools(filtered, args):
    spk_samples = defaultdict(list)
    spk_neutral = defaultdict(list)
    nidx = neutral_score_index()
    for item in filtered:
        spk = item["speaker_id"]
        spk_samples[spk].append(item)
        is_neutral = item.get("emo_label") == NEUTRAL_LABEL
        if is_neutral and item["duration_sec"] >= args.min_neutral_duration:
            spk_neutral[spk].append(item)

    if args.allow_neutral_fallback:
        for spk, samples in spk_samples.items():
            if spk_neutral.get(spk):
                continue
            # Pick highest neutral_score sample with duration >= min_neutral_duration.
            candidates = [s for s in samples if s["duration_sec"] >= args.min_neutral_duration]
            if not candidates:
                continue
            candidates.sort(key=lambda s: (s.get("emo_scores") or [0] * len(EMOTION_CLASSES))[nidx], reverse=True)
            spk_neutral[spk].append(candidates[0])
    return spk_samples, spk_neutral


def _build_entry(item, neutral_pool):
    return {
        "id": item["id"],
        "speaker_id": item["speaker_id"],
        "source_file": item.get("source_file"),
        "wav": item["wav"],
        "text": item["text"],
        "emo_vec": item["emo_vec"],
        "emo_label": item.get("emo_label"),
        "emo_scores": item.get("emo_scores"),
        "duration_sec": item["duration_sec"],
        "neutral_pool": [s["id"] for s in neutral_pool],
    }


def _print_dist(label, dist, total):
    print(f"\nEmotion distribution ({label}):")
    if total == 0:
        print("  <empty>")
        return
    for cls in EMOTION_CLASSES:
        cnt = dist.get(cls, 0)
        print(f"  {cls:>10s}: {cnt:>6d} ({100 * cnt / total:5.1f}%)")
    other = sum(c for k, c in dist.items() if k not in EMOTION_CLASSES)
    if other:
        print(f"  <other> : {other}")


def _summarize(nums):
    if not nums:
        return None
    s = sorted(nums)
    return {
        "min": s[0],
        "max": s[-1],
        "mean": round(sum(s) / len(s), 2),
        "median": s[len(s) // 2],
        "p10": s[max(0, int(0.1 * len(s)) - 1)],
        "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--data_root", required=True, help="Directory containing chunks/, embeddings/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=20.0)
    parser.add_argument("--min_neutral_duration", type=float, default=2.0,
                        help="Minimum duration for a sample to qualify as a reference (voice identity needs enough audio)")
    parser.add_argument("--val_speaker_ratio", type=float, default=0.05)
    parser.add_argument("--allow_neutral_fallback", action="store_true",
                        help="If set, use the highest-neutral-score sample as a fallback for speakers without a labeled-neutral utterance")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()

    raw = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    counts["raw_total"] = len(raw)

    filtered = _filter(raw, data_root, args, counts)
    counts["after_filter"] = len(filtered)

    print(f"Raw: {counts['raw_total']}, after filter: {counts['after_filter']}")
    print(f"  non_verbal      : {counts['filter_non_verbal']}")
    print(f"  empty_text      : {counts['filter_empty_text']}")
    print(f"  too_short(<{args.min_duration}s) : {counts['filter_too_short']}")
    print(f"  too_long (>{args.max_duration}s): {counts['filter_too_long']}")
    print(f"  missing_wav     : {counts['filter_missing_wav']}")
    print(f"  missing_emo     : {counts['filter_missing_emo']}")

    spk_samples, spk_neutral = _build_pools(filtered, args)
    n_speakers = len(spk_samples)
    speakers_with_neutral = [s for s in spk_samples if spk_neutral.get(s)]
    n_with_neutral = len(speakers_with_neutral)
    excluded_no_neutral = n_speakers - n_with_neutral
    pct = 100 * excluded_no_neutral / n_speakers if n_speakers else 0.0
    print(f"\nSpeakers total                                  : {n_speakers}")
    print(f"  with usable neutral pool (>= {args.min_neutral_duration}s)        : {n_with_neutral}")
    print(f"  excluded (no neutral; fallback={args.allow_neutral_fallback}): {excluded_no_neutral} ({pct:.1f}%)")

    if pct > 20.0 and not args.allow_neutral_fallback:
        print(f"  WARNING: >20% of speakers excluded. Consider --allow_neutral_fallback or revisit data policy.")

    # Speaker split
    speakers_with_neutral_sorted = sorted(speakers_with_neutral)  # deterministic
    random.shuffle(speakers_with_neutral_sorted)
    n_val = max(1, int(round(len(speakers_with_neutral_sorted) * args.val_speaker_ratio))) if speakers_with_neutral_sorted else 0
    val_speakers = set(speakers_with_neutral_sorted[:n_val])
    train_speakers = set(speakers_with_neutral_sorted[n_val:])
    print(f"\nTrain speakers: {len(train_speakers)} | Val speakers: {len(val_speakers)}")

    # Build entries (drop excluded speakers)
    train_entries, val_entries = [], []
    for spk in train_speakers:
        for item in spk_samples[spk]:
            train_entries.append(_build_entry(item, spk_neutral[spk]))
    for spk in val_speakers:
        for item in spk_samples[spk]:
            val_entries.append(_build_entry(item, spk_neutral[spk]))

    train_path = output_dir / "manifest_train.jsonl"
    val_path = output_dir / "manifest_val.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for e in train_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for e in val_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Stats
    print(f"\n--- Manifest Statistics ---")
    print(f"Train samples: {len(train_entries)} | Val samples: {len(val_entries)}")

    train_spk_counts = Counter(e["speaker_id"] for e in train_entries)
    val_spk_counts = Counter(e["speaker_id"] for e in val_entries)
    train_per_spk = _summarize(list(train_spk_counts.values()))
    val_per_spk = _summarize(list(val_spk_counts.values()))
    print(f"Train: samples per speaker -- {train_per_spk}")
    print(f"Val  : samples per speaker -- {val_per_spk}")

    pool_sizes = [len(e["neutral_pool"]) for e in train_entries]
    pool_sum = _summarize(pool_sizes)
    print(f"Train: neutral_pool size   -- {pool_sum}")

    train_durs = [e["duration_sec"] for e in train_entries]
    print(f"Train: duration_sec        -- {_summarize(train_durs)}")

    train_dist = Counter(e["emo_label"] for e in train_entries)
    val_dist = Counter(e["emo_label"] for e in val_entries)
    _print_dist("train", train_dist, len(train_entries))
    _print_dist("val",   val_dist,   len(val_entries))

    # Neutral pool duration distribution (informative for reference quality)
    neutral_durs = []
    for spk, pool in spk_neutral.items():
        if spk in train_speakers or spk in val_speakers:
            neutral_durs.extend(p["duration_sec"] for p in pool)
    print(f"\nNeutral pool duration_sec across kept speakers -- {_summarize(neutral_durs)}")

    stats = {
        "raw_total": counts["raw_total"],
        "after_filter": counts["after_filter"],
        "filter_breakdown": {k: v for k, v in counts.items() if k.startswith("filter_")},
        "speakers_total": n_speakers,
        "speakers_with_neutral": n_with_neutral,
        "speakers_excluded_no_neutral": excluded_no_neutral,
        "speakers_excluded_pct": round(pct, 2),
        "allow_neutral_fallback": args.allow_neutral_fallback,
        "train_speakers": len(train_speakers),
        "val_speakers": len(val_speakers),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "train_per_speaker": train_per_spk,
        "val_per_speaker": val_per_spk,
        "train_neutral_pool_size": pool_sum,
        "train_duration_summary": _summarize(train_durs),
        "neutral_pool_duration_summary": _summarize(neutral_durs),
        "emotion_distribution_train": dict(train_dist),
        "emotion_distribution_val": dict(val_dist),
        "args": vars(args),
    }
    with open(output_dir / "manifest_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nWrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {output_dir / 'manifest_stats.json'}")


if __name__ == "__main__":
    main()
