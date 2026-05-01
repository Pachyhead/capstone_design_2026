# coding=utf-8
"""Add audio_codes to a manifest using Qwen3TTSTokenizer.

The dataset/training code expects each manifest line to carry an 'audio_codes'
field (16-layer RVQ tokens of the target waveform). This script batches the
target wav files through Qwen3TTSTokenizer.encode and writes a new jsonl with
the codes appended.

Usage:
    python -m qwen_tts.scripts.add_audio_codes \\
        --input_jsonl manifests/manifest_train.jsonl \\
        --output_jsonl manifests/manifest_train.codes.jsonl \\
        --data_root /path/to/dataset_root
"""
import argparse
import json
from pathlib import Path

from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--tokenizer_model_path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model_path, device_map=args.device)
    data_root = Path(args.data_root)

    items = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    out = []
    batch_items, batch_paths = [], []

    def _flush():
        if not batch_items:
            return
        res = tokenizer.encode(batch_paths)
        for code, it in zip(res.audio_codes, batch_items):
            it["audio_codes"] = code.cpu().tolist()
            out.append(it)
        batch_items.clear()
        batch_paths.clear()

    for item in items:
        batch_items.append(item)
        batch_paths.append(str(data_root / item["wav"]))
        if len(batch_items) >= args.batch_size:
            _flush()
            print(f"  ... {len(out)} / {len(items)}")
    _flush()

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} entries with audio_codes -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
