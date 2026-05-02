# coding=utf-8
"""Baseline Qwen3-TTS-12Hz-1.7B-Base voice clone -- no emotion fine-tuning applied.

Use this side-by-side with infer_emotion.py to A/B compare:
- baseline: this script
- emotion-conditioned: infer_emotion.py with --emotion_npy

Usage (single output):
    python3 -m qwen_tts.scripts.infer_baseline \\
        --ref_audio /path/to/ref.wav \\
        --ref_text  "참고 음성에 해당하는 텍스트" \\
        --text      "합성할 문장입니다" \\
        --out_wav   out_baseline.wav

Usage (multiple texts, one wav per text):
    python3 -m qwen_tts.scripts.infer_baseline \\
        --ref_audio /path/to/ref.wav \\
        --ref_text  "..." \\
        --texts "안녕하세요" "오늘은 기분이 좋네요" "정말 화가 납니다" \\
        --out_dir out_baseline/
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--ref_audio", required=True, help="Reference audio path (24kHz wav)")
    parser.add_argument("--ref_text", default=None,
                        help="Reference text (enables ICL mode if given). "
                             "If omitted, falls back to x-vector-only mode.")
    parser.add_argument("--x_vector_only_mode", action="store_true",
                        help="If set, ignore ref_text and use speaker x-vector only")
    # Text input -- mutually exclusive (one or many)
    parser.add_argument("--text", default=None, help="Single text to synthesize")
    parser.add_argument("--texts", nargs="+", default=None, help="Multiple texts (one wav per text)")
    # Output
    parser.add_argument("--out_wav", default=None, help="Output wav path (single text mode)")
    parser.add_argument("--out_dir", default=None, help="Output directory (multi-text mode)")
    # Sampling
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    # Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", default="sdpa")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.text is None and not args.texts:
        raise SystemExit("Provide --text or --texts")
    if args.text and args.texts:
        raise SystemExit("Use --text OR --texts, not both")
    if args.text and args.out_wav is None:
        raise SystemExit("--out_wav required for --text")
    if args.texts and args.out_dir is None:
        raise SystemExit("--out_dir required for --texts")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # device_map (vs manual .to) installs accelerate hooks so CPU-side input
    # tensors are auto-moved to the model device at forward time.
    print(f"Loading {args.init_model_path} (no emotion projector) ...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map=args.device,
    )
    # Explicitly verify no projector is attached -- baseline should NOT have one.
    assert getattr(qwen3tts.model, "emotion_projector", None) is None, \
        "Loaded model unexpectedly carries an emotion_projector; this script is for baseline only."

    def _run_one(text: str, out_path: str):
        wavs, sr = qwen3tts.generate_voice_clone(
            text=text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only_mode,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        wav = wavs[0]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, wav, sr)
        print(f"  -> wrote {out_path}  (len={wav.shape[0]/sr:.2f}s @ {sr}Hz)  text=\"{text[:40]}\"")

    if args.text:
        _run_one(args.text, args.out_wav)
    else:
        out_root = Path(args.out_dir)
        for i, t in enumerate(args.texts):
            safe = "".join(c if c.isalnum() else "_" for c in t)[:40] or f"line{i}"
            _run_one(t, str(out_root / f"{i:02d}_{safe}.wav"))

    print("Done.")


if __name__ == "__main__":
    main()
