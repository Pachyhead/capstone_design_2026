# coding=utf-8
"""Inference script for emotion-conditioned Qwen3-TTS.

Generates speech for a given text + reference audio + emotion vector.
Supports running a small grid (one emotion per output wav) so you can A/B listen.

Usage (single emotion):
    python3 -m qwen_tts.scripts.infer_emotion \\
        --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --projector_dir   /home/cap/data/processed/checkpoint-epoch-2 \\
        --ref_audio       /path/to/ref_neutral.wav \\
        --ref_text        "참고 음성에 해당하는 텍스트" \\
        --text            "합성할 문장입니다" \\
        --emotion_npy     /path/to/some_angry.npy \\
        --out_wav         out_angry.wav

Usage (grid over multiple emotion npys -- one wav per file):
    python3 -m qwen_tts.scripts.infer_emotion \\
        --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --projector_dir   /home/cap/data/processed/checkpoint-epoch-2 \\
        --ref_audio       /path/to/ref_neutral.wav \\
        --ref_text        "..." \\
        --text            "..." \\
        --emotion_npys    /path/to/angry.npy /path/to/happy.npy /path/to/sad.npy \\
        --out_dir         out_grid

Usage (baseline comparison -- no emotion projector applied):
    add --no_emotion to skip emotion injection (output should be near-baseline).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts.inference.emotion_loader import load_emotion_projector
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _load_emotion_npy(path: str) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return torch.from_numpy(arr.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--projector_dir", required=True,
                        help="checkpoint-epoch-{N} directory produced by sft_emotion_12hz.py")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--ref_audio", required=True, help="Reference audio path (24kHz wav)")
    parser.add_argument("--ref_text", default=None, help="Reference text (enables ICL mode if given)")
    parser.add_argument("--x_vector_only_mode", action="store_true",
                        help="If set, ignore ref_text and use speaker x-vector only")
    # Emotion supply (mutually exclusive logical group)
    parser.add_argument("--emotion_npy", default=None, help="Single emotion .npy -> one out wav")
    parser.add_argument("--emotion_npys", nargs="+", default=None,
                        help="Multiple emotion .npy paths -> one wav per file (use with --out_dir)")
    parser.add_argument("--no_emotion", action="store_true",
                        help="Run baseline (no projector applied). Useful for A/B comparison")
    # Output
    parser.add_argument("--out_wav", default=None, help="Output wav path (single emotion mode)")
    parser.add_argument("--out_dir", default=None, help="Output directory (grid mode)")
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

    # Mode validation
    grid_mode = args.emotion_npys is not None and len(args.emotion_npys) > 0
    if not args.no_emotion and not args.emotion_npy and not grid_mode:
        raise SystemExit("Provide one of --emotion_npy / --emotion_npys / --no_emotion")
    if grid_mode and args.out_dir is None:
        raise SystemExit("--out_dir required when using --emotion_npys")
    if not grid_mode and not args.no_emotion and args.out_wav is None:
        raise SystemExit("--out_wav required for single emotion mode")
    if args.no_emotion and args.out_wav is None:
        raise SystemExit("--out_wav required for --no_emotion mode")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Load model. device_map (vs manual .to) installs accelerate hooks so input
    # tensors built on CPU (input_ids, ref_ids, etc.) are automatically moved to
    # the model's device at every forward -- avoids "tensors on different devices"
    # at embedding/index_select time.
    print(f"Loading {args.init_model_path} ...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map=args.device,
    )

    # Attach projector unless --no_emotion
    if not args.no_emotion:
        print(f"Loading emotion projector from {args.projector_dir} ...")
        load_emotion_projector(qwen3tts.model, args.projector_dir, device=torch.device(args.device), dtype=dtype)

    # ----- Generate -----
    def _run_one(emo_tensor, out_path):
        emotion_kwarg = None if emo_tensor is None else emo_tensor.unsqueeze(0)
        wavs, sr = qwen3tts.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only_mode,
            emotion_vec=emotion_kwarg,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        wav = wavs[0]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, wav, sr)
        print(f"  -> wrote {out_path}  (len={wav.shape[0]/sr:.2f}s @ {sr}Hz)")

    if args.no_emotion:
        print("[baseline] no emotion injection")
        _run_one(None, args.out_wav)
    elif grid_mode:
        print(f"[grid] {len(args.emotion_npys)} emotions -> {args.out_dir}")
        out_root = Path(args.out_dir)
        for path in args.emotion_npys:
            stem = Path(path).stem
            emo = _load_emotion_npy(path)
            print(f"  emotion: {stem} (shape {tuple(emo.shape)})")
            _run_one(emo, str(out_root / f"{stem}.wav"))
    else:
        print(f"[single] emotion: {args.emotion_npy}")
        emo = _load_emotion_npy(args.emotion_npy)
        _run_one(emo, args.out_wav)

    print("Done.")


if __name__ == "__main__":
    main()
