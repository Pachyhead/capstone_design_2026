# coding=utf-8
"""End-to-end evaluation of emotion fine-tuning.

For each (ref, test) pair sampled from the manifest:
  1. Synthesize TWO outputs:
     - baseline           : same model, emotion_vec=None (projector path skipped)
     - emotion-conditioned: same model + emotion_vec from the test sample
     Both use the SAME ref_audio (same speaker, neutral utterance, different text).
  2. Run Emotion2Vec on both wavs (utterance-level, [1024]).
  3. Compare each predicted emotion to the test sample's ground-truth emo_vec.
  4. Aggregate: cos(predicted, gt), top-1 label match, baseline vs emotion delta.

Pair construction:
  - test  : a manifest entry with a non-trivial emotion label (default skips
            neutral/other/unknown). Provides text, gt emo_vec, gt label.
  - ref   : a same-speaker neutral entry from test["neutral_pool"], with
            different text from test (so the model can't trivially copy ref).

Outputs (under --out_dir):
  - per_pair.jsonl          : one row per pair (final, written at end)
  - per_pair.partial.jsonl  : streamed during the run for crash safety
  - summary.json            : aggregate metrics

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python3 -u -m qwen_tts.scripts.eval_emotion \\
        --projector_dir /home/cap/data/processed/checkpoint-epoch-2 \\
        --manifest      /home/cap/data/processed/model_train/manifest_val.codes.jsonl \\
        --data_root     /home/cap/data/processed \\
        --out_dir       /home/cap/data/processed/eval_emotion
"""
import argparse
import gc
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

from qwen_tts.core.models.lora import set_lora_enabled
from qwen_tts.inference.emotion_loader import load_emotion_projector
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Use the project's existing EmotionExtractor wrapper (uses iic/emotion2vec_plus_large).
_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT / "src"))
from preprocess_pipeline.config import Config  # noqa: E402
from preprocess_pipeline.emotion import EmotionExtractor  # noqa: E402

EMOTION_LABELS = (
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
)


# ---------- helpers ----------

def _normalize_label(raw: str) -> str:
    """Emotion2Vec sometimes returns labels like '生气/angry' or '5/neutral'."""
    s = raw.strip().lower()
    if "/" in s:
        s = s.split("/")[-1].strip()
    return s


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _load_emo_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr.astype(np.float32)


def _gt_score(labels_list, scores_list, gt):
    for lb, sc in zip(labels_list, scores_list):
        if _normalize_label(lb) == gt:
            return float(sc)
    return None


def _fmt_hms(s: float) -> str:
    s = int(s)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- pair construction ----------

def build_pairs(items, seed: int, exclude_emotions, max_per_speaker: int = 2):
    """Pick (ref, test) pairs where ref is a same-speaker neutral with different text.

    - test must have an emo_label NOT in exclude_emotions (default: skip neutral/other/unknown)
    - ref is sampled from test["neutral_pool"] with id != test.id and text != test.text
    - At most `max_per_speaker` test samples per speaker (avoids one chatty speaker dominating)
    - Returns ALL eligible pairs (no count cap).
    """
    id_to_item = {it["id"]: it for it in items}
    candidates = [
        it for it in items
        if it.get("emo_label") not in exclude_emotions and it.get("neutral_pool")
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    seen_keys = set()
    per_speaker = defaultdict(int)
    pairs = []
    for test in candidates:
        spk = test.get("speaker_id")
        if per_speaker[spk] >= max_per_speaker:
            continue

        ref_candidates = [rid for rid in test["neutral_pool"] if rid != test["id"]]
        rng.shuffle(ref_candidates)
        ref = None
        for rid in ref_candidates:
            cand = id_to_item.get(rid)
            if cand is None:
                continue
            if cand["text"].strip() == test["text"].strip():
                continue
            ref = cand
            break
        if ref is None:
            continue

        key = (ref["id"], test["id"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        per_speaker[spk] += 1
        pairs.append((ref, test))
    return pairs


# ---------- synthesis ----------

def synthesize(qwen3tts, text, ref_audio_path, ref_text, language, emotion_vec, gen_kwargs):
    emotion_kwarg = None
    if emotion_vec is not None:
        emotion_kwarg = torch.from_numpy(np.asarray(emotion_vec, dtype=np.float32)).unsqueeze(0)  # [1, D]
    wavs, sr = qwen3tts.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        emotion_vec=emotion_kwarg,
        do_sample=True,
        **gen_kwargs,
    )
    return wavs[0], sr


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--projector_dir", required=True,
                        help="checkpoint-epoch-{N} directory from sft_emotion_12hz.py")
    parser.add_argument("--manifest", required=True,
                        help="Validation manifest jsonl (audio_codes not required)")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_per_speaker", type=int, default=2,
                        help="Cap samples per speaker so one chatty speaker doesn't dominate")
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--exclude_emotions", nargs="+",
                        default=["neutral", "other", "unknown"],
                        help="Skip test samples with these labels (default focuses on emotional content)")
    parser.add_argument("--seed", type=int, default=42)
    # sampling
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    # runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", default="sdpa")
    parser.add_argument("--emotion_device", default="cuda:0",
                        help="GPU for Emotion2Vec (use a different device than --device if VRAM is tight)")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print running summary every N pairs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "per_pair.partial.jsonl"
    if partial_path.exists():
        partial_path.unlink()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 1. Load manifest, build pairs ----
    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} manifest items from {args.manifest}", flush=True)
    pairs = build_pairs(
        items,
        seed=args.seed,
        exclude_emotions=tuple(args.exclude_emotions),
        max_per_speaker=args.max_per_speaker,
    )
    print(f"Built {len(pairs)} (ref, test) pairs", flush=True)
    if not pairs:
        raise SystemExit(
            "No pairs constructed. Check manifest filters / neutral_pool / emotion labels."
        )

    # ---- 2. Load Qwen3TTS + projector ----
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"Loading {args.init_model_path} (device_map={args.device}) ...", flush=True)
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map=args.device,
    )
    print(f"Attaching emotion projector from {args.projector_dir} ...", flush=True)
    load_emotion_projector(qwen3tts.model, args.projector_dir, device=torch.device(args.device), dtype=dtype)

    # Detect whether the loaded checkpoint also injected LoRA into the talker.
    # If yes, the baseline branch must run with LoRA OFF so it represents the true
    # upstream model, not "LoRA-modified-talker without emotion".
    n_lora = set_lora_enabled(qwen3tts.model, True)  # canonical state = on
    if n_lora > 0:
        print(f"[lora-toggle] {n_lora} LoRA modules detected -- baseline pass will disable them", flush=True)
    else:
        print(f"[lora-toggle] no LoRA modules (Stage-1 checkpoint); baseline = no projector path only", flush=True)

    # ---- 3. Load Emotion2Vec ----
    print(f"Loading Emotion2Vec on {args.emotion_device} ...", flush=True)
    extractor = EmotionExtractor(Config(emotion_device=args.emotion_device))

    data_root = Path(args.data_root)
    gen_kwargs = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    # ---- 4. Per-pair: synthesize + extract + score ----
    rows = []
    total = len(pairs)
    t_start = time.time()

    print("\n" + "=" * 72, flush=True)
    print(f" Evaluating {total} pairs (partial -> {partial_path.name})", flush=True)
    print("=" * 72, flush=True)

    for i, (ref, test) in enumerate(pairs):
        t0 = time.time()
        ref_path = str(data_root / ref["wav"])
        gt_emo_vec = _load_emo_npy(str(data_root / test["emo_vec"]))
        gt_label = _normalize_label(test["emo_label"])

        print(f"\n[{i+1:>4d}/{total}] ref={ref['id']} ({ref['emo_label']}) -> "
              f"test={test['id']} ({gt_label})", flush=True)
        print(f"   ref_text:  {ref['text'][:60]}", flush=True)
        print(f"   test_text: {test['text'][:60]}", flush=True)

        # Synthesize
        # Per-pair seeding: baseline and emotion branches start from the same RNG
        # state so the only difference is the emotion projector path, not sampling noise.
        # Baseline branch: LoRA OFF + emotion_vec=None -> bit-exact upstream model.
        # Emotion branch:  LoRA ON  + emotion_vec=gt   -> our fine-tuned model.
        try:
            t_syn = time.time()
            set_lora_enabled(qwen3tts.model, False)
            torch.manual_seed(args.seed + i)
            wav_b, sr = synthesize(qwen3tts, test["text"], ref_path, ref["text"], args.language,
                                    emotion_vec=None, gen_kwargs=gen_kwargs)
            set_lora_enabled(qwen3tts.model, True)
            torch.manual_seed(args.seed + i)
            wav_e, _ = synthesize(qwen3tts, test["text"], ref_path, ref["text"], args.language,
                                  emotion_vec=gt_emo_vec, gen_kwargs=gen_kwargs)
            syn_sec = time.time() - t_syn

            # Extract emotion (Emotion2Vec uses 16kHz; resample for safety)
            t_ex = time.time()
            wav_b_16k = librosa.resample(wav_b.astype(np.float32), orig_sr=sr, target_sr=16000)
            wav_e_16k = librosa.resample(wav_e.astype(np.float32), orig_sr=sr, target_sr=16000)
            emo_b, labels_b, scores_b = extractor.extract(wav_b_16k, 16000)
            emo_e, labels_e, scores_e = extractor.extract(wav_e_16k, 16000)
            ex_sec = time.time() - t_ex
        except torch.cuda.OutOfMemoryError as e:
            print(f"   [SKIP] OOM on this pair: {e}", flush=True)
            _free_gpu()
            continue

        cos_b = _cosine(emo_b, gt_emo_vec)
        cos_e = _cosine(emo_e, gt_emo_vec)
        delta = cos_e - cos_b

        top_b = _normalize_label(labels_b[int(np.argmax(scores_b))])
        top_e = _normalize_label(labels_e[int(np.argmax(scores_e))])
        match_b = (top_b == gt_label)
        match_e = (top_e == gt_label)

        row = {
            "pair_idx": i,
            "ref_id": ref["id"],
            "test_id": test["id"],
            "speaker_id": ref.get("speaker_id"),
            "ref_text": ref["text"],
            "test_text": test["text"],
            "gt_emo_label": gt_label,
            "predicted_baseline_label": top_b,
            "predicted_emotion_label": top_e,
            "match_baseline": match_b,
            "match_emotion": match_e,
            "cos_baseline_to_gt": cos_b,
            "cos_emotion_to_gt": cos_e,
            "cos_delta": delta,
            "gt_label_score_baseline": _gt_score(labels_b, scores_b, gt_label),
            "gt_label_score_emotion": _gt_score(labels_e, scores_e, gt_label),
        }
        rows.append(row)

        pair_sec = time.time() - t0
        print(f"   pred  baseline={top_b:>10s}  emotion={top_e:>10s}  gt={gt_label}", flush=True)
        print(f"   cos(b,gt)={cos_b:.4f}  cos(e,gt)={cos_e:.4f}  Δ={delta:+.4f}  "
              f"match_b={match_b}  match_e={match_e}", flush=True)
        print(f"   timing: synth={syn_sec:.1f}s  emo2vec={ex_sec:.1f}s  total={pair_sec:.1f}s", flush=True)

        # Append partial line for crash safety
        with open(partial_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Free per-pair tensors and reclaim GPU memory to prevent fragmentation OOM
        del wav_b, wav_e, wav_b_16k, wav_e_16k, emo_b, emo_e
        _free_gpu()

        # Running summary
        done = i + 1
        if done % args.log_every == 0 or done == total:
            cos_b_run = float(np.mean([r["cos_baseline_to_gt"] for r in rows]))
            cos_e_run = float(np.mean([r["cos_emotion_to_gt"] for r in rows]))
            delta_run = cos_e_run - cos_b_run
            mb = sum(r["match_baseline"] for r in rows)
            me = sum(r["match_emotion"] for r in rows)
            elapsed = time.time() - t_start
            avg_per = elapsed / done
            eta = avg_per * (total - done)
            print(f"\n   --- running [{done}/{total}] (kept={len(rows)}) | "
                  f"cos_b={cos_b_run:.4f} cos_e={cos_e_run:.4f} Δ={delta_run:+.4f} | "
                  f"match_b={mb}/{len(rows)} ({100*mb/len(rows):.0f}%) "
                  f"match_e={me}/{len(rows)} ({100*me/len(rows):.0f}%) | "
                  f"elapsed={_fmt_hms(elapsed)} ETA={_fmt_hms(eta)} avg={avg_per:.1f}s/pair ---",
                  flush=True)

    # ---- 5. Aggregate ----
    if not rows:
        raise SystemExit("No successful pairs -- nothing to aggregate.")

    n = len(rows)
    cos_b_arr = np.array([r["cos_baseline_to_gt"] for r in rows])
    cos_e_arr = np.array([r["cos_emotion_to_gt"] for r in rows])
    delta_arr = cos_e_arr - cos_b_arr
    match_b_n = sum(r["match_baseline"] for r in rows)
    match_e_n = sum(r["match_emotion"] for r in rows)
    delta_pos_n = int((delta_arr > 0).sum())

    by_label = defaultdict(lambda: {"n": 0, "match_b": 0, "match_e": 0, "cos_b": 0.0, "cos_e": 0.0})
    for r in rows:
        d = by_label[r["gt_emo_label"]]
        d["n"] += 1
        d["match_b"] += int(r["match_baseline"])
        d["match_e"] += int(r["match_emotion"])
        d["cos_b"] += r["cos_baseline_to_gt"]
        d["cos_e"] += r["cos_emotion_to_gt"]
    per_emotion = {
        lb: {
            "n": d["n"],
            "top1_match_baseline_rate": d["match_b"] / d["n"],
            "top1_match_emotion_rate":  d["match_e"] / d["n"],
            "mean_cos_baseline_to_gt":  d["cos_b"] / d["n"],
            "mean_cos_emotion_to_gt":   d["cos_e"] / d["n"],
        }
        for lb, d in by_label.items()
    }

    summary = {
        "n_pairs": n,
        "manifest": str(args.manifest),
        "projector_dir": str(args.projector_dir),
        "init_model_path": args.init_model_path,
        "exclude_emotions": list(args.exclude_emotions),
        "mean_cos_baseline_to_gt": float(cos_b_arr.mean()),
        "mean_cos_emotion_to_gt": float(cos_e_arr.mean()),
        "mean_delta": float(delta_arr.mean()),
        "median_delta": float(np.median(delta_arr)),
        "delta_positive_count": delta_pos_n,
        "delta_positive_rate": delta_pos_n / n,
        "top1_match_baseline_rate": match_b_n / n,
        "top1_match_emotion_rate": match_e_n / n,
        "predicted_label_dist_baseline": dict(Counter(r["predicted_baseline_label"] for r in rows)),
        "predicted_label_dist_emotion":  dict(Counter(r["predicted_emotion_label"]  for r in rows)),
        "gt_label_dist": dict(Counter(r["gt_emo_label"] for r in rows)),
        "per_emotion_breakdown": per_emotion,
    }

    print("\n" + "=" * 72, flush=True)
    print(" Summary", flush=True)
    print("=" * 72, flush=True)
    print(f"  n_pairs                       : {n}", flush=True)
    print(f"  mean cos(baseline_pred, gt)   : {summary['mean_cos_baseline_to_gt']:.4f}", flush=True)
    print(f"  mean cos(emotion_pred,  gt)   : {summary['mean_cos_emotion_to_gt']:.4f}", flush=True)
    print(f"  mean Δ (emotion - baseline)   : {summary['mean_delta']:+.4f}", flush=True)
    print(f"  pairs where Δ > 0             : {delta_pos_n} / {n} ({100*delta_pos_n/n:.1f}%)", flush=True)
    print(f"  top-1 match (baseline)        : {match_b_n} / {n} ({100*match_b_n/n:.1f}%)", flush=True)
    print(f"  top-1 match (emotion)         : {match_e_n} / {n} ({100*match_e_n/n:.1f}%)", flush=True)
    print()
    print("  Per-emotion breakdown:", flush=True)
    for lb, d in per_emotion.items():
        print(f"    {lb:>10s}  n={d['n']:>3d}  "
              f"match_b={d['top1_match_baseline_rate']*100:5.1f}%  "
              f"match_e={d['top1_match_emotion_rate']*100:5.1f}%  "
              f"cos_b={d['mean_cos_baseline_to_gt']:.3f}  "
              f"cos_e={d['mean_cos_emotion_to_gt']:.3f}", flush=True)
    print("=" * 72, flush=True)

    with open(out_dir / "per_pair.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_dir / 'per_pair.jsonl'}", flush=True)
    print(f"Wrote: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
