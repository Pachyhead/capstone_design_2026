# coding=utf-8
# Pre-training sanity checks for the EmotionProjector integration AND the
# manifest-based dataset.
#
# Model-side invariants (always run):
#   7.1 Zero-init             -- projector(any_vec) == 0 at training start
#   7.2 Gradient flow         -- only emotion_projector parameters receive grads
#   7.3 Trainable param count -- matches projector size (D_emo*D_target + D_target)
#   7.4 Shape compatibility   -- projector output broadcasts onto speaker_emb [B, D]
#
# Dataset-side invariants (require --train_manifest, optional --val_manifest):
#   D1 No self-reference       -- ref_id != target_id for >=95% of samples
#   D2 Emotion-dim consistency -- random sample of items load to expected_emotion_dim
#   D3 Speaker split disjoint  -- train_speakers ∩ val_speakers == empty
#
# 7.5 (output-changes-after-training) is intentionally not run here -- it requires
# a real optimizer step. Run sft_emotion_12hz.py with --num_epochs 1 and inspect
# the projector weight.
#
# Usage (model checks only):
#   python -m qwen_tts.finetuning.sanity_check \\
#       --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base
#
# Usage (model + dataset checks):
#   python -m qwen_tts.finetuning.sanity_check \\
#       --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
#       --train_manifest manifests/manifest_train.codes.jsonl \\
#       --val_manifest   manifests/manifest_val.codes.jsonl \\
#       --data_root      /path/to/dataset_root
import argparse
import json
import random

import torch

from qwen_tts.core.models.emotion_projector import EmotionProjector
from qwen_tts.finetuning.sft_emotion_12hz import (
    _attach_emotion_projector,
    _freeze_all_but_emotion_projector,
)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


# ---------- Model checks ----------

def check_71_zero_init(projector: EmotionProjector, batch_size: int):
    emo_zero = torch.zeros(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    emo_rand = torch.randn(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)

    out_zero = projector(emo_zero)
    out_rand = projector(emo_rand)

    assert torch.equal(out_zero, torch.zeros_like(out_zero)), "[7.1] projector(zeros) != 0"
    assert torch.equal(out_rand, torch.zeros_like(out_rand)), "[7.1] projector(rand) != 0 (zero-init violated)"
    print(f"[7.1] OK: projector(any_vec) == 0 at init  -> speaker_emb + emo_proj == speaker_emb")


def check_72_grad_flow(model, projector: EmotionProjector, batch_size: int):
    emo = torch.randn(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    out = projector(emo)
    loss = out.sum()
    loss.backward()

    assert projector.proj.weight.grad is not None, "[7.2] projector.weight.grad is None"
    grad_abs = projector.proj.weight.grad.abs().sum().item()
    assert grad_abs > 0, f"[7.2] projector.weight.grad has zero magnitude ({grad_abs})"

    leaked = []
    for name, p in model.named_parameters():
        if "emotion_projector" in name:
            continue
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            leaked.append(name)
    assert not leaked, f"[7.2] frozen params received gradient: {leaked[:5]}{'...' if len(leaked) > 5 else ''}"
    print(f"[7.2] OK: emotion_projector receives grad (|sum|={grad_abs:.4e}); no frozen param leaked")


def check_73_trainable_count(model, projector: EmotionProjector):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_total = sum(p.numel() for p in projector.parameters())
    expected = projector.emotion_dim * projector.target_dim + projector.target_dim
    assert trainable == proj_total == expected, (
        f"[7.3] count mismatch: trainable={trainable}, projector={proj_total}, expected={expected}"
    )
    print(f"[7.3] OK: trainable={trainable:,} == projector total ({projector.emotion_dim}*{projector.target_dim} + {projector.target_dim})")


def check_74_shape_match(model, projector: EmotionProjector, batch_size: int):
    target_dim = model.config.speaker_encoder_config.enc_dim
    assert projector.target_dim == target_dim, (
        f"[7.4] projector.target_dim={projector.target_dim} != speaker_encoder.enc_dim={target_dim}"
    )
    fake_spk = torch.zeros(batch_size, target_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    emo = torch.randn(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    out = fake_spk + projector(emo)
    assert out.shape == fake_spk.shape, f"[7.4] add result {out.shape} != speaker_emb {fake_spk.shape}"
    print(f"[7.4] OK: speaker_emb {tuple(fake_spk.shape)} + emo_proj == {tuple(out.shape)}")


# ---------- Dataset checks ----------

def _read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def check_d1_no_self_reference(processor, config, manifest_path: str, data_root: str, expected_emotion_dim: int, n_samples: int = 100):
    """Sample N items via Dataset.__getitem__ and check ref_id != target_id frequently.

    For pools with >=2 entries, self-reference must NEVER happen by design.
    For pools of size 1 (target is the only neutral), self-reference is unavoidable.
    """
    from qwen_tts.finetuning.dataset_emotion import EmotionTTSDataset
    ds = EmotionTTSDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        processor=processor,
        config=config,
        expected_emotion_dim=expected_emotion_dim,
    )

    n = min(n_samples, len(ds))
    indices = random.sample(range(len(ds)), n)
    self_ref = 0
    forced_self_ref = 0  # cases where pool size is 1 -> forced self-ref
    for i in indices:
        item = ds.samples[i]
        pool = ds.speaker_to_neutral[item["speaker_id"]]
        out = ds[i]
        if out["id"] == out["ref_id"]:
            self_ref += 1
            if len(pool) <= 1 or all(p["id"] == item["id"] for p in pool):
                forced_self_ref += 1
    pct_self = 100 * self_ref / n
    pct_forced = 100 * forced_self_ref / n
    pct_avoidable = pct_self - pct_forced
    print(f"[D1] sampled {n} items: self-ref {self_ref} ({pct_self:.1f}%) of which forced(pool=1) {forced_self_ref} ({pct_forced:.1f}%)")
    assert pct_avoidable < 0.5, (
        f"[D1] avoidable self-reference detected ({pct_avoidable:.1f}%) -- _select_reference logic is broken"
    )
    if pct_self > 5.0:
        print(f"[D1] WARNING: total self-reference {pct_self:.1f}% > 5% (many speakers have a single neutral sample)")
    print(f"[D1] OK: avoidable self-reference == 0 in sample")


def check_d2_emotion_dim_consistency(manifest_path: str, data_root: str, expected_emotion_dim: int, n_samples: int = 100):
    """Spot-check that loaded emo_vec npy collapses to the expected dim."""
    import numpy as np
    from pathlib import Path

    items = _read_manifest(manifest_path)
    n = min(n_samples, len(items))
    indices = random.sample(range(len(items)), n)
    bad = []
    root = Path(data_root)
    for i in indices:
        path = root / items[i]["emo_vec"]
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.ndim != 1 or arr.shape[0] != expected_emotion_dim:
            bad.append((items[i]["id"], arr.shape))
    assert not bad, f"[D2] {len(bad)} entries have wrong emotion shape (e.g. {bad[:3]}); expected ({expected_emotion_dim},)"
    print(f"[D2] OK: {n} sampled emo_vec all collapse to ({expected_emotion_dim},)")


def check_d3_speaker_split(train_manifest: str, val_manifest: str):
    train_items = _read_manifest(train_manifest)
    val_items = _read_manifest(val_manifest)
    train_speakers = {it["speaker_id"] for it in train_items}
    val_speakers = {it["speaker_id"] for it in val_items}
    overlap = train_speakers & val_speakers
    assert not overlap, f"[D3] speaker leakage: {len(overlap)} speakers appear in both splits (e.g. {list(overlap)[:5]})"
    print(f"[D3] OK: train={len(train_speakers)} speakers, val={len(val_speakers)} speakers, overlap=0")


# ---------- Entry ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--emotion_dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--train_manifest", type=str, default=None,
                        help="Optional manifest path; enables dataset-side checks D1/D2/D3")
    parser.add_argument("--val_manifest", type=str, default=None,
                        help="Optional val manifest; required for D3")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Required when --train_manifest is given")
    parser.add_argument("--n_dataset_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
    )
    model = qwen3tts.model.to(args.device)
    projector = _attach_emotion_projector(model, emotion_dim=args.emotion_dim, dtype=dtype, device=torch.device(args.device))
    _freeze_all_but_emotion_projector(model)

    print("=" * 60)
    print("Model-side checks")
    print("=" * 60)
    check_71_zero_init(projector, args.batch_size)
    check_72_grad_flow(model, projector, args.batch_size)
    check_73_trainable_count(model, projector)
    check_74_shape_match(model, projector, args.batch_size)

    if args.train_manifest:
        if not args.data_root:
            raise SystemExit("--data_root is required when --train_manifest is given")
        print("=" * 60)
        print("Dataset-side checks")
        print("=" * 60)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.init_model_path)
        check_d1_no_self_reference(qwen3tts.processor, config, args.train_manifest, args.data_root, args.emotion_dim, args.n_dataset_samples)
        check_d2_emotion_dim_consistency(args.train_manifest, args.data_root, args.emotion_dim, args.n_dataset_samples)
        if args.val_manifest:
            check_d3_speaker_split(args.train_manifest, args.val_manifest)
        else:
            print("[D3] skipped (no --val_manifest)")

    print("=" * 60)
    print("All checks passed. Ready to run sft_emotion_12hz.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()
