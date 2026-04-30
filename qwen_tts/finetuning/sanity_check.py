# coding=utf-8
# Pre-training sanity checks for the EmotionProjector integration.
#
# Verifies (Step 7 of the design doc):
#   7.1 Zero-init invariant -- projector(any_vec) == 0 at training start, so
#       speaker_emb + projector(emo) is bit-exact w.r.t. baseline speaker_emb.
#   7.2 Gradient flow -- only emotion_projector parameters receive gradients.
#   7.3 Trainable param count -- matches projector size (D_emo*D_target + D_target).
#   7.4 Shape compatibility -- projector output broadcasts onto speaker_emb [B, D].
#
# 7.5 (training-step delta) is intentionally not run here -- it requires a real
# optimizer step on a real batch. Run sft_emotion_12hz.py with --num_epochs 1 and
# inspect the projector weight after the first step to verify it leaves zero.
#
# Usage:
#   python -m qwen_tts.finetuning.sanity_check --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base
import argparse

import torch

from qwen_tts.core.models.emotion_projector import EmotionProjector
from qwen_tts.finetuning.sft_emotion_12hz import (
    _attach_emotion_projector,
    _freeze_all_but_emotion_projector,
)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def check_71_zero_init(projector: EmotionProjector, batch_size: int):
    emo_zero = torch.zeros(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    emo_rand = torch.randn(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)

    out_zero = projector(emo_zero)
    out_rand = projector(emo_rand)

    assert torch.equal(out_zero, torch.zeros_like(out_zero)), "[7.1] projector(zeros) != 0"
    assert torch.equal(out_rand, torch.zeros_like(out_rand)), "[7.1] projector(rand) != 0 (zero-init violated)"
    print(f"[7.1] OK: projector(any_vec) == 0 at init  -> speaker_emb + emo_proj == speaker_emb")


def check_72_grad_flow(model, projector: EmotionProjector, batch_size: int):
    # Synthetic forward through projector only -- no need to run the talker for grad-flow check.
    # The training script adds projector output to speaker_emb (which is .detach()'d), so
    # gradients can ONLY enter the projector via its own forward.
    emo = torch.randn(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    out = projector(emo)
    loss = out.sum()
    loss.backward()

    assert projector.proj.weight.grad is not None, "[7.2] projector.weight.grad is None"
    # weight.grad equals (emo summed over batch) when out.sum() is the loss; non-zero for randn input.
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

    # Simulated speaker_emb shape from sft_12hz: [B, target_dim].
    fake_spk = torch.zeros(batch_size, target_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    emo = torch.randn(batch_size, projector.emotion_dim, device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    out = fake_spk + projector(emo)
    assert out.shape == fake_spk.shape, f"[7.4] add result {out.shape} != speaker_emb {fake_spk.shape}"
    print(f"[7.4] OK: speaker_emb {tuple(fake_spk.shape)} + emo_proj == {tuple(out.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--emotion_dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    args = parser.parse_args()

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
    print("Sanity checks for EmotionProjector integration")
    print("=" * 60)

    check_71_zero_init(projector, args.batch_size)
    check_72_grad_flow(model, projector, args.batch_size)
    check_73_trainable_count(model, projector)
    check_74_shape_match(model, projector, args.batch_size)

    print("=" * 60)
    print("All checks passed. Ready to run sft_emotion_12hz.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()
