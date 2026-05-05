# coding=utf-8
"""Helpers to attach a trained EmotionProjector checkpoint to a loaded Qwen3TTS model.

Supports two checkpoint shapes produced by the project's training scripts:
  - Stage-1 (sft_emotion_12hz.py): emotion_projector.safetensors + emotion_projector_config.json
  - Stage-2 (sft_emotion_lora_12hz.py): the above PLUS lora_adapter.safetensors + lora_config.json
"""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from qwen_tts.core.models.emotion_projector import EmotionProjector
from qwen_tts.core.models.lora import apply_lora_to_talker, load_lora_state


def load_emotion_projector(model, checkpoint_dir: str, device: torch.device = None, dtype: torch.dtype = None):
    """Attach a trained EmotionProjector to a Qwen3TTSForConditionalGeneration model.

    Args:
        model: Qwen3TTSForConditionalGeneration (the underlying model, not the wrapper).
        checkpoint_dir: directory produced by sft_emotion_12hz.py for one epoch
                        (contains emotion_projector.safetensors and emotion_projector_config.json).
        device, dtype: where to place the projector. Default: match model.

    Returns:
        The attached EmotionProjector module (also assigned to model.emotion_projector).
    """
    ckpt_dir = Path(checkpoint_dir)
    weight_path = ckpt_dir / "emotion_projector.safetensors"
    config_path = ckpt_dir / "emotion_projector_config.json"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing {weight_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    projector = EmotionProjector(emotion_dim=cfg["emotion_dim"], target_dim=cfg["target_dim"])
    state = load_file(str(weight_path))
    projector.load_state_dict(state)
    projector = projector.to(device=device, dtype=dtype)

    model.emotion_projector = projector
    if hasattr(model, "config"):
        model.config.use_emotion_projector = True
        model.config.emotion_dim = cfg["emotion_dim"]

    # Stage-2 checkpoints additionally contain a LoRA adapter -- inject it now
    # so the talker forward includes the trained low-rank deltas.
    lora_weight = ckpt_dir / "lora_adapter.safetensors"
    lora_config = ckpt_dir / "lora_config.json"
    if lora_weight.exists() and lora_config.exists():
        with open(lora_config) as f:
            lcfg = json.load(f)
        replaced = apply_lora_to_talker(
            model,
            r=lcfg["rank"],
            alpha=lcfg["alpha"],
            dropout=0.0,                       # disable dropout at inference
            include_mlp=lcfg.get("include_mlp", False),
        )
        lora_state = load_file(str(lora_weight))
        loaded, missing = load_lora_state(model, lora_state)
        if missing:
            raise RuntimeError(
                f"LoRA load: {len(missing)} keys in checkpoint not found on model "
                f"(first few: {missing[:3]}). Did model architecture or rank change?"
            )
        # Sanity: every saved key must have landed
        if len(loaded) != len(lora_state):
            raise RuntimeError(
                f"LoRA load count mismatch: loaded={len(loaded)} state_keys={len(lora_state)}"
            )
        print(f"[load_emotion_projector] also loaded LoRA: {len(replaced)} layers wrapped, "
              f"{len(loaded)} adapter params restored")

    return projector
