# coding=utf-8
"""Helpers to attach a trained EmotionProjector checkpoint to a loaded Qwen3TTS model."""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from qwen_tts.core.models.emotion_projector import EmotionProjector


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

    return projector
