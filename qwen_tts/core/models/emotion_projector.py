# coding=utf-8
# Project-local module added on top of the upstream Qwen3-TTS baseline.
# Not part of the original Qwen-TTS distribution.
import torch
import torch.nn as nn


class EmotionProjector(nn.Module):
    """Emotion2Vec utterance-level feature -> speaker embedding (LM hidden) dim.

    Zero-init: at training start, projector(emotion_vec) == 0 for any emotion_vec,
    so speaker_emb + projector(emotion_vec) == speaker_emb (bit-exact w.r.t. baseline).
    """

    def __init__(self, emotion_dim: int, target_dim: int):
        super().__init__()
        assert isinstance(emotion_dim, int) and emotion_dim > 0, f"emotion_dim must be a positive int, got {emotion_dim!r}"
        assert isinstance(target_dim, int) and target_dim > 0, f"target_dim must be a positive int, got {target_dim!r}"
        self.emotion_dim = emotion_dim
        self.target_dim = target_dim
        self.proj = nn.Linear(emotion_dim, target_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, emotion_vec: torch.Tensor) -> torch.Tensor:
        # emotion_vec: [B, emotion_dim] -> [B, target_dim]
        if emotion_vec.dim() != 2 or emotion_vec.shape[-1] != self.emotion_dim:
            raise ValueError(
                f"EmotionProjector expects [B, {self.emotion_dim}], got shape {tuple(emotion_vec.shape)}"
            )
        return self.proj(emotion_vec)
