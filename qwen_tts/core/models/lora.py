# coding=utf-8
"""Minimal LoRA (Low-Rank Adaptation) for Qwen3-TTS talker fine-tuning.

Why inline (no peft dependency):
- Avoids version-pinning headaches with peft / accelerate / transformers
- Transparent: every wrapped Linear is exactly `base(x) + (alpha/r) * B(A(x))`
- Zero-init on B (output side) -> the wrapped layer is bit-exact w.r.t. baseline
  at training start, so the model behavior matches the un-LoRA'd baseline until
  gradients move B off zero. Mirrors the EmotionProjector zero-init invariant.
"""
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a LoRA adapter.

    Forward:  out = base(x) + scaling * dropout( B( A(x) ) )
              where scaling = alpha / r, A: in_features -> r, B: r -> out_features
    Init:
        A ~ Kaiming uniform (standard for low-rank)
        B = 0            (so initial output equals baseline)
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # Adapters (kept in fp32 for stability; cast in forward)
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Runtime toggle. When False, forward returns base(x) -- bit-exact w.r.t. the
        # un-adapted nn.Linear. Used at eval to compare LoRA-on vs LoRA-off without
        # reloading the model.
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        # Run adapter in matching dtype to avoid bf16<->fp32 mismatch
        a_w = self.lora_A.weight.to(x.dtype)
        b_w = self.lora_B.weight.to(x.dtype)
        adapter = torch.nn.functional.linear(self.dropout(x), a_w)
        adapter = torch.nn.functional.linear(adapter, b_w)
        return out + self.scaling * adapter


def apply_lora_to_talker(
    model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_suffixes: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    include_mlp: bool = False,
    include_code_predictor: bool = False,
) -> List[str]:
    """Wrap target Linear layers inside model.talker.model.layers[*].

    Sub-talker attention is optionally wrapped with include_code_predictor=True.
    Speaker encoder, embeddings, lm_head are NOT touched -- they stay frozen.

    Args:
        model: Qwen3TTSForConditionalGeneration
        r, alpha, dropout: LoRA hyperparameters
        target_suffixes: attention sub-layer names. Standard Qwen names.
        include_mlp: also wrap Talker MLP projections
        include_code_predictor: also wrap Sub-talker attention projections

    Returns:
        list of fully-qualified module paths that were replaced.
    """
    targets = [("talker.model", model.talker.model, include_mlp)]
    if include_code_predictor:
        targets.append(("talker.code_predictor.model", model.talker.code_predictor.model, False))

    replaced = []
    for prefix, decoder, wrap_mlp in targets:
        suffixes = tuple(target_suffixes)
        if wrap_mlp:
            suffixes += ("gate_proj", "up_proj", "down_proj")
        for layer_idx, layer in enumerate(decoder.layers):
            for name in suffixes:
                host_name = None
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, name):
                    host_name = "self_attn"
                elif wrap_mlp and hasattr(layer, "mlp") and hasattr(layer.mlp, name):
                    host_name = "mlp"
                if host_name is None:
                    continue
                host = getattr(layer, host_name)
                base = getattr(host, name)
                if not isinstance(base, nn.Linear):
                    continue
                adapter = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
                adapter = adapter.to(device=base.weight.device, dtype=base.weight.dtype)
                setattr(host, name, adapter)
                replaced.append(f"{prefix}.layers.{layer_idx}.{host_name}.{name}")
    return replaced


def lora_parameters(model) -> Iterable[torch.nn.Parameter]:
    """Yield LoRA-only parameters (lora_A, lora_B)."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield param


def lora_state_dict(model) -> dict:
    """Return only LoRA params as cpu state dict (for save_file)."""
    return {n: p.detach().to("cpu") for n, p in model.named_parameters()
            if "lora_A" in n or "lora_B" in n}


def load_lora_state(model, state_dict: dict) -> Tuple[List[str], List[str]]:
    """Load LoRA weights into model. Returns (loaded_keys, missing_keys)."""
    own = dict(model.named_parameters())
    loaded, missing = [], []
    for k, v in state_dict.items():
        if k in own:
            with torch.no_grad():
                own[k].copy_(v.to(own[k].device).to(own[k].dtype))
            loaded.append(k)
        else:
            missing.append(k)
    return loaded, missing


def set_lora_enabled(model, enabled: bool) -> int:
    """Toggle every LoRALinear adapter in `model` on/off. Returns the count toggled.

    Disabled LoRALinear is bit-exact to its underlying nn.Linear -- so this lets us
    run a "true baseline" pass (no LoRA contribution) on the same model instance.
    Returns 0 if the model has no LoRA modules (Stage-1 checkpoint, etc.).
    """
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled
            n += 1
    return n
