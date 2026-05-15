# coding=utf-8
"""Stage-2 fine-tuning: LoRA on talker (attention) + EmotionProjector.

Differences from Stage-1 (sft_emotion_12hz.py):
- Adds LoRA adapters (rank-r) to talker.model.layers[*].self_attn.{q,k,v,o}_proj.
  Sub-talker (code_predictor), speaker_encoder, embeddings, lm_head all stay frozen.
- Trainable params: ~7M (LoRA, rank=16 default) + ~2M (EmotionProjector) = ~9M.
- LoRA is zero-initialized on the B side -> at step 0 the LoRA-augmented forward
  is bit-exact w.r.t. baseline. Combined with EmotionProjector zero-init, the
  whole model still starts from baseline behavior.

Usage:
    python3 -u -m qwen_tts.finetuning.sft_emotion_lora_12hz \\
        --init_model_path  Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --manifest_path    /home/cap/data/processed/model_train/manifest_train.codes.jsonl \\
        --data_root        /home/cap/data/processed \\
        --batch_size       1  --grad_accum 8 \\
        --lr               1e-4 \\
        --num_epochs       2 \\
        --lora_rank        16 \\
        --output_model_path /home/cap/data/processed/output_emotion_lora
"""
import argparse
import json
import os

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

from qwen_tts.core.models.emotion_projector import EmotionProjector
from qwen_tts.core.models.lora import (
    apply_lora_to_talker,
    lora_parameters,
    lora_state_dict,
)
from qwen_tts.finetuning.dataset_emotion import EmotionTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _attach_emotion_projector(model, emotion_dim: int, dtype, device) -> EmotionProjector:
    if getattr(model, "emotion_projector", None) is None:
        target_dim = model.config.speaker_encoder_config.enc_dim
        proj = EmotionProjector(emotion_dim=emotion_dim, target_dim=target_dim)
        proj = proj.to(device=device, dtype=dtype)
        model.emotion_projector = proj
        model.config.use_emotion_projector = True
        model.config.emotion_dim = emotion_dim
    return model.emotion_projector


def _freeze_for_lora_stage(model):
    """Freeze everything except LoRA params + EmotionProjector."""
    for p in model.parameters():
        p.requires_grad = False
    # EmotionProjector
    for p in model.emotion_projector.parameters():
        p.requires_grad = True
    # LoRA params
    for p in lora_parameters(model):
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_emo = sum(p.numel() for p in model.emotion_projector.parameters())
    n_lora = sum(p.numel() for p in lora_parameters(model))
    assert n_trainable == n_emo + n_lora, \
        f"Freeze accounting mismatch: trainable={n_trainable}, emo={n_emo}, lora={n_lora}"
    return n_trainable, n_emo, n_lora


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", default="output_emotion_lora")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LoRA + EmotionProjector LR. 1e-4 default (lower than Stage-1's 2e-4 since now we have more params)")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--emotion_dim", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=8)
    # LoRA hyperparams
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_include_mlp", action="store_true",
                        help="Also wrap MLP gate/up/down_proj. ~2x more LoRA params.")
    # logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    args = parser.parse_args()

    accel_kwargs = dict(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    if not args.no_tensorboard:
        project_config = ProjectConfiguration(
            project_dir=args.output_model_path,
            logging_dir=os.path.join(args.output_model_path, "tb_logs"),
        )
        accel_kwargs.update(log_with="tensorboard", project_config=project_config)
    accelerator = Accelerator(**accel_kwargs)
    if not args.no_tensorboard:
        accelerator.init_trackers("emotion_lora", config=vars(args))

    # ---- Load base model ----
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    config = AutoConfig.from_pretrained(args.init_model_path)
    base_model = qwen3tts.model

    # ---- Attach EmotionProjector (zero-init) ----
    _attach_emotion_projector(
        base_model,
        emotion_dim=args.emotion_dim,
        dtype=torch.bfloat16,
        device=base_model.device if hasattr(base_model, "device") else accelerator.device,
    )

    # ---- Apply LoRA to talker attention (and optionally MLP) ----
    replaced = apply_lora_to_talker(
        base_model,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        include_mlp=args.lora_include_mlp,
    )
    accelerator.print(f"LoRA applied to {len(replaced)} layers (first few: {replaced[:3]})")

    # ---- Freeze everything except LoRA + EmotionProjector ----
    n_trainable, n_emo, n_lora = _freeze_for_lora_stage(base_model)
    accelerator.print(
        f"Trainable params: {n_trainable:,} (EmotionProjector={n_emo:,}, LoRA={n_lora:,})"
    )

    # ---- Data ----
    dataset = EmotionTTSDataset(
        manifest_path=args.manifest_path,
        data_root=args.data_root,
        processor=qwen3tts.processor,
        config=config,
        expected_emotion_dim=args.emotion_dim,
    )
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    # ---- Optimizer over LoRA + EmotionProjector ----
    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer, train_dataloader = accelerator.prepare(base_model, optimizer, train_dataloader)
    model.train()

    global_step = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]
                emotion_vec = batch["emotion_vec"].to(model.device).to(model.dtype)

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                emo_proj = model.emotion_projector(emotion_vec)
                speaker_embedding = speaker_embedding + emo_proj

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )
                loss = outputs.loss + 0.3 * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if not args.no_tensorboard and global_step % args.log_every == 0:
                    proj_w = accelerator.unwrap_model(model).emotion_projector.proj.weight
                    # Aggregate LoRA-B norm as a single scalar -- starts at 0, grows during training
                    lora_b_norms = []
                    for n, p in accelerator.unwrap_model(model).named_parameters():
                        if "lora_B" in n:
                            lora_b_norms.append(p.detach().float().norm().item())
                    lora_b_mean_norm = (sum(lora_b_norms) / max(len(lora_b_norms), 1)) if lora_b_norms else 0.0
                    accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/talker_loss": outputs.loss.item(),
                            "train/subtalker_loss": sub_talker_loss.item(),
                            "train/emo_proj_w_norm": proj_w.detach().float().norm().item(),
                            "train/lora_B_mean_norm": lora_b_mean_norm,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} (global {global_step}) | Loss: {loss.item():.4f}")

        # ---- Save checkpoint per epoch ----
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)

            # 1) EmotionProjector
            proj_state = {k: v.detach().to("cpu") for k, v in unwrapped.emotion_projector.state_dict().items()}
            save_file(proj_state, os.path.join(output_dir, "emotion_projector.safetensors"))
            with open(os.path.join(output_dir, "emotion_projector_config.json"), "w") as f:
                json.dump(
                    {
                        "emotion_dim": args.emotion_dim,
                        "target_dim": unwrapped.config.speaker_encoder_config.enc_dim,
                        "init_model_path": args.init_model_path,
                        "epoch": epoch,
                    },
                    f, indent=2,
                )

            # 2) LoRA adapter (state dict + config)
            save_file(lora_state_dict(unwrapped), os.path.join(output_dir, "lora_adapter.safetensors"))
            with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
                json.dump(
                    {
                        "rank": args.lora_rank,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "include_mlp": args.lora_include_mlp,
                        "init_model_path": args.init_model_path,
                        "epoch": epoch,
                    },
                    f, indent=2,
                )
            accelerator.print(f"Saved checkpoint -> {output_dir}")

    if not args.no_tensorboard:
        accelerator.end_training()


if __name__ == "__main__":
    train()
