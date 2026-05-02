# coding=utf-8
# Project-local extension of finetuning/sft_12hz.py:
# Trains ONLY the EmotionProjector (1024 -> 2048, zero-init) on top of the
# Qwen3-TTS-12Hz-1.7B-Base backbone. Backbone (talker, code_predictor, speaker_encoder)
# stays frozen. Emotion is added element-wise to the speaker embedding that occupies
# sequence position 6 -- sequence length, position ids, and every other tensor shape
# are preserved.
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
from qwen_tts.finetuning.dataset_emotion import EmotionTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _attach_emotion_projector(model, emotion_dim: int, dtype: torch.dtype, device: torch.device) -> EmotionProjector:
    """Attach a fresh, zero-init EmotionProjector if not already present.

    We attach in-place rather than relying on config-driven instantiation in
    Qwen3TTSForConditionalGeneration.__init__ because the upstream config.json
    lacks `use_emotion_projector` -- a base checkpoint loaded via from_pretrained
    therefore arrives without the module.
    """
    if getattr(model, "emotion_projector", None) is None:
        target_dim = model.config.speaker_encoder_config.enc_dim
        proj = EmotionProjector(emotion_dim=emotion_dim, target_dim=target_dim)
        proj = proj.to(device=device, dtype=dtype)
        model.emotion_projector = proj
        model.config.use_emotion_projector = True
        model.config.emotion_dim = emotion_dim
    return model.emotion_projector


def _freeze_all_but_emotion_projector(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.emotion_projector.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_total = sum(p.numel() for p in model.emotion_projector.parameters())
    assert trainable == proj_total, (
        f"Freeze failed: trainable={trainable} != emotion_projector={proj_total}"
    )
    return trainable


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output_emotion")
    parser.add_argument("--manifest_path", type=str, required=True,
                        help="Train manifest jsonl produced by build_manifest.py + add_audio_codes.py")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory the manifest's relative wav/emo_vec paths resolve against")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--emotion_dim", type=int, default=1024,
                        help="Verified Emotion2Vec dim (run scripts/verify_emotion_dim.py first)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log scalar metrics to tensorboard every N optimizer steps")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable tensorboard logging")
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention implementation. sdpa = PyTorch built-in (no extra install). "
                             "flash_attention_2 requires `pip install flash-attn` and CUDA-matched build.")
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
        accelerator.init_trackers("emotion_projector", config=vars(args))

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    config = AutoConfig.from_pretrained(args.init_model_path)

    # Attach zero-init EmotionProjector to the underlying model (not the wrapper)
    base_model = qwen3tts.model
    _attach_emotion_projector(
        base_model,
        emotion_dim=args.emotion_dim,
        dtype=torch.bfloat16,
        device=base_model.device if hasattr(base_model, "device") else accelerator.device,
    )
    n_trainable = _freeze_all_but_emotion_projector(base_model)
    accelerator.print(f"Trainable params: {n_trainable:,} (EmotionProjector only)")

    # Data
    dataset = EmotionTTSDataset(
        manifest_path=args.manifest_path,
        data_root=args.data_root,
        processor=qwen3tts.processor,
        config=config,
        expected_emotion_dim=args.emotion_dim,
    )
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Optimizer over emotion_projector only
    optimizer = AdamW(base_model.emotion_projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
                emotion_vec = batch["emotion_vec"].to(model.device).to(model.dtype)  # [B, 1024]

                # Speaker encoder is frozen -> detach the speaker embedding stream so we
                # never push gradients into the (frozen) speaker encoder. Emotion injection
                # is bypassed via projector(emotion_vec); since projector is zero-init at
                # step 0, this branch outputs zero and the assignment below is bit-exact
                # to the baseline behavior. Gradients flow through emotion_projector ONLY.
                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                emo_proj = model.emotion_projector(emotion_vec)  # [B, 2048], zero at start
                speaker_embedding = speaker_embedding + emo_proj  # element-wise add

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding  # one-token slot

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

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.emotion_projector.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if not args.no_tensorboard and global_step % args.log_every == 0:
                    proj_w = accelerator.unwrap_model(model).emotion_projector.proj.weight
                    accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/talker_loss": outputs.loss.item(),
                            "train/subtalker_loss": sub_talker_loss.item(),
                            "train/emo_proj_w_norm": proj_w.detach().float().norm().item(),
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if step % 100 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} (global {global_step}) | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            proj_state = {k: v.detach().to("cpu") for k, v in unwrapped_model.emotion_projector.state_dict().items()}
            save_file(proj_state, os.path.join(output_dir, "emotion_projector.safetensors"))

            with open(os.path.join(output_dir, "emotion_projector_config.json"), "w") as f:
                json.dump(
                    {
                        "emotion_dim": args.emotion_dim,
                        "target_dim": unwrapped_model.config.speaker_encoder_config.enc_dim,
                        "init_model_path": args.init_model_path,
                        "epoch": epoch,
                    },
                    f,
                    indent=2,
                )
            accelerator.print(f"Saved EmotionProjector checkpoint -> {output_dir}")

    if not args.no_tensorboard:
        accelerator.end_training()


if __name__ == "__main__":
    train()
