# coding=utf-8
# Project-local extension of finetuning/dataset.py.
#
# Differences from upstream TTSDataset:
#   - manifest-based: takes a path + data_root instead of a pre-loaded list
#   - per-utterance Emotion2Vec feature ([D]) loaded from emo_vec npy
#   - reference (ref_mel) is sampled at __getitem__ time from the same speaker's
#     neutral pool; raw 'ref_audio' field is no longer expected in the manifest
#   - self-reference avoidance: target id != ref id whenever the pool has >= 2
#     entries (single-entry pools fall back to self-reference)
#
# Manifest entry contract (produced by scripts/build_manifest.py +
# scripts/add_audio_codes.py):
#   id, speaker_id, wav, text, audio_codes, emo_vec, emo_label, emo_scores,
#   duration_sec, neutral_pool: [list of same-speaker-neutral ids]
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

NEUTRAL_LABEL = "neutral"


class EmotionTTSDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        processor,
        config: Qwen3TTSConfig,
        sample_rate: int = 24000,
        expected_emotion_dim: int = 1024,
        max_audio_sec: float = 20.0,
        lag_num: int = -1,
    ):
        self.manifest_path = manifest_path
        self.data_root = Path(data_root)
        self.processor = processor
        self.config = config
        self.sample_rate = sample_rate
        self.expected_emotion_dim = expected_emotion_dim
        self.max_audio_sec = max_audio_sec
        self.lag_num = lag_num

        self.samples: List[dict] = []
        self.id_to_sample: dict = {}
        self.speaker_to_samples: dict = defaultdict(list)
        self.speaker_to_neutral: dict = defaultdict(list)

        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)
                self.id_to_sample[item["id"]] = item
                spk = item["speaker_id"]
                self.speaker_to_samples[spk].append(item)
                if item.get("emo_label") == NEUTRAL_LABEL:
                    self.speaker_to_neutral[spk].append(item)

        # Every speaker in the manifest must carry at least one neutral sample.
        # Manifest builder is responsible for filtering; raise loudly if violated.
        missing = [spk for spk in self.speaker_to_samples if not self.speaker_to_neutral.get(spk)]
        if missing:
            raise ValueError(
                f"{len(missing)} speakers have no neutral sample (e.g. {missing[:3]}). "
                "Re-run scripts/build_manifest.py (drops such speakers by default)."
            )

    def __len__(self):
        return len(self.samples)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _resample_if_needed(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        if sr == self.sample_rate:
            return audio, sr
        return librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate).astype(np.float32), self.sample_rate

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_texts(self, text) -> torch.Tensor:
        out = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = out["input_ids"]
        return input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 24000:
            audio, sr = self._resample_if_needed(audio, sr)
        return mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)

    def _load_emotion_vec(self, path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.ndim != 1 or arr.shape[0] != self.expected_emotion_dim:
            raise ValueError(
                f"emo_vec shape {arr.shape} from {path} incompatible with expected_emotion_dim={self.expected_emotion_dim}"
            )
        return torch.from_numpy(arr.astype(np.float32))

    def _select_reference(self, item: dict) -> dict:
        """Same-speaker neutral with self-reference avoidance.

        If the pool has >=2 entries we always return one whose id != item['id'].
        For pools of size 1 we fall back to self-reference (rare; the manifest
        builder filters most of these via min_neutral_duration / drop-no-neutral).
        """
        pool = self.speaker_to_neutral[item["speaker_id"]]
        if len(pool) > 1:
            others = [s for s in pool if s["id"] != item["id"]]
            return random.choice(others) if others else pool[0]
        return pool[0]

    def __getitem__(self, idx):
        item = self.samples[idx]

        if "audio_codes" not in item:
            raise KeyError(
                f"Item {item.get('id')} is missing 'audio_codes'. "
                "Run scripts/add_audio_codes.py over the manifest before training."
            )

        text = self._build_assistant_text(item["text"])
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        ref_item = self._select_reference(item)
        ref_wav, ref_sr = self._load_audio_to_np(str(self.data_root / ref_item["wav"]))
        ref_mel = self.extract_mels(audio=ref_wav, sr=ref_sr)

        emo_vec = self._load_emotion_vec(str(self.data_root / item["emo_vec"]))

        return {
            "id": item["id"],
            "ref_id": ref_item["id"],
            "speaker_id": item["speaker_id"],
            "emo_label": item.get("emo_label"),
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
            "emotion_vec": emo_vec,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8 + text_ids_len + codec_ids_len] = True

            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # speaker slot, replaced at training time
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8:8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3:8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # speaker slot, filled by speaker_emb (+emotion)

            codec_mask[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, :8 + text_ids_len + codec_ids_len] = True

        ref_mels = torch.cat([data["ref_mel"] for data in batch], dim=0)
        emotion_vec = torch.stack([data["emotion_vec"] for data in batch], dim=0)  # [B, expected_emotion_dim]

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "emotion_vec": emotion_vec,
            # diagnostic-only fields (not consumed by the talker forward)
            "ids": [b["id"] for b in batch],
            "ref_ids": [b["ref_id"] for b in batch],
            "speaker_ids": [b["speaker_id"] for b in batch],
            "emo_labels": [b["emo_label"] for b in batch],
        }


# Backward-compat alias (legacy name from the previous step).
TTSEmotionDataset = EmotionTTSDataset
