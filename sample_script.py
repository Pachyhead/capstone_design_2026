import argparse
import os
from pathlib import Path
import random

import numpy as np
from scipy.io import wavfile
from collections import Counter, defaultdict
import torch

from qwen_tts.inference.emotion_loader import load_emotion_projector
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.lora import set_lora_enabled

import json

def _load_emotion_npy(path: str) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return torch.from_numpy(arr.astype(np.float32))

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

def build_pairs(items, all_items, seed: int, exclude_emotions, max_per_speaker: int = 2):
    """Pick (ref, test) pairs where ref is a same-speaker neutral with different text.

    - test must have an emo_label NOT in exclude_emotions (default: skip neutral/other/unknown)
    - ref is sampled from test["neutral_pool"] with id != test.id and text != test.text
    - At most `max_per_speaker` test samples per speaker (avoids one chatty speaker dominating)
    - Returns ALL eligible pairs (no count cap).
    """
    id_to_item = {it["id"]: it for it in all_items}
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--projector_dir", required=True,
                        help="checkpoint-epoch-{N} directory produced by sft_emotion_12hz.py")
    parser.add_argument("--language", default="Korean")

    parser.add_argument("--vaild_file", default="Korean")
    parser.add_argument("--out_dir", default=None, help="Output directory (grid mode)")
    parser.add_argument("--data_root", required=True)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", default="sdpa")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_speaker", type=int, default=2,
                            help="Cap samples per speaker so one chatty speaker doesn't dominate")
    parser.add_argument("--exclude_emotions", nargs="+",
                            default=["other", "unk"],
                            help="Skip test samples with these labels (default focuses on emotional content)")
    parser.add_argument("--sample_per_ammout", default=10,
                            help="Skip test samples with these labels (default focuses on emotional content)")

    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    gen_kwargs = dict(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
    )

    try :
        data_root = Path(args.data_root)
        out_path = Path(args.out_dir)
        out_path.mkdir(exist_ok=True)
    except  Exception as e:
        print(e)
        exit(0)

    print(f"Loading {args.init_model_path} ...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map=args.device
    )

    print(f"Loading emotion projector from {args.projector_dir} ...")
    load_emotion_projector(qwen3tts.model, args.projector_dir, device=torch.device(args.device), dtype=dtype)


    data_list = []
    with open(args.vaild_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)

    dict_per_emotion = dict()
    for sample in data_list:
        if dict_per_emotion.setdefault(sample["emo_label"], None) != None:
            dict_per_emotion[sample["emo_label"]].append(sample)
        else :
            dict_per_emotion[sample["emo_label"]] = []
            dict_per_emotion[sample["emo_label"]].append(sample)
    print('\n\n\n\n', "-"*20)
    print("Test audio checking")
    for key, value in dict_per_emotion.items():
        print(f"{key} :", f"{len(build_pairs(value, data_list,seed=args.seed, exclude_emotions=tuple(args.exclude_emotions), max_per_speaker=args.max_per_speaker))}")
        dict_per_emotion[key] = build_pairs(value, data_list,seed=args.seed, exclude_emotions=tuple(args.exclude_emotions), max_per_speaker=args.max_per_speaker)
    print("-"*20, '\n\n\n\n')

    for key, value in dict_per_emotion.items():
        if len(value) <= args.sample_per_ammout:
            continue
        random.seed(args.seed)
        sampled_value = random.sample(value, k=args.sample_per_ammout)
        emotion_path = out_path / key
        emotion_path.mkdir(exist_ok=True)
        log_dir_emotion = dict()
        for i, (ref, test) in enumerate(sampled_value):
            sample_path = emotion_path / f"{i}"
            sample_path.mkdir(exist_ok=True)
            print(f"Emotion : {key}, generate at {str(sample_path)}")
            ref_path = str(data_root / ref["wav"])
            test_path = str(data_root / test["wav"])
            gt_emo_vec = _load_emotion_npy(str(data_root / test["emo_vec"]))

            fs_s, wav_source = wavfile.read(ref_path)
            fs_t, wav_target = wavfile.read(test_path)
            print(f"Generate by base model, generate at {str(sample_path)}")
            set_lora_enabled(qwen3tts.model, False)
            torch.manual_seed(args.seed + i)
            wav_b, sr_b = synthesize(qwen3tts, test["text"], ref_path, ref["text"], args.language,
                                    emotion_vec=None, gen_kwargs=gen_kwargs)
            print(f"Generate by Emotion model, generate at {str(sample_path)}")
            set_lora_enabled(qwen3tts.model, True)
            torch.manual_seed(args.seed + i)
            wav_e, sr_e = synthesize(qwen3tts, test["text"], ref_path, ref["text"], args.language,
                                    emotion_vec=gt_emo_vec, gen_kwargs=gen_kwargs)

            wavfile.write(str(sample_path / "source.wav"), fs_s, wav_source)
            wavfile.write(str(sample_path / "reference.wav"), fs_t, wav_target)
            wavfile.write(str(sample_path / "generate_base.wav"), sr_b, wav_b)
            wavfile.write(str(sample_path / "generate_emotion.wav"), sr_e, wav_e)


            log_dir_emotion[i] = {
                "ref_text" : ref["text"],
                "target_text" : test["text"]
            }
        fp = open(str(emotion_path / "log.json"), "w", encoding="utf-8")
        json.dump(log_dir_emotion, fp, indent=4, ensure_ascii=False)

if __name__ == "__main__": 
    main()