import json
from pathlib import Path
from collections import defaultdict, Counter
import random
 
import numpy as np
import torch
from tqdm import tqdm 

EXCLUDE_LABEL = {"unk"}
PER_EMOTION = 7000

def filter_excluded(metadata):
    """
    제외 라벨에 속하는 엔트리 제거
    """
    before = len(metadata)
    filtered = [m for m in metadata if m["emo_label"] not in EXCLUDE_LABEL]
    print(f"제외 라벨 {EXCLUDE_LABEL}: {before - len(filtered):,}개 제외 -> {len(filtered):,}개")
    return filtered

def balance_by_emotion(metadata, per_emotion, seed):
    """
    감정별로 최대 per_emotion개씩 추출
 
    감정별 데이터 수가 per_emotion보다
    - 많으면: 무작위로 per_emotion개 샘플링
    - 적으면: 있는 만큼 모두 사용
    """
    rng = random.Random(seed)
 
    # 감정별 그룹화
    buckets = defaultdict(list)
    for m in metadata:
        buckets[m["emo_label"]].append(m)
 
    # 감정별로 cap 적용
    balanced = []
    for items in buckets.values():
        if len(items) > per_emotion:
            items = rng.sample(items, per_emotion)
        balanced.extend(items)
 
    rng.shuffle(balanced)
    print(f"감정당 cap={per_emotion}: {len(metadata):,} -> {len(balanced):,}개")
    print(f"감정별 분포: {dict(Counter(m['emo_label'] for m in balanced))}")
    return balanced

def load_embeddings(metadata, emb_dir):
    """
    metadata 순서대로 .npy 파일 로드, (N, D) 배열 반환
    """
    sample = np.load(emb_dir / f"{metadata[0]['id']}.npy")
    print(f"샘플 shape: {sample.shape}, dtype: {sample.dtype}")
    D = sample.shape[-1]
 
    embeddings = np.empty((len(metadata), D), dtype=np.float32)
    ids, speakers, emotions = [], [], []
 
    for i, entry in enumerate(tqdm(metadata, desc="Loading from HDD")):
        npy_path = emb_dir / f"{entry['id']}.npy"
        embeddings[i] = np.load(npy_path).astype(np.float32).squeeze()
        ids.append(entry["id"])
        speakers.append(entry["source_file"])
        emotions.append(entry["emo_label"])
 
    return embeddings, ids, speakers, emotions

def prepare(metadata_path: str, emb_dir: str, out_path: str, seed: int = 42):
    """
    metadata.jsonl 읽고 각 id에 해당하는 .npy 로드해서 단일 .pt 저장
    """
    metadata_path = Path(metadata_path)
    emb_dir = Path(emb_dir)
 
    # 메타데이터 로드 + 필터링 + 균형 추출
    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    print(f"총 엔트리: {len(metadata):,}개")
 
    if len(metadata) == 0:
        raise ValueError(f"메타데이터가 비어있음: {metadata_path}")
 
    metadata = filter_excluded(metadata)
 
    if PER_EMOTION is not None:
        metadata = balance_by_emotion(metadata, PER_EMOTION, seed)
 
    # 임베딩 로드
    embeddings, ids, speakers, emotions = load_embeddings(metadata, emb_dir)
    print(f"고유 화자: {len(set(speakers)):,}개")
 
    # 정규화 통계
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0).clip(min=1e-6)
    print(f"차원별 std 범위:  [{embeddings.std(0).min():.4f}, {embeddings.std(0).max():.4f}]")
    print(f"차원별 mean 범위: [{embeddings.mean(0).min():.4f}, {embeddings.mean(0).max():.4f}]")
 
    # 저장
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": torch.from_numpy(embeddings),
        "ids": ids,
        "speakers": speakers,
        "emotions": emotions,
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }, out_path)
 
    print(f"저장 완료: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")