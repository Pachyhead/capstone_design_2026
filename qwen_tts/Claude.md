# Qwen3-TTS Emotion Fine-tuning 작업 기록

## 0. 목표

Qwen3-TTS-12Hz-1.7B-Base에 **utterance-level Emotion2Vec feature**를 zero-init linear projection으로 speaker embedding에 elementwise add → 학습 시작 시 baseline과 bit-exact 동일, 학습이 진행되며 emotion conditioning이 활성화되는 fine-tuning 파이프라인.

핵심 설계 원칙:
- Emotion vector(1024d) → `Linear(1024 → 2048, zero-init)` → speaker embedding과 element-wise add (sequence position 6 slot에서 합산)
- 학습 시작 시점에 baseline 모델과 **bit-exact 동일** 출력 (zero-init invariant)
- 1단계: `EmotionProjector`만 학습 (≈2.1M params, base 대비 0.12%). Backbone/speaker_encoder 전체 freeze.

---

## 1. 코드베이스 layout

```
qwen_tts/
├── Claude.md                          # 이 파일 (작업 메모)
├── README.md, __init__.py, __main__.py, cli/  # upstream copy (변경 없음)
│
├── model/                             # ★ upstream baseline (read-only, sync용)
│   ├── qwen_tts/                      # github.com/QwenLM/Qwen3-TTS clone
│   └── finetuning/                    # upstream sft_12hz.py / dataset.py / prepare_data.py
│
├── core/
│   └── models/
│       ├── __init__.py                # ✏️ EmotionProjector export 추가
│       ├── configuration_qwen3_tts.py # ✏️ Qwen3TTSConfig에 emotion_dim, use_emotion_projector
│       ├── modeling_qwen3_tts.py      # ✏️ EmotionProjector sub-module 등록
│       ├── emotion_projector.py       # ★ Linear(1024→2048, zero-init), explicit dim asserts
│       └── processing_qwen3_tts.py    # 변경 없음
├── inference/                         # 변경 없음
│
├── scripts/                           # ★ 데이터 준비 (모두 신규)
│   ├── verify_emotion_dim.py          # A2: 무작위 N개 emo_vec npy로 차원 자동 검증
│   ├── build_manifest.py              # A1: 필터링 + speaker 인덱싱 + neutral_pool + train/val split
│   ├── add_audio_codes.py             # A1.5: Qwen3TTSTokenizer로 audio_codes 추가
│   └── validate_dataset.py            # A3: 학습 직전 무결성 체크
│
└── finetuning/
    ├── README.md, prepare_data.py     # upstream 그대로 (참고용)
    ├── dataset_emotion.py             # ★ EmotionTTSDataset (manifest-based, speaker-aware reference)
    ├── sft_emotion_12hz.py            # ★ freeze + emotion projector 학습 entry point
    └── sanity_check.py                # ★ Step 7 + dataset invariants 검증
```

**규칙**: `model/` 트리는 upstream sync/비교용으로 보존. 모든 수정은 `qwen_tts/` 직속에 둠.

---

## 2. 데이터 형식 (raw → manifest)

### Raw jsonl ([src/preprocess_pipeline/](../src/preprocess_pipeline/) 산출물)
```json
{
  "id": "000000",
  "source_file": "2517G2A2_person_L",
  "wav": "chunks/000000.wav",
  "text": "엄마",
  "non_verbal": false,
  "emo_vec": "embeddings/000000.npy",
  "emo_label": "disgusted",
  "emo_scores": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "start_sec": 5.564, "end_sec": 6.084, "duration_sec": 0.52
}
```
- Speaker identity: `source_file`이 `{rec}_person_{pos}` → `parse_speaker_id()` → `{rec}_{pos}` (예: `2517G2A2_L`)
- Emotion classes (9개, 순서 고정):
  `angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown`

### Manifest jsonl (build_manifest.py 출력)
원본 + `speaker_id`, `neutral_pool`(같은 speaker의 neutral sample id 리스트). `non_verbal`/duration/empty text/missing files 모두 필터링됨. `audio_codes`는 add_audio_codes.py로 별도 추가.

---

## 3. 결정사항 (사용자 확인 완료)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 수정 코드 위치 | `qwen_tts/` 직속. `qwen_tts/model/`은 baseline. |
| 2 | 통합 방식 | Modeling 파일 자체 수정 (config + modeling에 EmotionProjector sub-module) |
| 3 | Emotion2Vec 모델 | `iic/emotion2vec_plus_large`, **utterance-level 1024d** |
| 4 | Reference 선택 | 같은 speaker의 **neutral** sample 중 무작위 (self-ref 회피) |
| 5 | Conditioning 입력 | Target utterance의 `emo_vec` (.npy) |
| 6 | non_verbal 처리 | manifest 단계에서 **완전 제외** |
| 7 | Train/Val split | **Speaker-level** (val에 약 5% speaker, default) |
| 8 | Neutral 없는 speaker | default 제외 (`--allow_neutral_fallback`로 emo_scores 최대 sample 대체 가능) |

---

## 4. 핵심 발견 (Step 1 탐색)

- LM `hidden_size = 2048`, `speaker_encoder.enc_dim = 2048` (동일 → projection 없이 직접 add 가능)
- Speaker embedding은 별도 텐서가 아닌 **sequence position 6번 token 한 칸**을 차지 ([model/finetuning/sft_12hz.py:91](model/finetuning/sft_12hz.py#L91)에서 `input_codec_embedding[:, 6, :] = speaker_embedding`)
- 16-layer RVQ codec embedding은 단순 sum (text + 16 codec layers all sum into talker input)
- M-RoPE (3D rotary), position id는 cache_position에서 자동 생성 → sequence 길이만 의존 → pos 6 슬롯 그대로 두고 add하면 position 변화 0
- **Sequence 길이/position id/모든 shape 변화 없이 emotion 신호 주입 가능**

---

## 5. End-to-end 파이프라인

### Step 1 — Emotion 차원 검증 (A2)
```bash
python -m qwen_tts.scripts.verify_emotion_dim \
    --input_jsonl raw_metadata.jsonl --data_root /data
```
출력 예: `Detected emotion dimension: 1024 (utterance-level [D])` → `EMOTION_DIM` 확정.

### Step 2 — Manifest 빌드 (A1)
```bash
python -m qwen_tts.scripts.build_manifest \
    --input_jsonl raw_metadata.jsonl \
    --data_root /data \
    --output_dir manifests \
    --min_duration 1.0 --max_duration 20.0 \
    --min_neutral_duration 2.0 \
    --val_speaker_ratio 0.05
```
출력: `manifests/manifest_train.jsonl`, `manifests/manifest_val.jsonl`, `manifests/manifest_stats.json`. **stats.json을 보고 `excluded_no_neutral` 비율을 확인**:
- < 5% → 그대로 진행
- 5~20% → `--allow_neutral_fallback` 추가해 재실행 권장
- > 20% → 데이터 정책 자체 재검토

### Step 3 — Audio codes 추가 (A1.5)
```bash
python -m qwen_tts.scripts.add_audio_codes \
    --input_jsonl manifests/manifest_train.jsonl \
    --output_jsonl manifests/manifest_train.codes.jsonl \
    --data_root /data --device cuda:0 --batch_size 32
```
val manifest도 동일하게 처리.

### Step 4 — 무결성 체크 (A3)
```bash
python -m qwen_tts.scripts.validate_dataset \
    --manifest manifests/manifest_train.codes.jsonl --data_root /data
```

### Step 5 — Sanity check (model + dataset invariants)
```bash
python -m qwen_tts.finetuning.sanity_check \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_manifest manifests/manifest_train.codes.jsonl \
    --val_manifest   manifests/manifest_val.codes.jsonl \
    --data_root /data
```
**모두 통과해야 학습 시작.**

### Step 6 — 학습
```bash
python -m qwen_tts.finetuning.sft_emotion_12hz \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --manifest_path manifests/manifest_train.codes.jsonl \
    --data_root /data \
    --batch_size 2 --lr 2e-4 --num_epochs 3 \
    --output_model_path output_emotion
```
각 epoch 종료 시 `output_emotion/checkpoint-epoch-{N}/emotion_projector.safetensors` 저장.

---

## 6. Invariants (학습 전 모두 통과 확인)

### Model-side (sanity_check `7.1~7.4`)
- **7.1 Zero-init**: `projector(any_vec) == 0` → speaker_emb 그대로 (bit-exact baseline)
- **7.2 Gradient flow**: emotion_projector에만 grad. frozen param 누출 0.
- **7.3 Trainable count**: `1024 × 2048 + 2048 = 2,099,200` 일치
- **7.4 Shape match**: `[B, 2048] + [B, 2048]` 보존

### Dataset-side (sanity_check `D1~D3`)
- **D1 No self-reference**: pool ≥ 2면 ref_id ≠ target_id 100% (avoidable self-ref < 0.5%). Pool 크기 1인 경우만 forced self-ref 허용.
- **D2 Emotion dim consistency**: 무작위 100개 npy 모두 `(EMOTION_DIM,)` (또는 frame-level이면 mean(0)) 통과
- **D3 Speaker split**: `train_speakers ∩ val_speakers == ∅`

### Training-time (post-step)
- **7.5 Output changes after training**: 1 epoch 후 `emotion_projector.proj.weight` magnitude > 0 → emotion이 출력에 영향. (수동 확인 또는 별도 스크립트로)

---

## 7. 핵심 코드 흐름 (학습 루프)

[finetuning/sft_emotion_12hz.py](finetuning/sft_emotion_12hz.py) 안에서:

```python
# Speaker encoder는 frozen + .detach() → 절대 grad 안 받음
speaker_embedding = model.speaker_encoder(ref_mels...).detach()  # [B, 2048]

# Emotion projector는 학습 가능. zero-init이라 시작 시 0
emo_proj = model.emotion_projector(emotion_vec)                  # [B, 2048]
speaker_embedding = speaker_embedding + emo_proj                 # element-wise add

# 그 외는 upstream sft_12hz.py와 동일 — pos 6 slot에 assign
input_codec_embedding[:, 6, :] = speaker_embedding
```
→ Sequence 길이/position id/RoPE/모든 shape 변화 0. Gradient는 EmotionProjector를 통해서만 흘러감.

---

## 8. 데이터 형식 (Dataset 입력)

`EmotionTTSDataset(manifest_path, data_root, processor, config, expected_emotion_dim=1024, ...)` 가 기대하는 manifest entry:
```json
{
  "id": "000000",
  "speaker_id": "2517G2A2_L",
  "wav": "chunks/000000.wav",
  "text": "...",
  "audio_codes": [[...], ...],    // [T, 16] from add_audio_codes.py
  "emo_vec": "embeddings/000000.npy",
  "emo_label": "disgusted",
  "emo_scores": [...],
  "duration_sec": 0.52,
  "neutral_pool": ["000123", "000456", ...]
}
```

`__getitem__` 출력 (collate에 넘김):
- `text_ids`, `audio_codes`, `ref_mel`, `emotion_vec`(`[1024]`)
- 진단용: `id`, `ref_id`, `speaker_id`, `emo_label`

---

## 9. 향후 확장 (TODO만)

- [ ] LoRA on backbone (projector saturate 후 2단계)
- [ ] Speaker–emotion disentanglement loss
- [ ] Frame-level emotion 전환 (`granularity="frame"` 추출, dataset의 `mean(0)` 부분 제거)
- [ ] Inference 파이프라인에 emotion 공급 (`inference/qwen3_tts_model.py`의 `generate_voice_clone`/`generate_speaker_prompt`에 `emotion_vec` 인자 추가)
- [ ] Validation 시 WER, speaker similarity, **emotion similarity** (emotion2vec cosine sim) 메트릭 추가
- [ ] StyleBench 같은 외부 evaluation suite와의 연결

---

## 10. 디버깅 시 참고할 위치

| 증상 | 확인 위치 |
|---|---|
| Projector가 학습이 안됨 | `emotion_vec`이 GPU에 올라갔는지, dtype 일치(`bfloat16`) — [sft_emotion_12hz.py:113](finetuning/sft_emotion_12hz.py#L113) |
| Backbone에 grad 흐름 | `speaker_embedding`에 `.detach()` 빠졌는지 — [sft_emotion_12hz.py:118](finetuning/sft_emotion_12hz.py#L118) |
| Output shape mismatch | sanity_check 7.4 / `enc_dim` vs `target_dim` |
| Baseline 출력이 달라짐 | sanity_check 7.1. `projector.proj.weight` 직접 확인 |
| `emo_vec` 차원 에러 | [dataset_emotion.py:_load_emotion_vec](finetuning/dataset_emotion.py) — frame-level이면 `mean(0)` 거치는지 / `expected_emotion_dim` 일치 |
| Speaker leak | sanity_check D3 |
| 학습 중 self-ref 너무 많음 | `manifest_stats.json`의 `train_neutral_pool_size.median == 1`이면 단일 neutral speaker 다수 → fallback 정책 또는 pool minimum 강화 |
| audio_codes 누락 에러 | manifest에 `audio_codes` 없음 → `add_audio_codes.py` 실행 필요 |

---

## 11. 작업 이력

1. **1차** (이전 세션): Step 1 코드베이스 탐색 → EmotionProjector + 학습 entry point 초안 (`TTSEmotionDataset`, `sft_emotion_12hz.py`).
2. **2차** (이번 세션): 실제 데이터 형식 확정 + 데이터 준비 파이프라인 구축
   - emotion_dim 동적 처리 (explicit assert)
   - `EmotionTTSDataset` manifest-based로 재작성, speaker-aware reference selection (random neutral, self-ref 회피)
   - `scripts/` 디렉토리에 데이터 준비 4종 추가 (verify / build / add_codes / validate)
   - sanity_check에 dataset-side invariants D1/D2/D3 추가
   - sft_emotion_12hz.py CLI를 `--manifest_path` + `--data_root`로 변경
