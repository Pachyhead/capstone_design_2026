# Qwen3-TTS Emotion Fine-tuning 작업 기록

## 0. 목표

Qwen3-TTS-12Hz-1.7B-Base 모델에 **utterance-level Emotion2Vec feature**를 추가 conditioning으로 주입하는 fine-tuning 파이프라인 구축.

핵심 설계 원칙:
- Emotion vector(1024d) → zero-init linear projection으로 LM hidden_dim(2048)에 맞춤
- Speaker embedding과 **element-wise add**하여 통합 (sequence 구조/길이/position id 보존)
- 학습 시작 시점에 baseline 모델과 **bit-exact 동일** 출력 (zero-init invariant)
- 1단계: `EmotionProjector`만 학습. Backbone/speaker_encoder 전체 freeze.

---

## 1. 코드베이스 layout

```
qwen_tts/
├── Claude.md                          # 이 파일
├── model/                             # ★ upstream baseline (read-only, sync용)
│   ├── qwen_tts/                      # https://github.com/QwenLM/Qwen3-TTS clone
│   └── finetuning/                    # upstream sft_12hz.py / dataset.py / prepare_data.py
│
├── __init__.py, __main__.py, cli/     # upstream에서 복사 (변경 없음)
├── core/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py                # ✏️ EmotionProjector export 추가
│   │   ├── configuration_qwen3_tts.py # ✏️ Qwen3TTSConfig에 emotion_dim, use_emotion_projector 추가
│   │   ├── modeling_qwen3_tts.py      # ✏️ EmotionProjector import + Qwen3TTSForConditionalGeneration에 sub-module 등록
│   │   ├── processing_qwen3_tts.py    # 변경 없음
│   │   └── emotion_projector.py       # ★ 신규: Linear(1024→2048, zero-init)
│   └── tokenizer_12hz/, tokenizer_25hz/  # 변경 없음
├── inference/
│   ├── qwen3_tts_model.py             # 변경 없음
│   └── qwen3_tts_tokenizer.py         # 변경 없음
└── finetuning/
    ├── README.md, prepare_data.py     # upstream 그대로
    ├── dataset_emotion.py             # ★ 신규: upstream dataset.py + emo_vec 로드/collate
    ├── sft_emotion_12hz.py            # ★ 신규: upstream sft_12hz.py + freeze + projector 학습
    └── sanity_check.py                # ★ 신규: invariant 7.1~7.4 검증
```

**규칙**: `model/` 트리는 upstream 비교/롤백용으로 보존. 모든 수정은 `qwen_tts/` 직속에 둠.

---

## 2. Step 1 — 코드베이스 탐색 핵심 결과

### Speaker encoder
- 클래스: `Qwen3TTSSpeakerEncoder` (ECAPA-TDNN 기반), [model/qwen_tts/core/models/modeling_qwen3_tts.py:311](model/qwen_tts/core/models/modeling_qwen3_tts.py#L311)
- 입력: 24kHz mel-spectrogram `[B, T_mel, 128]`
- **출력 shape: `[B, 2048]`** (`enc_dim`)
- LM `hidden_size`(2048)와 **동일** → 별도 projection 없음

### Speaker embedding이 LM 입력 sequence에 들어가는 위치
**중요 발견**: 사용자의 초기 가정과 달리, speaker embedding은 별도 텐서가 아니라 **sequence의 한 token slot(position 6)을 차지**한다.

학습 시 ([model/finetuning/sft_12hz.py:91](model/finetuning/sft_12hz.py#L91)):
```python
speaker_embedding = model.speaker_encoder(ref_mels...).detach()  # [B, 2048]
input_codec_embedding[:, 6, :] = speaker_embedding   # pos 6 slot에 assign
```

추론 시 ([model/qwen_tts/core/models/modeling_qwen3_tts.py:2169-2172](model/qwen_tts/core/models/modeling_qwen3_tts.py#L2169-L2172)):
```python
codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                   speaker_embed.view(1, 1, -1),  # [1,1,2048] 토큰 한 칸
                                   codec_input_emebdding_1], dim=1)
```

**Collate에서 pos 6은 dummy id `0`으로 채워두고 `codec_embedding_mask[i, 6] = False`로 0 처리** ([model/finetuning/dataset.py:185, 201](model/finetuning/dataset.py#L185)) → speaker_embedding으로 덮어씀.

### LM 차원
| 파라미터 | 값 |
|---|---|
| `talker_config.hidden_size` | **2048** |
| `talker_config.num_hidden_layers` | 28 |
| `talker_config.num_code_groups` (RVQ) | 16 |
| `talker_config.text_hidden_size` | 2048 |
| `speaker_encoder_config.enc_dim` | 2048 |
| `code_predictor_config.hidden_size` | 1024 (sub-talker만 작음) |

### 16-layer RVQ codec embedding aggregation
- Layer 0: `talker.model.codec_embedding` (`nn.Embedding(3072, 2048)`)
- Layer 1..15: `talker.code_predictor.codec_embedding[i-1]` (`ModuleList`, 15개)
- 16개 RVQ를 **단순 sum** → text embedding과도 sum → talker 입력

### Position encoding
**Multimodal RoPE (M-RoPE, 3D rotary)**. Position id는 cache_position에서 자동 생성 → sequence **길이만** 의존. pos 6 슬롯 그대로 두고 그 위에 emotion add → position 변화 0.

---

## 3. 결정사항 (사용자 확인 완료)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 수정 코드 위치 | `qwen_tts/` 직속. `qwen_tts/model/`은 baseline. |
| 2 | 통합 방식 | **Modeling 파일 자체 수정** (configuration + modeling에 EmotionProjector sub-module 등록) |
| 3 | Emotion2Vec 모델 | `iic/emotion2vec_plus_large`, **utterance-level 출력 1024d** ([src/preprocess_pipeline/emotion.py](../src/preprocess_pipeline/emotion.py) 참고) |

**원래 task spec과 차이**:
- emotion_dim: 768 → **1024**
- 추출 스크립트: 새로 작성 X (기존 [src/preprocess_pipeline/](../src/preprocess_pipeline/)가 이미 utterance-level 1024d npy 저장)
- 저장 형식: utterance-level `(1024,)` (frame-level 변환은 향후)

---

## 4. 구현 결과 (Phase별)

### Phase A — upstream code 복사
[github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)를 `qwen_tts/model/`에 clone 후, 핵심 트리(`qwen_tts/`, `finetuning/`)를 `qwen_tts/` 직속으로 복사. import 경로(`from qwen_tts.core.models...`)가 자동으로 우리 코드를 가리킴.

### Phase B — EmotionProjector 통합

**[qwen_tts/core/models/emotion_projector.py](core/models/emotion_projector.py)** (신규):
```python
class EmotionProjector(nn.Module):
    def __init__(self, emotion_dim=1024, target_dim=2048):
        super().__init__()
        self.proj = nn.Linear(emotion_dim, target_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    def forward(self, emotion_vec):  # [B, 1024] -> [B, 2048]
        return self.proj(emotion_vec)
```

**[configuration_qwen3_tts.py](core/models/configuration_qwen3_tts.py)** 수정:
- `Qwen3TTSConfig.__init__`에 `emotion_dim=1024`, `use_emotion_projector=False` 추가
- 기본값 False라 baseline checkpoint 로드 시 영향 없음

**[modeling_qwen3_tts.py](core/models/modeling_qwen3_tts.py)** 수정:
- `from .emotion_projector import EmotionProjector` import 추가
- `Qwen3TTSForConditionalGeneration.__init__`에서 `config.use_emotion_projector=True`일 때만 `self.emotion_projector` 생성
- **forward signature는 건드리지 않음** — emotion add는 학습 script에서 처리

### Phase C — 학습 파이프라인

**[dataset_emotion.py](finetuning/dataset_emotion.py)** (신규): upstream `TTSDataset` 미러.
- 기대 jsonl 필드: `{audio, text, audio_codes, ref_audio, emo_vec}` (+ optional `language`)
- `__getitem__`: `emo_vec` npy 로드. `[T,1024]` (frame) → `mean(0)` → `[1024]`로 collapse도 지원.
- `collate_fn`: 기존 출력 + `emotion_vec: [B, 1024]` stack

**[sft_emotion_12hz.py](finetuning/sft_emotion_12hz.py)** (신규):
- `_attach_emotion_projector(model, ...)`: 모델에 zero-init projector 부착, `device`/`dtype` 정렬
- `_freeze_all_but_emotion_projector(model)`: backbone 전체 freeze. assertion으로 trainable param 수 검증.
- 학습 루프 핵심:
  ```python
  speaker_embedding = model.speaker_encoder(ref_mels...).detach()  # 기존
  emo_proj = model.emotion_projector(emotion_vec)                  # zero-init: 시작 시 0
  speaker_embedding = speaker_embedding + emo_proj                 # element-wise add
  input_codec_embedding[:, 6, :] = speaker_embedding               # pos 6 slot
  ```
- Optimizer: `AdamW(model.emotion_projector.parameters(), lr=2e-4, wd=0.01)` — projector만
- Loss: 기존 `outputs.loss + 0.3 * sub_talker_loss` 그대로
- 저장: epoch마다 `emotion_projector.safetensors` + `emotion_projector_config.json` (별도 파일, full state_dict 저장 안 함)

### Phase D — Sanity check

**[sanity_check.py](finetuning/sanity_check.py)** (신규): invariant 검증
- **7.1 Zero-init**: `projector(zeros) == 0`, `projector(rand) == 0` (zero-init 확인)
- **7.2 Gradient flow**: projector에 grad 들어옴, frozen param에 grad 누출 없음
- **7.3 Trainable count**: `1024*2048 + 2048 = 2,099,200` 일치 확인
- **7.4 Shape match**: `speaker_emb [B, 2048] + emo_proj [B, 2048]` shape 보존
- 7.5 (학습 후 변화)는 실제 optimizer step 필요해서 별도 — `sft_emotion_12hz.py --num_epochs 1` 후 projector weight가 0이 아닌지 확인

### 정적 검증
- 모든 신규/수정 파일 AST parse 통과
- 실행 검증은 사용자 GPU 환경에서 진행 예정 (`librosa`, `funasr`, `flash_attn` 등 의존성 필요)

---

## 5. 사용자 환경에서 실행할 것

### 데이터 준비
[src/preprocess_pipeline/](../src/preprocess_pipeline/)로 생성된 metadata.jsonl(예: `{id, wav, text, emo_vec, ...}`)을 학습용 jsonl로 변환:

1. `wav` → `audio` 필드로 매핑
2. `ref_audio` 필드 추가 (같은 화자의 다른 발화 또는 동일 utterance)
3. upstream `prepare_data.py`로 `audio_codes` 추가:
   ```bash
   python qwen_tts/finetuning/prepare_data.py \
       --input_jsonl <step1_output>.jsonl \
       --output_jsonl train_emotion.jsonl
   ```
4. 최종 jsonl 한 줄: `{audio, text, audio_codes, ref_audio, emo_vec, ...}`

### Sanity check
```bash
python -m qwen_tts.finetuning.sanity_check \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

### 학습
```bash
python -m qwen_tts.finetuning.sft_emotion_12hz \
    --train_jsonl train_emotion.jsonl \
    --batch_size 2 \
    --lr 2e-4 \
    --num_epochs 3 \
    --output_model_path output_emotion
```

각 epoch 종료 시 `output_emotion/checkpoint-epoch-{N}/emotion_projector.safetensors` 저장.

---

## 6. Invariant 정리 (수정 후에도 보존)

1. ✅ `emotion_vec`이 zero이거나 projector가 zero-init이면 **bit-exact baseline**
2. ✅ Sequence 길이 변화 0 (pos 6 슬롯 그대로 사용)
3. ✅ Position id / RoPE 변화 0 (cache_position 기반, 길이만 의존)
4. ✅ Speaker embedding shape `[B, 2048]` 유지
5. ✅ Frozen param에 grad 누출 없음 (sft script + sanity_check에서 검증)
6. ✅ Baseline checkpoint 로드 시 추가 부담 없음 (`use_emotion_projector=False` 기본)

---

## 7. 향후 확장 (TODO만, 현재 미구현)

- [ ] LoRA on backbone (projector saturate 후 2단계)
- [ ] Speaker–emotion disentanglement loss
- [ ] Frame-level emotion 전환 (`granularity="frame"`로 재추출, dataset의 mean(0) 부분 제거)
- [ ] Inference 파이프라인에 emotion 공급 (현재 학습 only — `inference/qwen3_tts_model.py`의 `generate_speaker_prompt` 또는 `generate_voice_clone`에 `emotion_vec` 인자 추가 필요)
- [ ] 작은 데이터셋(~수십 시간)으로 overfit 테스트 → emotion conditioning이 실제로 학습되는지 확인

---

## 8. 디버깅 시 참고할 위치

| 증상 | 확인 위치 |
|---|---|
| Projector가 학습이 안됨 | `emotion_vec`이 GPU에 올라갔는지, dtype 일치(`bfloat16`) — [sft_emotion_12hz.py:113](finetuning/sft_emotion_12hz.py#L113) |
| Backbone에 grad가 흘러감 | `speaker_embedding`에 `.detach()` 빠졌는지 — [sft_emotion_12hz.py:118](finetuning/sft_emotion_12hz.py#L118) |
| Output shape mismatch | sanity_check 7.4 실행. `enc_dim` vs `target_dim` |
| Baseline 출력이 달라짐 | sanity_check 7.1 — zero-init 깨졌을 가능성. `projector.proj.weight` 직접 확인 |
| `emo_vec` 차원 에러 | `dataset_emotion.py:_load_emotion_vec` — frame-level이면 `mean(0)` 거치는지 |
