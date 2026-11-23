# Tone 오프라인 데모 (mock UI)

네트워크나 서버 없이 Tone을 시연하기 위한 화면이다. 실제 Tone 화면 코드(`tone/src`)를
그대로 쓰고, 백엔드만 미리 준비한 음성과 시나리오로 바꿨다.

- **맨 위:** 자동 시연 제어 줄. 4단계 진행 표시와 시작·일시정지·정지 버튼.
- **가운데:** 실제와 같은 Tone 화면.
- **아래:** 포스터의 System Architecture 그림. 지금 단계와 그 구역이 강조된다.

## 실행

`site/index.html`을 더블클릭해서 Chrome이나 Edge로 연다. 설치·빌드·인터넷이 필요 없다.
발표용 PC로 옮길 때는 `site/` 폴더를 통째로 복사한다(`audio/` 폴더가 함께 있어야 소리가 난다).

## 자동 시연

시작 화면에서 `자동 시연 시작`을 누르면 가상 커서가 실제 화면을 순서대로 조작한다.
한 번에 약 1분 15초가 걸린다.

| 단계 | 화면에서 일어나는 일 | 들리는 소리 |
|---|---|---|
| 1. 프로필 생성 | 보내는 사람(재웅)의 프로필을 새로 만든다 | — |
| 2. 목소리 등록 | 정해진 문장을 읽어 목소리를 등록한다 | `reference.wav` |
| 3. 녹음 · 전송 | 상대(종찬)를 고르고, 마이크를 눌러 말한 뒤 전송한다 | `source.wav` |
| 4. 상대측 재생 | 종찬 프로필로 바꿔 받은 메시지를 재생한다 | `output.wav` |

3·4단계에서는 파이프라인 패널이 세부 단계를 차례로 보여 준다.

1. 분리 — 음성을 Whisper STT와 emotion2vec에 넣는다
2. 분리 결과 — 텍스트와 1024차원 감정 벡터로 나뉜다
3. 압축 — FSQ 인코더가 1024차원을 8차원으로 줄인다
4. 양자화 — 8차원 격자를 24bit 코드로 만든다
5. 전송 — 텍스트와 24bit를 한 번에 보낸다
6. 복원 — 서버가 24bit를 다시 1024차원으로 늘린다
7. 저장 — MySQL에 저장한다
8. 합성 준비 → 합성(Qwen3-TTS) → 전달(gRPC 스트리밍) → 재생

| 키 | 동작 |
|---|---|
| `Space` | 일시정지 / 계속 |
| `Esc` | 정지 |
| `P` | 파이프라인 패널 접기 / 펴기 |

시연이 끝나거나 정지한 뒤에는 화면을 직접 조작할 수 있다. 시작 화면에서
`직접 조작하기`를 고르면 프로필 3명(종찬·재웅·경택)이 있는 상태로 바로 쓴다.
`처음부터`를 누르면 메시지와 설정이 지워진다.

## 음성과 감정 코드 넣기

지금 들어 있는 음성은 임시다. 실제 파일을 받으면 아래처럼 바꾼다.

| 받을 것 | 넣는 곳 |
|---|---|
| 사용자 ref (정해진 문장을 읽은 목소리) | `<세트 폴더>/reference.wav` |
| 메시지를 보낼 때 녹음한 파일 | `<세트 폴더>/source.wav` |
| 합성 후 파일 | `<세트 폴더>/generate_emotion.wav` |
| DB에 저장된 감정 코드 | `src/scenario.ts`의 `emotionCode` |

1. 세트 폴더(예: `mock_ui_example_wav/mock_ui_example_wav/angry/`)의 세 파일을 바꾼다.
2. 브라우저용으로 변환한다. 출력되는 `source.wav`의 길이와 크기를 적어 둔다.

   ```
   endpoint/.venv/Scripts/python.exe mock_ui/tools/build_assets.py
   ```

3. `src/scenario.ts`에서 해당 세트를 고친다.
   - `text`: 합성 결과가 말하는 문장(채팅 메시지로 보인다)
   - `emotion`, `emoType`: 감정 이름과 번호
   - `sourceDuration`, `sourceBytes`: 2번에서 적어 둔 값
   - `emotionCode`: FSQ 인덱스 8개, 각각 0~7. 예: `emotionCode: [3, 7, 0, 5, 2, 6, 1, 4]`
4. `npm run build`로 `site/`를 다시 만든다.

`emotionCode`를 넣으면 패널의 8d 막대와 24bit 칸이 그 값을 그대로 그린다. 넣지 않으면
크기만 맞춘 장식으로 그려진다(1024d 막대는 항상 장식이다).

자동 시연에서 누가 누구에게 보내는지는 `src/scenario.ts`의 `AUTOPLAY`에서 바꾼다.
프로필마다 쓰는 음성 세트는 `DEMO_PROFILES`의 `setId`다.

## 고쳐서 다시 만들기

```
cd mock_ui
npm install     # 처음 한 번
npm run build   # site/ 를 다시 만든다
npm run dev     # 고치면서 볼 때 (http://localhost:3100)
```

| 파일 | 내용 |
|---|---|
| `src/scenario.ts` | 프로필, 음성 세트, 문장, 감정 코드, 자동 시연의 보내는 사람·받는 사람 |
| `src/autoplay.ts` | 자동 시연 순서(어떤 버튼을 언제 누르는지) |
| `src/mockApi.ts` | 가짜 백엔드. 각 호출이 파이프라인의 어느 단계인지와 머무는 시간(`MS`) |
| `src/PipelinePanel.tsx`, `src/mock.css`, `src/autoplay.css` | 파이프라인 패널과 시연 화면 스타일 |
| `src/AutoplayBar.tsx` | 위쪽 제어 줄과 시작 화면 |
| `src/main.tsx` | 화면 배치와 처음 상태 |
| `vite.config.ts` | `tone/src`의 어떤 모듈을 mock으로 바꿔 끼우는지 |
| `tools/build_assets.py` | 원본 음성을 브라우저용(16비트 WAV)으로 변환 |

`tone/src`의 화면을 고치면 이 데모에도 반영된다. 다만 `site/`는 빌드 결과물이라
`npm run build`를 다시 해야 바뀐다. 자동 시연은 버튼의 글자와 `aria-label`로 화면을
찾으므로, `tone/src`에서 그 글자를 바꾸면 `src/autoplay.ts`도 맞춰 고쳐야 한다.
