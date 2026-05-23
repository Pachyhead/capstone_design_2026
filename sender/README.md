# Sender

발신측 FastAPI 서버. 사용자의 음성을 받아 STT/감정 인코딩/FSQ 양자화 후 gRPC로 수신자에게 전송한다.

## 실행

```powershell
cd sender
uvicorn main:app --port 8000
```

- 첫 실행 전: `pip install -e .` ([pyproject.toml](pyproject.toml)의 `where = ["sender_src"]` 설정으로 `sender_src/` 하위 모듈이 top-level import 가능해짐 — `from sender import Sender`, `from config import PROJECT_ROOT`).
- 저장 경로: [sender/storage/](storage/) — 녹음 wav 파일이 여기에 저장되고 `/storage/*`로 정적 서빙됨.
- FSQ 체크포인트: `sender_models/skip_kl_8d_8L_kl05_1e-4.pt` ([main.py:24](main.py#L24)).

## 엔드포인트

모두 `POST`. 응답은 JSON.

| 엔드포인트 | 파라미터 | 동작 | 응답 |
|---|---|---|---|
| `/set_my_id` | `value: int` (0-3) | `Sender.user_id` 갱신 | `{message, user_id}` |
| `/set_receiver_id` | `value: int` (0-3) | `Sender.peer_id` (수신자 id) 갱신 | `{message, receiver_id}` |
| `/start_recording` | — | sounddevice InputStream 시작, 마이크 버퍼링 | `{status}` |
| `/stop_recording` | — | 스트림 종료 → wav 저장 → Whisper STT + emotion2vec + FSQ 인코딩 → `Sender.temp_result`에 보관 | `{text, emotion, duration, audio_url}` |
| `/send` | `message: str` (선택, 생략 시 `temp_result.text`) | gRPC `Send`로 텍스트 + 감정 인덱스 전송 | `{message, text}` |
| `/send_ref` | `duration: int = 5` | (현재) 내부적으로 새 녹음을 `duration`초 진행 후 gRPC `SendVoice`로 wav 파일 전송 | `{message, file}` |
| `/get_emotion_label` | — | 마지막 인코딩 결과의 감정 라벨과 score 반환 | `{emotion_label, emotion_score}` |

`emotion` 값은 [EmotionLabel](sender_src/tone_core/types.py) enum의 lowercase 이름: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `other`, `sad`, `surprised`, `unk`.

## 흐름

### A. 텍스트 메시지 전송 (ChatDetail 화면)

```
프론트                     sender API                  내부                            gRPC
─────                     ──────────                  ────                            ────
[프로필 선택]
                          POST /set_my_id ─────────→  Sender.user_id = N
[채팅방 진입]
                          POST /set_receiver_id ───→  Sender.peer_id = N
[녹음 버튼 누름]
                          POST /start_recording ───→  AudioRecorder.start_recording()
                                                      InputStream 시작
[녹음 버튼 놓음]
                          POST /stop_recording ────→  stop_recording(encording=True)
                                                      └─ wav 저장
                                                      └─ Whisper STT
                                                      └─ emotion2vec 추출
                                                      └─ FSQ 양자화
                                                      └─ temp_result에 보관
                          ←── {text, emotion, duration, audio_url}
[draft 미리보기 확인 후 전송]
                          POST /send ──────────────→  Sender.send(message)         ──→ Send(user, peer, text,
                                                                                          emo_label, emo_indices)
                          ←── {message, text}
```

### B. 보이스 레퍼런스 등록 (Onboarding 화면)

```
프론트                     sender API                  내부                            gRPC
─────                     ──────────                  ────                            ────
[녹음 시작 버튼]
                          POST /start_recording ───→  InputStream 시작
[정지 버튼]
                          POST /send_ref?duration=N ─→ Sender.send_voice(N)
                                                       └─ (현재) 새 InputStream 시작
                                                       └─ sleep(N)
                                                       └─ stop_recording(encording=False)
                                                       └─ wav 저장                  ──→ SendVoice(user, filepath)
                          ←── {message, file}
```

> **기대 시맨틱**: `/send_ref`는 `/start_recording`이 캡처한 버퍼를 그대로 마감하고 그 파일을 `SendVoice`로 보내는 게 자연스럽다. 현재 `Sender.send_voice()`는 자체적으로 새 녹음을 시작하므로 위 흐름에서 사용자의 실제 발화는 버려진다 — 백엔드 정리 대상.

## UI 연결

프론트 [tone/src/lib/api.ts](../tone/src/lib/api.ts)의 `api` 객체가 위 엔드포인트들을 모두 메서드로 노출한다. 각 메서드를 호출하는 화면:

| 엔드포인트 | 프론트 호출 위치 |
|---|---|
| `/set_my_id` | [Profiles.tsx](../tone/src/screens/Profiles.tsx) — `handlePick` (프로필 선택 시) |
| `/set_receiver_id` | [ChatDetail.tsx](../tone/src/screens/ChatDetail.tsx) — `useEffect` (채팅방 진입 시) |
| `/start_recording` | [ChatDetail.tsx](../tone/src/screens/ChatDetail.tsx) — `handlePressStart` / [Onboarding.tsx](../tone/src/screens/Onboarding.tsx) — `handleStart` |
| `/stop_recording` | [ChatDetail.tsx](../tone/src/screens/ChatDetail.tsx) — `handlePressEnd` (응답의 `audio_url`로 draft 미리듣기 생성) |
| `/send` | [ChatDetail.tsx](../tone/src/screens/ChatDetail.tsx) — `handleConfirmSend` |
| `/send_ref` | [Onboarding.tsx](../tone/src/screens/Onboarding.tsx) — `Recording.handleStop` |
| `/get_emotion_label` | (미연결) `api.getEmotionLabel` 클라이언트만 존재. score를 UI에 노출할 결정이 서면 연결. |

## 상태 관리

- `app.state.sender: Sender` — 라이프스팬 동안 단일 인스턴스, gRPC 채널 보유
- `app.state.sender_lock: Lock` — 녹음·전송이 동시에 일어나지 않도록 직렬화
- `app.state.last_audio_file: str | None` — `/stop_recording`이 저장한 마지막 wav 경로

## CORS

[main.py:35-39](main.py#L35-L39)에서 `http://localhost:3000`, `http://127.0.0.1:3000`만 허용. 프론트(Vite 5173)를 같은 호스트에서 띄울 거면 same-origin이라 무관하지만, 다른 포트에서 dev server를 띄울 거면 `allow_origins`에 추가 필요.
