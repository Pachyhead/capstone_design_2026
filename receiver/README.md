# Receiver

수신측 FastAPI 서버. sender가 보낸 텍스트 + 감정 인덱스를 받아 클로닝된 음성으로 재합성하고, 사용자 요청 시 재생한다.

## 실행

```powershell
cd receiver
uvicorn main:app --port 8000
```

- 첫 실행 전: `pip install -e .` ([receiver/pyproject.toml](pyproject.toml) 설정에 따라 `receiver_src/`가 import path에 올라감 — `from receiver import Receiver`, `from config import PROJECT_ROOT`).
- 저장 경로: `PROJECT_ROOT / "storage"` ([main.py:15](main.py#L15)) — 수신된 메시지 JSON, 합성된 wav가 여기 저장.
- sender와 동일 머신에서 띄울 경우 포트 충돌 주의 (한 머신에 둘 다 띄우려면 receiver는 `--port 8001` 등으로 분리).

## 엔드포인트

모두 `POST`. 응답은 JSON.

| 엔드포인트 | 파라미터 | 동작 | 응답 |
|---|---|---|---|
| `/set_my_id` | `my_id: int` (0-3) | `Receiver.user_id` 갱신 + `get_pending_messages()` 호출 | `{message, user_id, get_pending_messages}` |
| `/set_sender_id` | `sender_id: int` (0-3) | `Receiver.peer_id` (발신자 id) 갱신 + `get_pending_messages()` 호출 | `{message, sender_id}` |
| `/play_voice` | `message_id: int` | 해당 message_id의 합성된 음성 재생 (서버 측 스피커 출력) | `"successfully play voice"` |
| `/get_message` | `message_id: int` | 메시지 메타 반환 — **현재는 `message_id` 무시하고 `storage/000001.json` 하드코딩 읽음** | `{message_id, sender_id, message, emo_type, send_time}` |

## 흐름

### A. 로그인 (Profiles 화면에서 프로필 선택)

```
프론트                     receiver API                내부
─────                     ────────────                ────
[프로필 클릭]
                          POST /set_my_id ─────────→  Receiver.user_id = N
                                                      get_pending_messages()
                                                      (pending json 파일 스캔)
                          ←── {user_id, get_pending_messages}
                          POST /get_message ───────→  storage/000001.json 읽기
                          ←── {message_id, sender_id, message, emo_type, send_time}
```

> **현재 한계**: `/get_message`가 `message_id`를 무시한다. 백엔드가 `message_id` 기준으로 정리되면 프론트 호출도 의미 있는 ID로 바꿔야 함.

### B. 채팅방 진입

```
프론트                     receiver API                내부
─────                     ────────────                ────
[채팅방 클릭]
                          POST /set_sender_id ─────→  Receiver.peer_id = N
                                                      get_pending_messages()
                          ←── {sender_id}
```

### C. 음성 재생

```
프론트                     receiver API                내부
─────                     ────────────                ────
[메시지의 재생 버튼 클릭]
                          POST /play_voice?       ──→ Receiver.play_voice(id)
                          message_id=N                (서버 머신의 스피커로 출력)
                          ←── "successfully play voice"
```

> 음성 재생이 **수신측 서버의 스피커**에서 일어남에 주의. 브라우저 audio 재생이 아님.

## UI 연결

프론트 [tone/src/lib/api.ts](../tone/src/lib/api.ts)의 `api` 객체가 receiver 엔드포인트들도 메서드로 노출한다. sender와 동일 base URL을 공유하므로 단방향 통신(머신 분리) 가정 하에 동작.

| 엔드포인트 | 프론트 호출 위치 |
|---|---|
| `/set_my_id` | [Profiles.tsx](../tone/src/screens/Profiles.tsx) — `handlePick`. 프론트가 `value`/`my_id` 두 키를 같이 보내서 sender/receiver 양쪽 시그니처에 대응 ([api.ts:69](../tone/src/lib/api.ts#L69)) |
| `/set_sender_id` | [ChatDetail.tsx](../tone/src/screens/ChatDetail.tsx) — `useEffect` (채팅방 진입 시) |
| `/play_voice` | [PlayButton.tsx](../tone/src/components/PlayButton.tsx) — 재생 버튼 클릭 |
| `/get_message` | [Profiles.tsx](../tone/src/screens/Profiles.tsx) — `handlePick` (`setMyId` 성공 후 임시 호출, message_id=1 고정) |

## 상태 관리

- `app.state.receiver: Receiver` — 라이프스팬 동안 단일 인스턴스
- `app.state.receiver_lock: Lock` — 동시성 직렬화
- `app.state.last_audio_file: str | None`

## CORS

[main.py:27-32](main.py#L27-L32)에서 `http://localhost:3000`, `http://127.0.0.1:3000` 허용. 프론트와 같은 호스트에서 실행할 거면 same-origin이라 무관.

## 알려진 한계 (백엔드 정리 필요)

1. **`/get_message` 하드코딩** — `message_id`를 무시하고 `storage/000001.json`만 읽음 ([main.py:74](main.py#L74)). `message_id` 기준 조회로 변경 필요.
2. **메시지 목록 엔드포인트 부재** — `GET /messages?conversation_id=X` 같은 list API가 없어 프론트는 현재 mock 데이터(`tone/src/data/mock.ts`)로 채팅 스레드를 채우고 있음.
3. **푸시 채널 부재** — 새 메시지 수신 시 프론트에 알릴 WebSocket/SSE 없음. `get_pending_messages()`도 서버 콘솔 출력만 함.

위 셋 중 (1) 또는 (2)·(3)이 해결되면 `useMessages` 훅을 실데이터로 전환할 수 있다.
