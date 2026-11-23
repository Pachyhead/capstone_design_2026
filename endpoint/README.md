# Endpoint: gRPC TLS, 실시간 수신 & 목 서버 테스트

GPU 서버(DB, TTS 모델) 없이 엔드포인트를 테스트할 때는 `mock_relay.py`를 사용합니다. 서버 연결은 기본적으로 **평문**이고, `SERV_TLS=1`일 때만 TLS로 동작합니다. 목 서버도 같은 규칙을 따릅니다.

## 1. 테스트 인증서 생성 (TLS 모드에서만 필요, 최초 1회, Git Bash)

```bash
cd endpoint
bash gen_mock_certs.sh
```

- `endpoint/mock_certs/`에 테스트 전용 CA와 서버 인증서가 만들어집니다.
- 이 폴더는 `.gitignore`로 제외되어 있으므로 각자 생성해야 합니다.
- GPU 서버 인증서와는 별개의 체인입니다. 서버 개인키는 테스트 환경에 두지 않습니다.

## 2. `.env` 설정 (저장소 루트)

`.env`는 **저장소 루트에 하나만** 둡니다. 엔드포인트, 목 서버, gRPC 연결 코드가 모두 이 파일을 읽습니다.

```bash
# 저장소 루트에서
cp .env.client.example .env
```

TLS 모드로 목 서버를 쓸 때의 예시입니다.

```
SERV_IP=127.0.0.1
SERV_PORT=50051
SERV_TLS=1
CA_CERT_PATH=C:/절대/경로/capstone_design_2026/endpoint/mock_certs/ca.pem
```

- 서버 머신에서는 `.env.server.example`을 루트 `.env`로 복사해서 씁니다.
- `SERV_IP`는 TLS일 때 서버 인증서의 SAN과 같아야 합니다 (목 서버는 `127.0.0.1`).
- `CA_CERT_PATH`는 실행 위치와 무관하도록 절대 경로를 권장합니다.
- 이미 설정된 환경변수는 `.env`보다 우선합니다.

## 3. 목 서버 실행

```bash
cd endpoint
uv run python mock_relay.py
```

- 시작 시 `Mock relay listening on 127.0.0.1:50051 (plaintext)` 또는 `(TLS)`가 출력되면 정상입니다.
- 목 서버도 엔드포인트와 같은 `SERV_TLS` 값으로 실행해야 합니다. 값이 다르면 연결이 끊깁니다.

## TLS 모드

- 루트 `.env`에 `SERV_TLS=1`을 설정하면 엔드포인트와 목 서버가 함께 TLS로 동작합니다.
- 값이 없거나 `0`이면 평문입니다. 평문에서는 전송 내용이 암호화되지 않으므로, 실제 배포에서는 TLS를 켜는 것을 권장합니다.

## 실시간 수신 (`GET /events`)

새 메시지를 새로고침 없이 화면에 보내는 SSE(Server-Sent Events) 엔드포인트입니다.

```
GPU 서버 ──SubscribeMessages(gRPC 서버 스트리밍)──> 엔드포인트 ──/events(SSE)──> 브라우저
```

- 엔드포인트는 `/set_my_id`로 사용자가 정해지면 서버 구독을 시작합니다 (`sender_src/inbox_stream.py`).
- 서버는 메시지가 저장될 때마다 **보낸 사람과 받는 사람 모두에게** 보냅니다. 그래서 보낸 사람의 다른 기기에도 표시됩니다.
- 화면은 `tone/src/hooks/useInboxEvents.ts`에서 `EventSource('/events')`로 받습니다.

### 이벤트 형식

| 이벤트 | `data` | 화면 동작 |
|---|---|---|
| `message` | 메시지 1건 (`message_id`, `sender_id`, `receiver_id`, `message`, `emo_type`, `send_time`) | 상대 대화방에 추가 (`message_id`가 같으면 무시) |
| `resync` | `{"user_id": N}` 또는 `null` | `/set_my_id`로 목록을 다시 받음 |
| `: ping` (주석) | — | 15초마다 보내는 연결 확인용, 화면에서는 무시 |

```
retry: 3000

event: message
data: {"message_id": "…", "sender_id": "2", "receiver_id": "0", "message": "…", "emo_type": "4", "send_time": "…"}

event: resync
data: {"user_id": 2}
```

### 끊김과 재연결

| 상황 | 동작 |
|---|---|
| 서버(GPU) 재시작, 네트워크 끊김 | 엔드포인트가 1초부터 두 배씩(최대 10초) 늘려 가며 다시 구독. 구독이 맺어지면 `resync`를 보내 끊긴 동안의 메시지를 목록 재조회로 채움 |
| 엔드포인트 재시작 | 브라우저가 3초마다(`retry: 3000`) 다시 접속. 다시 연결되면 화면이 `set_my_id` → `set_receiver_id`를 다시 보냄 |
| 유휴 | gRPC keepalive(클라이언트 30초)와 SSE ping(15초)으로 연결 유지. 실서버에서 20분 유휴 확인 |

- 엔드포인트는 재시작하면 기본값(사용자 1, 수신자 2)으로 돌아갑니다. 그래서 `/set_my_id`를 받기 전에는 `/start_recording`, `/send_ref`를, `/set_receiver_id`까지 받기 전에는 `/send`를 **409**로 거부합니다. 화면은 409를 받으면 id를 다시 보내고 "잠시 후 다시 시도해 주세요" 안내를 띄웁니다.
- `/events`는 SPA fallback(`GET /{full_path:path}`)보다 **먼저** 등록해야 합니다. 뒤에 두면 fallback이 요청을 가져가 `index.html`이 돌아갑니다.
- 리버스 프록시를 둘 경우 `/events`의 응답 버퍼링을 꺼야 합니다 (`X-Accel-Buffering: no` 헤더는 이미 보냄).

### 제약

- 엔드포인트 하나는 사용자 한 명만 관리합니다. 같은 엔드포인트에 **다른 사용자로** 탭 두 개를 열면 서로 사용자를 바꾸며 `/set_my_id`가 계속 오갑니다. 한 PC에서 두 사용자를 쓰려면 포트를 나눠 엔드포인트를 따로 띄우세요 (예: `--port 8000`, `--port 8010`).
- 같은 사용자로 탭 여러 개를 여는 것은 문제없습니다.
- 서버와 엔드포인트는 같은 버전으로 배포해야 합니다. 구버전 서버에서는 구독이 `UNIMPLEMENTED`로 실패하고 재시도를 반복합니다 (새로고침 버튼으로 목록 조회는 가능).

### 확인 방법

```bash
# 브라우저 없이 SSE 스트림 보기
curl -N http://127.0.0.1:8000/events

# 서버 구독 동작 확인 (저장소 루트에서, 수신자·발신자 구독 → 전송 → 양쪽 도착 확인)
python tests/subscribe_check.py --ip 127.0.0.1 --port 50051 --plain

# 브로커/서버 단위 테스트
python -m pytest tests/test_message_broker.py
```

- 목 서버(`mock_relay.py`)도 `SubscribeMessages`를 지원합니다. 구독 연결과 해제가 `[Subscribe]` 로그로 출력됩니다.
- 실서버 TLS 모드에서는 `subscribe_check.py`에 `--plain` 대신 `--ca <ca.pem 경로>`를 넘깁니다.

## 측정 스크립트

- `tests/channel_reuse_bench.py`: 채널 재사용 전후 지연 비교
- `tests/tls_wire_capture.py`: TLS 전후 와이어 바이트와 메시지 노출 여부 확인
