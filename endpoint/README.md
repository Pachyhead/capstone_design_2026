# Endpoint: gRPC TLS & 목 서버 테스트

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

## 측정 스크립트

- `tests/channel_reuse_bench.py`: 채널 재사용 전후 지연 비교
- `tests/tls_wire_capture.py`: TLS 전후 와이어 바이트와 메시지 노출 여부 확인
