# Endpoint: gRPC TLS & 목 서버 테스트

GPU 서버(DB, TTS 모델) 없이 엔드포인트를 테스트할 때는 `mock_relay.py`를 사용합니다. 서버 연결은 기본적으로 TLS이며, 목 서버도 같은 규칙을 따릅니다.

## 1. 테스트 인증서 생성 (최초 1회, Git Bash)

```bash
cd endpoint
bash gen_mock_certs.sh
```

- `endpoint/mock_certs/`에 테스트 전용 CA와 서버 인증서가 만들어집니다.
- 이 폴더는 `.gitignore`로 제외되어 있으므로 각자 생성해야 합니다.
- GPU 서버 인증서와는 별개의 체인입니다. 서버 개인키는 테스트 환경에 두지 않습니다.

## 2. 엔드포인트 `.env` 설정

```
SERV_IP=127.0.0.1
SERV_PORT=50051
CA_CERT_PATH=C:/절대/경로/capstone_design_2026/endpoint/mock_certs/ca.pem
```

- `SERV_IP`는 목 서버 인증서의 SAN(`127.0.0.1`)과 맞아야 합니다.
- `CA_CERT_PATH`는 실행 위치와 무관하도록 절대 경로를 권장합니다.

## 3. 목 서버 실행

```bash
cd endpoint
uv run python mock_relay.py
```

- 시작 시 `Mock relay listening on 127.0.0.1:50051 (TLS)`가 출력되면 정상입니다.

## 평문 모드 (디버깅용)

- `SERV_TLS=0`을 엔드포인트 `.env`와 목 서버 실행 환경 모두에 설정합니다.
- 값이 없으면 TLS로 동작합니다. 평문은 `0`을 명시했을 때만 켜집니다.

## 측정 스크립트

- `tests/channel_reuse_bench.py`: 채널 재사용 전후 지연 비교
- `tests/tls_wire_capture.py`: TLS 전후 와이어 바이트와 메시지 노출 여부 확인
