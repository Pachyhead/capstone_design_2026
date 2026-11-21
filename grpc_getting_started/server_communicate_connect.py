import grpc
from .env_config import load_env
import json
import os
import threading
from . import server_communicate_pb2_grpc

# 프로세스당 채널 하나를 만들어 재사용한다 (호출마다 TCP+TLS 핸드셰이크를 하지 않도록)
_channel = None
_channel_lock = threading.Lock()

# RPC 타임아웃(초). 서버에 닿지 않을 때 요청이 무한정 기다리지 않도록 한다 (재시도 시간 포함)
RPC_TIMEOUT = 10      # 목록 조회, 패킷 전송
UPLOAD_TIMEOUT = 30   # 레퍼런스 음성 업로드
STREAM_TIMEOUT = 120  # TTS 음성 스트리밍 (서버 합성 시간 포함, 실서버 첫 데이터 6~8초)

# 유휴 후 끊긴 연결에 대한 재시도 (UNAVAILABLE). 읽기 RPC에만 적용한다.
# SendPacket/SendVoice는 재시도하면 같은 메시지가 두 번 저장될 수 있어서 제외한다.
_RETRY_SERVICE_CONFIG = json.dumps({
    "methodConfig": [{
        "name": [
            {"service": "SpeechRelay", "method": "GetPendingMessages"},
            {"service": "SpeechRelay", "method": "GetVoice"},
        ],
        "retryPolicy": {
            "maxAttempts": 3,
            "initialBackoff": "0.2s",
            "maxBackoff": "2s",
            "backoffMultiplier": 2,
            "retryableStatusCodes": ["UNAVAILABLE"],
        },
    }]
})
_CHANNEL_OPTIONS = [
    ("grpc.enable_retries", 1),
    ("grpc.service_config", _RETRY_SERVICE_CONFIG),
]


def _build_channel():
    load_env() # 저장소 루트 .env를 os.environ으로 읽음
    server_addr = f"{os.environ.get('SERV_IP', '')}:{os.environ.get('SERV_PORT', '')}"

    # SERV_TLS=1일 때만 TLS. 값이 없으면 평문 (인증서가 준비되지 않아도 오류 없이 연결되도록 기본값은 평문)
    if os.environ.get('SERV_TLS', '0') == '1':
        # 사설 CA 인증서로 서버 인증서를 검증하는 TLS 채널
        with open(os.environ.get('CA_CERT_PATH', ''), 'rb') as f:
            ca_cert = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
        return grpc.secure_channel(server_addr, credentials, options=_CHANNEL_OPTIONS)


def set_connection(): # set connection to server
    global _channel
    with _channel_lock:
        if _channel is None:
            _channel = _build_channel() # 환경변수(SERV_IP, SERV_TLS 등)는 첫 호출 시점에 고정된다
    return server_communicate_pb2_grpc.SpeechRelayStub(_channel)


def close_connection():
    """종료 시 공유 채널을 닫는다. 다음 set_connection() 호출에서 새로 만든다."""
    global _channel
    with _channel_lock:
        if _channel is not None:
            _channel.close()
            _channel = None