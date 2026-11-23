"""The Python implementation of the gRPC server communicate server."""

import logging
from concurrent import futures

import grpc

from . import server_communicate_pb2
from . import server_communicate_pb2_grpc

import os
from .env_config import load_env

from datetime import datetime
from google.protobuf.timestamp_pb2 import Timestamp

from src.utils.bitpacking import decode_packet
from src.utils.text_compress import decompress_text
from .message_broker import MessageBroker

# 열린 구독 스트림(SubscribeMessages) 하나가 워커 스레드 하나를 계속 차지하므로 넉넉히 둔다
MAX_WORKERS = 50

# 구독 스트림은 오래 조용히 열려 있으므로 keepalive로 끊긴 연결을 정리하고, 클라이언트의 keepalive ping을 허용한다
SERVER_OPTIONS = [
    ("grpc.keepalive_time_ms", 60000),                              # 서버 -> 클라이언트 ping 주기
    ("grpc.keepalive_timeout_ms", 20000),                           # ping 응답 대기
    ("grpc.keepalive_permit_without_calls", 1),                     # 호출이 없어도 ping 허용
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),   # 클라이언트 ping 최소 간격(이보다 잦으면 거부)
    ("grpc.http2.max_pings_without_data", 0),                       # 데이터 없이 보내는 ping 수 제한 없음
]


def to_metadata_item(metadata: dict) -> server_communicate_pb2.MetadataItem:
    """핸들러가 돌려준 메타데이터 dict를 MetadataItem으로 변환한다"""
    return server_communicate_pb2.MetadataItem(
        message_id = metadata['message_id'],
        sender_id = metadata['sender_id'],
        receiver_id = metadata['receiver_id'],
        message = metadata['message'],
        emo_type = metadata['emo_type'],
        send_time = metadata['send_time']
    )


class SpeechRelayServicer(server_communicate_pb2_grpc.SpeechRelayServicer): # pb2_grpc.SpeechRelayServicer 서브클래스화
    """Provides methods that implement functionality of server_communicate server."""
    def __init__(self, handler, broker: MessageBroker | None = None):
        self.handler = handler
        self.broker = broker or MessageBroker()

    def _notify_saved(self, message_id: str) -> None:
        """저장된 메시지를 받는 사람과 보낸 사람(다른 기기/탭)의 구독자에게 전달한다.
        전달 실패가 전송 결과에 영향을 주지 않도록 예외는 기록만 한다."""
        try:
            metadata = self.handler.get_message_metadata(message_id)
            if not metadata:
                return
            item = to_metadata_item(metadata)
            for user_id in {item.sender_id, item.receiver_id}:
                self.broker.publish(user_id, item)
        except Exception as e:
            logging.warning(f"notify failed for {message_id}: {e}")

    def SendVoice(self, request, context): # rpc에 대한 SpeechReferenceRequest 요청 전달
        """ implement SendVoice here."""
        success = self.handler.save_incoming_reference(
            sender_id = request.sender_id,
            audio_content = request.audio_content
        )
        return server_communicate_pb2.UploadStatus(accepted=success)

    def Send(self, request, context): # rpc에 대한 SpeechUploadRequest 요청 전달. 제한 시간 한도 등 rpc 관련 정보 제공하는 ServicerContext 객체 전달. --> deprecated. 호환성을 위해 남겨둠.
        """Codelab Hint: implement Send here."""
        message_id = self.handler.save_incoming_speech(
            sender_id = request.sender_id,
            receiver_id = request.receiver_id,
            message = request.message,
            emo_type = request.emo_type,
            emotion_vector = list(request.emotion_vector)
        )
        if message_id:
            self._notify_saved(message_id)
        return server_communicate_pb2.UploadStatus(accepted=bool(message_id))

    def SendPacket(self, request, context):
        send_id, receiver_id, emo_type, emotion_indices, message = decode_packet(request.packet)
        message = decompress_text(message) # bytes -> 텍스트 복원 (엔드포인트 compress_text의 역연산)

        message_id = self.handler.save_incoming_speech(
            sender_id = send_id,
            receiver_id = receiver_id,
            message = message,
            emo_type = emo_type,
            emotion_vector = list(emotion_indices)
        )
        if message_id:
            self._notify_saved(message_id)
        return server_communicate_pb2.UploadStatus(accepted=bool(message_id))

    def GetVoice(self, request, context): # rpc에 대한 MessageIdentifier 요청 전달.
        """ implement GetVoice here."""
        audio_chunks = self.handler.generate_voice_stream(request.message_id)

        if audio_chunks is None:
            yield server_communicate_pb2.AudioFrame(audio_content=[], message_id=request.message_id, is_final=True)
            return
        
        for chunk in audio_chunks:
            yield server_communicate_pb2.AudioFrame(
                audio_content=chunk,
                message_id=request.message_id,
                is_final=False
            ) # 반환값으로 AudioFrame 리턴
        
        yield server_communicate_pb2.AudioFrame(
            audio_content=b'',
            message_id=request.message_id,
            is_final=True
        ) # 마지막 조각임을 표시
    
    def GetPendingMessages(self, request, context): # rpc에 대한 UserIdentifier 요청 전달
        """ implement GetPendingMessages here."""
        raw_chatroom_items = self.handler.get_pending_metadata(request.user_id)

        chatroom_items = []
        for raw_chatroom_item in raw_chatroom_items:
            chatroom_item = [to_metadata_item(metadata) for metadata in raw_chatroom_item]
            chatroom_items.append(server_communicate_pb2.MetadataList(items=chatroom_item))

        return server_communicate_pb2.MetadataResponse(lists = chatroom_items)

    def SubscribeMessages(self, request, context): # rpc에 대한 UserIdentifier 요청 전달. 새 메시지를 스트림으로 보냄
        """ 연결이 유지되는 동안 user_id가 받거나 보낸 새 메시지를 보낸다."""
        if not request.user_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id is required")
        yield from self.broker.stream(request.user_id, context)

def serve(): # grpc 서버 시작하는 부분
    """
    Codelab Hint: Logic for starting up a gRPC Server will be added here.
    Steps include:
     1. create gRPC server using grpc.server().
     2. register SpeechRelayServicer to the server.
     3. start the server.
    """

    # TODO: AI_TTS_Handler() 클래스로 변경해야 함!!
    from .file_speech_handler import FileSpeechHandler
    current_handler = FileSpeechHandler()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS), options=SERVER_OPTIONS)
    server_communicate_pb2_grpc.add_SpeechRelayServicer_to_server(
        SpeechRelayServicer(current_handler),
        server,
    )

    load_env() # 저장소 루트 .env를 os.environ으로 읽음
    serv_port = os.environ.get('SERV_PORT', '')
    listen_addr = f"0.0.0.0:{serv_port}" # get request from anywhere
    # SERV_TLS=1일 때만 TLS. 값이 없으면 평문 (인증서가 준비되지 않아도 오류 없이 뜨도록 기본값은 평문)
    tls = os.environ.get('SERV_TLS', '0') == '1'
    if tls:
        with open(os.environ.get('SERVER_KEY_PATH', ''), 'rb') as f:
            server_key = f.read()
        with open(os.environ.get('SERVER_CERT_PATH', ''), 'rb') as f:
            server_cert = f.read()
        server_credentials = grpc.ssl_server_credentials([(server_key, server_cert)])
        server.add_secure_port(listen_addr, server_credentials)
    else:
        server.add_insecure_port(listen_addr)

    print(f"Starting server on {listen_addr} ({'TLS' if tls else 'plaintext'})")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
