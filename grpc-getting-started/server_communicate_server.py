"""The Python implementation of the gRPC server communicate server."""

import logging
from concurrent import futures

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv

from datetime import datetime
from google.protobuf.timestamp_pb2 import Timestamp

class SpeechRelayServicer(server_communicate_pb2_grpc.SpeechRelayServicer): # pb2_grpc.SpeechRelayServicer 서브클래스화
    """Provides methods that implement functionality of server_communicate server."""
    def __init__(self, handler):
        self.handler = handler

    def Send(self, request, context): # rpc에 대한 SpeechUploadRequest 요청 전달. 제한 시간 한도 등 rpc 관련 정보 제공하는 ServicerContext 객체 전달.
        """Codelab Hint: implement Send here."""
        success = self.handler.save_incoming_speech(
            sender_id = request.sender_id,
            receiver_id = request.receiver_id,
            message = request.message,
            emo_type = request.emo_type,
            emotion_vector = list(request.emotion_vector)
        )
        return server_communicate_pb2.UploadStatus(accepted=success)
    
    def GetVoice(self, request, context): # rpc에 대한 MessageIdentifier 요청 전달.
        """ implement GetVoice here."""
        audio_chunks = self.handler.generate_voice_stream(request.message_id)

        if audio_chunks is None:
            yield server_communicate_pb2.AudioFrame(audio_content=[], sender_id=request.sender_id, message_id=request.message_id, is_final=True)
            return
        
        for chunk in audio_chunks:
            yield server_communicate_pb2.AudioFrame(
                audio_content=chunk,
                sender_id=request.sender_id,
                message_id=request.message_id,
                is_final=False
            ) # 반환값으로 AudioFrame 리턴
        
        yield server_communicate_pb2.AudioFrame(
            audio_content=b'',
            sender_id=request.sender_id,
            message_id=request.message_id,
            is_final=True
        ) # 마지막 조각임을 표시
    
    def GetPendingMessages(self, request, context): # rpc에 대한 UserIdentifier 요청 전달
        """ implement GetPendingMessages here."""
        raw_metadata_list = self.handler.get_pending_metadata(request.user_id)

        proto_items = []
        for metadata in raw_metadata_list:
            dt = datetime.strptime(metadata['send_time'], "%Y-%m-%d %H:%M:%S.%f")
            timestamp = Timestamp()
            timestamp.FromDatetime(dt)

            proto_item = server_communicate_pb2.MetadataItem(
                message_id = metadata['message_id'],
                sender_id = metadata['sender_id'],
                message = metadata['message'],
                emo_type = metadata['emo_type'],
                send_time = timestamp
            )
            proto_items.append(proto_item)
        
        return server_communicate_pb2.MetadataResponse(items = proto_items)

def serve(): # grpc 서버 시작하는 부분
    """
    Codelab Hint: Logic for starting up a gRPC Server will be added here.
    Steps include:
     1. create gRPC server using grpc.server().
     2. register SpeechRelayServicer to the server.
     3. start the server.
    """

    # TODO: AI_TTS_Handler() 클래스로 변경해야 함!!
    from file_speech_handler import FileSpeechHandler
    current_handler = FileSpeechHandler()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_communicate_pb2_grpc.add_SpeechRelayServicer_to_server(
        SpeechRelayServicer(current_handler),
        server,
    )

    load_dotenv() # load .env file's variables to os.environ
    serv_port = os.environ.get('SERV_PORT', '')
    listen_addr = f"0.0.0.0:{serv_port}" # get request from anywhere
    server.add_insecure_port(listen_addr)
    print(f"Starting server on {listen_addr}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
