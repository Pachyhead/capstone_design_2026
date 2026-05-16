"""The Python implementation of the gRPC server communicate server."""

import logging
from concurrent import futures

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv
import use_e2v_text_data

from datetime import datetime
from google.protobuf.timestamp_pb2 import Timestamp

def get_sender_response(request): # request 받으면 doSomething을 하고, uploadStatus(True) 반환
    try:
        use_e2v_text_data.doSomething(request) # 이후 서버단 동작으로 교체(autoencoder / stt)
    except OSError as e:
        printf(f"Response error: {e}")
        return None
    
    return server_communicate_pb2.UploadStatus(accepted=True) # 반환값으로 UploadStatus 리턴

def get_metadata_response(request): # request 받으면 get_metadata를 호출해 MetadataResponse 반환
    metadata_list = use_e2v_text_data.getMetadataList(request.user_id) # 이후 서버단 동작으로 교체(sqlite)
    if metadata_list is None:
        return None

    dt = datetime.strptime(metadata_list[0]['send_time'], "%Y-%m-%d %H:%M:%S.%f")
    timestamp = Timestamp()
    timestamp.FromDatetime(dt)
    metadata_list[0]['send_time'] = timestamp

    return server_communicate_pb2.MetadataResponse(items=metadata_list) # 반환값으로 MetadataResponse 리턴

class SpeechRelayServicer(server_communicate_pb2_grpc.SpeechRelayServicer): # pb2_grpc.SpeechRelayServicer 서브클래스화
    """Provides methods that implement functionality of server_communicate server."""

    def Send(self, request, context): # rpc에 대한 SpeechUploadRequest 요청 전달. 제한 시간 한도 등 rpc 관련 정보 제공하는 ServicerContext 객체 전달.
        """Codelab Hint: implement Send using get_sender_response() here."""
        
        response = get_sender_response(request)
        if response is None:
            return server_communicate_pb2.UploadStatus(accepted=False)
        else:
            return response
    
    def GetVoice(self, request, context): # rpc에 대한 MessageIdentifier 요청 전달.
        """ implement GetVoice here."""
        audio_contents = use_e2v_text_data.getVoice(request) # 이후 서버단 동작으로 교체(tts)
        
        if audio_contents is None:
            audio_frame = server_communicate_pb2.AudioFrame(audio_content=[], sender_id=request.sender_id, message_id=request.message_id, is_final=True)
            yield audio_frame
        
        for audio_content in audio_contents:
            yield server_communicate_pb2.AudioFrame(
                audio_content=audio_content,
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

        response = get_metadata_response(request)
        if response is None:
            return server_communicate_pb2.MetadataResponse(lists=[])
        else:
            return response

def serve(): # grpc 서버 시작하는 부분
    """
    Codelab Hint: Logic for starting up a gRPC Server will be added here.
    Steps include:
     1. create gRPC server using grpc.server().
     2. register SpeechRelayServicer to the server.
     3. start the server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_communicate_pb2_grpc.add_SpeechRelayServicer_to_server(
        SpeechRelayServicer(),
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
