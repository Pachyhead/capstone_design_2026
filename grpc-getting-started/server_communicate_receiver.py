"""The Python implementation of the gRPC server communicate receive client."""

import logging

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv


def run():
    """
    Codelab Hint: Logic for your gRPC receive Client will be added here.
    Steps include:
     1. Create a connection to the gRPC server using grpc.insecure_channel()
     2. Call service methods on the client to interact with the server.
    """
    load_dotenv() # load .env file's variables to os.environ
    server_addr = f"{os.environ.get('SERV_IP', '')}:{os.environ.get('SERV_PORT', '')}"
    channel = grpc.insecure_channel(server_addr) # SpeechRelayStub 인스턴스화
    stub = server_communicate_pb2_grpc.SpeechRelayStub(channel)
    
    user_identifier = server_communicate_pb2.UserIdentifier(user_id="receiveR") # 서비스를 호출할 userIdentifier 정의
    audio_frames = stub.SubscribeSpeechStream(user_identifier) # 서버 메서드 호출(rpc 호출)
    
    for audio_frame in audio_frames:
        print(audio_frame)

        if audio_frame.sender_id:
            print(f"receive success '{audio_frame.sender_id}'")
        else:
            print(f"receive failed")


if __name__ == "__main__":
    logging.basicConfig()
    run()
