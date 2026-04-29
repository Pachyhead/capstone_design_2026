"""The Python implementation of the gRPC server communicate client."""

import logging

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv


def run():
    """
    Codelab Hint: Logic for your gRPC Client will be added here.
    Steps include:
     1. Create a connection to the gRPC server using grpc.insecure_channel()
     2. Call service methods on the client to interact with the server.
    """
    load_dotenv() # load .env file's variables to os.environ
    server_addr = f"{os.environ.get('SERV_IP', '')}:{os.environ.get('SERV_PORT', '')}"
    channel = grpc.insecure_channel(server_addr) # SpeechRelayStub 인스턴스화
    stub = server_communicate_pb2_grpc.SpeechRelayStub(channel)
    
    uploadRequest = server_communicate_pb2.SpeechUploadRequest(sender_id="sendR", receiver_id="receiveR", text="anonymous text", emotion_vector=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.72]) # 서비스를 호출할 SpeechUploadRequest 정의
    status = stub.UploadSpeechTask(uploadRequest) # 서버 메서드 호출(rpc 호출)
    print(status)

    if status.accepted:
        print(f"Upload success '{status.task_id}'")
    else:
        printf(f"Upload faild")    


if __name__ == "__main__":
    logging.basicConfig()
    run()
