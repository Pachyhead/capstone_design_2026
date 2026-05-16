"""The Python implementation of the gRPC server communicate client."""

import logging

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv
import numpy as np
import json

def load_vector():
    data = np.load('../000001.npy')
    return data

def load_text():
    with open('../000001_metadata.jsonl', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['text']

def run():
    """
    Codelab Hint: Logic for your gRPC sender Client will be added here.
    Steps include:
     1. Create a connection to the gRPC server using grpc.insecure_channel()
     2. Call service methods on the client to interact with the server.
    """
    load_dotenv() # load .env file's variables to os.environ
    server_addr = f"{os.environ.get('SERV_IP', '')}:{os.environ.get('SERV_PORT', '')}"
    channel = grpc.insecure_channel(server_addr) # SpeechRelayStub 인스턴스화
    stub = server_communicate_pb2_grpc.SpeechRelayStub(channel)
    

    uploadRequest = server_communicate_pb2.SpeechUploadRequest(sender_id="sendR", receiver_id="000001", message=load_text(), emo_type=0, emotion_vector=load_vector()) # 서비스를 호출할 SpeechUploadRequest 정의
    status = stub.Send(uploadRequest) # 서버 메서드 호출(rpc 호출)
    
    if status.accepted:
        print(f"Upload success!")
    else:
        printf(f"Upload failed!")    


if __name__ == "__main__":
    logging.basicConfig()
    run()
