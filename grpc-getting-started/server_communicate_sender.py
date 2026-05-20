"""The Python implementation of the gRPC server communicate client."""

import logging

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv
import numpy as np
import json
from server_communicate_connect import set_connection

def load_vector():
    data = np.load('../000001.npy')
    return data

def load_text():
    with open('../000001_metadata.jsonl', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['text']

def SendVoice(sender_id, audio_content):
    stub = set_connection()

    uploadRequest = server_communicate_pb2.SpeechReferenceRequest(sender_id=sender_id, audio_content=audio_content) # 서비스를 호출할 SpeechReferenceRequest 정의
    status = stub.SendVoice(uploadRequest) # 서버 메서드 호출(rpc 호출)

    return status.accepted

def Send(sender_id, receiver_id, message, emo_type, emotion_vector):
    stub = set_connection()

    uploadRequest = server_communicate_pb2.SpeechUploadRequest(sender_id=sender_id, receiver_id=receiver_id, message=message, emo_type=emo_type, emotion_vector=emotion_vector) # 서비스를 호출할 SpeechUploadRequest 정의
    status = stub.Send(uploadRequest) # 서버 메서드 호출(rpc 호출)

    return status.accepted

def run():
    """
    Codelab Hint: Logic for your gRPC sender Client will be added here.
    Steps include:
     1. Create a connection to the gRPC server using grpc.insecure_channel()
     2. Call service methods on the client to interact with the server.
    """
    stub = set_connection()

    uploadRequest = server_communicate_pb2.SpeechUploadRequest(sender_id="sendR", receiver_id="000001", message=load_text(), emo_type=0, emotion_vector=load_vector()) # 서비스를 호출할 SpeechUploadRequest 정의
    status = stub.Send(uploadRequest) # 서버 메서드 호출(rpc 호출)
    
    if status.accepted:
        print(f"Upload success!")
    else:
        print(f"Upload failed!")    


if __name__ == "__main__":
    logging.basicConfig()
    run()
