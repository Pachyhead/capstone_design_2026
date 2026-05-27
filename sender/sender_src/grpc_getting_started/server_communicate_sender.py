"""The Python implementation of the gRPC server communicate client."""

import logging

import grpc

from . import server_communicate_pb2
from . import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv
import numpy as np
import json
from .server_communicate_connect import set_connection

def load_vector():
    data = np.load('../000001.npy')
    return data

def load_text():
    with open('../000001_metadata.jsonl', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['text']

def SendVoice(sender_id, audio_path):
    stub = set_connection()

    if not os.path.exists(audio_path):
        return False

    with open(audio_path, "rb") as f:
        audio_content = f.read()

    uploadRequest = server_communicate_pb2.SpeechReferenceRequest(sender_id=sender_id, audio_content=audio_content) # 서비스를 호출할 SpeechReferenceRequest 정의
    status = stub.SendVoice(uploadRequest) # 서버 메서드 호출(rpc 호출)

    return status.accepted

def Send(sender_id, receiver_id, message, emo_type, emotion_vector):
    stub = set_connection()

    uploadRequest = server_communicate_pb2.SpeechUploadRequest(sender_id=sender_id, receiver_id=receiver_id, message=message, emo_type=emo_type, emotion_vector=list(emotion_vector)) # 서비스를 호출할 SpeechUploadRequest 정의
    status = stub.Send(uploadRequest) # 서버 메서드 호출(rpc 호출)

    return status.accepted

def GetPendingMessages(user_id):
    stub = set_connection()

    user_identifier = server_communicate_pb2.UserIdentifier(user_id=user_id) # 서비스를 호출할 userIdentifier 정의
    chatroom_items = stub.GetPendingMessages(user_identifier) # 서버 메서드 호출(메타데이터 리스트 가져옴. 약간의 구조체 있!!)   

    # 구조체 부분 제거
    chatroom_lists = []
    for chatroom_item in chatroom_items.lists:
        chatroom_list = []
        for chat_item in chatroom_item.items:
            chatroom_list.append(chat_item)
        chatroom_lists.append(chatroom_list)

    return chatroom_lists # list list dict

def run():
    """
    Codelab Hint: Logic for your gRPC sender Client will be added here.
    Steps include:
     1. Create a connection to the gRPC server using grpc.insecure_channel()
     2. Call service methods on the client to interact with the server.
    """
    voice_accepted = SendVoice(sender_id="sendR", audio_path="./000001.wav")
    if voice_accepted:
        print(f"Reference upload success!")
    else:
        print(f"Reference upload failed!")

    speech_accepted = Send(sender_id="sendR", receiver_id="000001", message=load_text(), emo_type=0, emotion_vector=[0,1,2,1,3,5])
    
    if speech_accepted:
        print(f"Speech upload success!")
    else:
        print(f"Speech upload failed!")    


if __name__ == "__main__":
    logging.basicConfig()
    run()