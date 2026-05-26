"""The Python implementation of the gRPC server communicate receive client."""

import logging

import grpc

from .  import server_communicate_pb2
from . import server_communicate_pb2_grpc

import os
from pathlib import Path
from dotenv import load_dotenv
import io
import json
from .server_communicate_connect import set_connection
from google.protobuf.json_format import MessageToDict

import datetime

def merge_wav_byte(wav_bytes_list, storage: Path | None = None, output_filename="combined") -> Path:
    if not storage: raise ValueError(f"storage path is not specified")
    try:
        # b''를 건너뛰고 유효한 데이터를 찾음
        valid_chunks = [b for b in wav_bytes_list if b and b != b""]
        
        if not valid_chunks:
            raise ValueError("Audio do not exist")

        # 모든 바이너리 청크를 하나로 병합
        complete_wav_bytes = b"".join(valid_chunks)
        fpath = storage / f"{output_filename}.wav"

        # 바이너리 스트림 전체를 그대로 .wav 파일로 씀
        with open(fpath, "wb") as f:
            f.write(complete_wav_bytes)

        print(f"file saved successfully: {output_filename}")
        return fpath
    except OSError as e:
        raise OSError(f"File save error: {e}")

def save_message_to_json(metadata_list):
    try:
        grouped_data = {} # message_id별로 데이터 담을 딕셔너리
        for metadata in metadata_list:
            mid = metadata.message_id

            new_message = { # 받은 내용 message로 만듦
                "message_id": str(metadata.message_id),
                "sender_id": str(metadata.sender_id),
                "message": str(metadata.message),
                "emo_type": str(metadata.emo_type),
                "send_time": str(metadata.send_time.ToDatetime(tzinfo=datetime.timezone.utc))
            }

            fname = f"{mid}.json"
            with open(fname, "w", encoding="utf-8") as f: # JSON으로 저장
                json.dump(new_message, f, ensure_ascii=False, indent=4)
        return True
    except (OSError, Exception) as e:
        print(f"Error occurred: {e}")
        return False    

def GetPendingMessages(user_id):
    stub = set_connection()

    user_identifier = server_communicate_pb2.UserIdentifier(user_id=user_id) # 서비스를 호출할 userIdentifier 정의
    chatroom_items = stub.GetPendingMessages(user_identifier) # 서버 메서드 호출(메타데이터 리스트 가져옴. 약간의 구조체 있!!)   

    # 구조체 부분 제거
    chatroom_lists = []
    for chatroom_item in chatroom_items.lists:
        chatroom_list = []
        for chat_item in chatroom_item.items:
            chatroom_list.append(MessageToDict(chat_item, preserving_proto_field_name=True))
        chatroom_lists.append(chatroom_list)

    return chatroom_lists # list list dict

def GetVoice(message_id):
    stub = set_connection()
    
    message_identifier = server_communicate_pb2.MessageIdentifier(message_id=message_id)
    audio_frames = stub.GetVoice(message_identifier) # 서버 메서드 호출(음성 가져옴)

    wav_bytes_list = []
    for audio_frame in audio_frames:
        if audio_frame.audio_content:
            wav_bytes_list.append(audio_frame.audio_content)

        if audio_frame.is_final:
            print("Received final chunk from server.")
    return wav_bytes_list, message_id


def run():
    """
    Codelab Hint: Logic for your gRPC receive Client will be added here.
    Steps include:
     1. Create a connection to the gRPC server using grpc.insecure_channel()
     2. Call service methods on the client to interact with the server.
    """
    # 수신한 메시지 .json으로 저장할 때
    stub = set_connection()

    metadata_list = GetPendingMessages(user_id="000001")
    save_message_to_json(metadata_list) # json으로 저장

    # 첫번째 metadata에 대해서 바로 음성파일을 받는다고 가정
    metadata = metadata_list[0]
    message_id = metadata.message_id
    sender_id = metadata.sender_id

    # 수신한 음성 wav로 저장할 때
    wav_bytes_list = GetVoice(message_id=message_id)
    merge_wav_byte(wav_bytes_list, f"{message_id}.wav")

if __name__ == "__main__":
    logging.basicConfig()
    run()
