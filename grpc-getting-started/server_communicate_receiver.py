"""The Python implementation of the gRPC server communicate receive client."""

import logging

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv
import io
import wave

def merge_wav_byte(wav_bytes_list, output_filename="combined.wav"):
    # b''를 건너뛰고 유효한 첫 데이터를 찾음
    first_wav_bytes = None
    for b in wav_bytes_list:
        if b and b != b"":
            first_wav_bytes = b
            break
    
    if first_wav_bytes is None:
        print("Audio do not exist")
        return

    with wave.open(io.BytesIO(first_wav_bytes), "rb") as first_wav:
        params = first_wav.getparams() # 첫 wav 데이터에서 오디오 설정 읽어옴

    with wave.open(output_filename, "wb") as output_wav: # 새로운 wav 파일 생성
        output_wav.setparams(params) # 오디어 설정 그대로 적용

        # 리스트의 모든 바이너리 데이터를 순서대로 이어붙임
        for wav_bytes in wav_bytes_list:
            if not wav_bytes or wav_bytes == b"": # b''이거나 빈 데이터는 패스
                continue
            with wave.open(io.BytesIO(wav_bytes), "rb") as input_wav:
                output_wav.writeframes(input_wav.readframes(input_wav.getnframes())) # 오디오의 실제 데이터만 읽어 결과 파일 맨 뒤에 씀
    print(f"file saved: {output_filename}")

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
    
    user_identifier = server_communicate_pb2.UserIdentifier(user_id="000001") # 서비스를 호출할 userIdentifier 정의
    metadata_list = stub.GetPendingMessages(user_identifier) # 서버 메서드 호출(메타데이터 리스트 가져옴)
    
    metadata = metadata_list.items[0]
    
    message_identifier = server_communicate_pb2.MessageIdentifier(message_id=metadata.message_id, sender_id=metadata.sender_id, receiver_id="000001")
    audio_frames = stub.GetVoice(message_identifier) # 서버 메서드 호출(음성 가져옴)
    
    wav_bytes_list = []
    for audio_frame in audio_frames:
        wav_bytes_list.append(audio_frame.audio_content)

        if audio_frame.sender_id:
            print(f"receive success '{audio_frame.sender_id}'")
        else:
            print(f"receive failed")

    merge_wav_byte(wav_bytes_list, f"{metadata.message_id}.wav")


if __name__ == "__main__":
    logging.basicConfig()
    run()
