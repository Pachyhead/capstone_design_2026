"""The Python implementation of the gRPC server communicate receive client."""

import logging

import grpc

import server_communicate_pb2
import server_communicate_pb2_grpc

import os
from dotenv import load_dotenv
import io
from pydub import AudioSegment

def merge_wav_byte(wav_bytes_list, output_filename="combined.wav"):
    combined = AudioSegment.empty()

    for wav_bytes in wav_bytes_list:
        audio_segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        combined += audio_segment

    combined.export(output_filename, format="wav") # 합쳐진 오디오 파일로 내보냄
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
