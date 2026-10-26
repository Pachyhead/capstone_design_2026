import grpc
from dotenv import load_dotenv
import os
from . import server_communicate_pb2_grpc

def set_connection(): # set connection to server
    load_dotenv() # load .env file's variables to os.environ
    server_addr = f"{os.environ.get('SERV_IP', '')}:{os.environ.get('SERV_PORT', '')}"

    # 사설 CA 인증서로 서버 인증서를 검증하는 TLS 채널
    with open(os.environ.get('CA_CERT_PATH', ''), 'rb') as f:
        ca_cert = f.read()
    credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
    channel = grpc.secure_channel(server_addr, credentials) # SpeechRelayStub 인스턴스화
    stub = server_communicate_pb2_grpc.SpeechRelayStub(channel)

    return stub