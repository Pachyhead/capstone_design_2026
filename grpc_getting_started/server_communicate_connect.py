import grpc
from dotenv import load_dotenv
import os
from . import server_communicate_pb2_grpc

def set_connection(): # set connection to server
    load_dotenv() # load .env file's variables to os.environ
    server_addr = f"{os.environ.get('SERV_IP', '')}:{os.environ.get('SERV_PORT', '')}"

    # SERV_TLS=0일 때만 평문. 값이 없으면 TLS (실수로 평문으로 떨어지지 않게 기본값은 TLS)
    if os.environ.get('SERV_TLS', '1') != '0':
        # 사설 CA 인증서로 서버 인증서를 검증하는 TLS 채널
        with open(os.environ.get('CA_CERT_PATH', ''), 'rb') as f:
            ca_cert = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
        channel = grpc.secure_channel(server_addr, credentials) # SpeechRelayStub 인스턴스화
    else:
        channel = grpc.insecure_channel(server_addr)
    stub = server_communicate_pb2_grpc.SpeechRelayStub(channel)

    return stub