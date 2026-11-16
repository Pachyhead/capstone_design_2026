"""Measure wire bytes and plaintext visibility for one SendPacket, TLS vs plaintext.

A TCP relay sits between client and mock server and records every byte in both
directions. No decryption: TLS is judged by whether the message bytes appear in the
captured stream at all.

Run from the repo root with the project venv:
    .venv\\Scripts\\python.exe tests\\tls_wire_capture.py
"""
import os
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT = os.path.join(ROOT, "endpoint")
PY = sys.executable  # 이 스크립트를 실행한 인터프리터(프로젝트 venv)를 그대로 사용
CA = os.path.join(ENDPOINT, "mock_certs", "ca.pem")
MARKER = "캡처마커-ABC123"
sys.path.insert(0, ROOT)

from src.utils.bitpacking import encode_packet


class Capture:
    def __init__(self):
        self.c2s = bytearray()
        self.s2c = bytearray()
        self.lock = threading.Lock()


def pipe(src, dst, buf, lock):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            with lock:
                buf.extend(data)
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def relay(listen_port, target_port, cap):
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", listen_port))
    ls.listen(8)
    while True:
        cs, _ = ls.accept()
        ss = socket.create_connection(("127.0.0.1", target_port))
        threading.Thread(target=pipe, args=(cs, ss, cap.c2s, cap.lock), daemon=True).start()
        threading.Thread(target=pipe, args=(ss, cs, cap.s2c, cap.lock), daemon=True).start()


def run(mode: str, mock_port: int, proxy_port: int) -> None:
    tls = mode == "tls"
    env = dict(os.environ, SERV_PORT=str(mock_port), SERV_TLS="1" if tls else "0")
    env.pop("PYTHONPATH", None)
    proc = subprocess.Popen(
        [PY, os.path.join(ENDPOINT, "mock_relay.py")], cwd=ENDPOINT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(4)
    if proc.poll() is not None:
        print(f"[{mode}] mock exited:", proc.stdout.read())
        return

    cap = Capture()
    threading.Thread(target=relay, args=(proxy_port, mock_port, cap), daemon=True).start()
    time.sleep(0.5)

    # 클라이언트는 저장소의 set_connection()을 그대로 사용 (프록시 포트로 접속)
    os.environ.update(SERV_IP="127.0.0.1", SERV_PORT=str(proxy_port), SERV_TLS="1" if tls else "0",
                      CA_CERT_PATH=CA)
    from grpc_getting_started import server_communicate_pb2 as pb
    from grpc_getting_started.server_communicate_connect import close_connection, set_connection

    close_connection()  # 이전 실행의 채널이 남지 않도록 초기화
    stub = set_connection()
    stub.GetPendingMessages(pb.UserIdentifier(user_id="0"), timeout=5)  # 워밍업 (핸드셰이크 분리)

    packet = encode_packet(send_id=2, receiver_id=3, emo_type=5,
                           emotion_indices=[0, 1, 2, 3, 4, 5, 6, 7], message=MARKER)
    with cap.lock:
        c_before, s_before = len(cap.c2s), len(cap.s2c)
    st = stub.SendPacket(pb.SpeechUploadPacketRequest(packet=packet), timeout=5)
    time.sleep(0.3)
    with cap.lock:
        c2s = bytes(cap.c2s[c_before:])
        s2c = bytes(cap.s2c[s_before:])
        whole = bytes(cap.c2s) + bytes(cap.s2c)

    marker_bytes = MARKER.encode("utf-8")
    print(f"== mode={mode}")
    print(f"   accepted={st.accepted}  app packet bytes={len(packet)}")
    print(f"   wire bytes SendPacket: client->server={len(c2s)}  server->client={len(s2c)}")
    print(f"   app packet visible in capture: {packet in whole}")
    print(f"   marker visible in capture:     {marker_bytes in whole}")

    close_connection()
    proc.terminate()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()


if __name__ == "__main__":
    run("plain", mock_port=50061, proxy_port=50062)
    run("tls", mock_port=50063, proxy_port=50064)
