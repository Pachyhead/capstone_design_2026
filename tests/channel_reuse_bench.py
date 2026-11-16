"""Channel reuse benchmark: per-call channel (old) vs cached channel (new), TLS on.

A relay proxy adds a one-way delay to every forwarded chunk (both directions) and counts
accepted TCP connections. The proxy accepts TCP locally, so the SYN is not delayed; the
TLS flights do cross the delayed path. Per-call channel cost is therefore a lower bound.

Run from the repo root with the project venv:
    .venv\\Scripts\\python.exe tests\\channel_reuse_bench.py
"""
import os
import queue
import socket
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT = os.path.join(ROOT, "endpoint")
PY = sys.executable  # 이 스크립트를 실행한 인터프리터(프로젝트 venv)를 그대로 사용
CA = os.path.join(ENDPOINT, "mock_certs", "ca.pem")
MOCK_PORT, PROXY_PORT = 50081, 50082
N = 20
sys.path.insert(0, ROOT)

from src.utils.bitpacking import encode_packet

state = {"delay": 0.0, "accepts": 0, "c2s": 0, "s2c": 0}
state_lock = threading.Lock()


def _writer(q, dst):
    while True:
        item = q.get()
        if item is None:
            break
        due, data = item
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        dst.sendall(data)


def _reader(src, q, counter_key):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            with state_lock:
                state[counter_key] += len(data)
            q.put((time.monotonic() + state["delay"], data))
    except OSError:
        pass
    finally:
        q.put(None)


def _relay_pair(cs, ss):
    q1, q2 = queue.Queue(), queue.Queue()
    for src, dst, q, key in ((cs, ss, q1, "c2s"), (ss, cs, q2, "s2c")):
        threading.Thread(target=_reader, args=(src, q, key), daemon=True).start()
        threading.Thread(target=_writer, args=(q, dst), daemon=True).start()


def proxy_loop():
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", PROXY_PORT))
    ls.listen(64)
    while True:
        cs, _ = ls.accept()
        with state_lock:
            state["accepts"] += 1
        ss = socket.create_connection(("127.0.0.1", MOCK_PORT))
        _relay_pair(cs, ss)


def reset_counters(delay):
    with state_lock:
        state.update(delay=delay, accepts=0, c2s=0, s2c=0)


def measure(label, call_once, delay):
    reset_counters(delay)
    lat = []
    for _ in range(N):
        t0 = time.perf_counter()
        call_once()
        lat.append((time.perf_counter() - t0) * 1000)
    time.sleep(0.3)
    with state_lock:
        accepts, c2s, s2c = state["accepts"], state["c2s"], state["s2c"]
    first, rest = lat[0], lat[1:]
    print(f"  {label:<8} delay={int(delay*1000):>3}ms  first={first:7.1f}ms  "
          f"mean(rest)={statistics.mean(rest):7.1f}ms  median(rest)={statistics.median(rest):7.1f}ms  "
          f"TCP conns={accepts:>3}  wire c2s={c2s:>6}B s2c={s2c:>6}B")


def main():
    env = dict(os.environ, SERV_PORT=str(MOCK_PORT), SERV_TLS="1")
    env.pop("PYTHONPATH", None)
    # 파이프 대신 파일로 받는다: 읽지 않는 PIPE가 차면 목 서버가 print에서 멈춘다
    log_path = os.path.join(os.environ.get("TEMP", ROOT), "channel_bench_mock.log")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([PY, os.path.join(ENDPOINT, "mock_relay.py")], cwd=ENDPOINT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, text=True)
    time.sleep(4)
    if proc.poll() is not None:
        print(f"mock exited, see {log_path}")
        return
    threading.Thread(target=proxy_loop, daemon=True).start()
    time.sleep(0.5)

    os.environ.update(SERV_IP="127.0.0.1", SERV_PORT=str(PROXY_PORT), SERV_TLS="1", CA_CERT_PATH=CA)
    from grpc_getting_started import server_communicate_pb2 as pb
    from grpc_getting_started import server_communicate_pb2_grpc as pbg
    from grpc_getting_started import server_communicate_connect as conn

    packet = encode_packet(send_id=2, receiver_id=3, emo_type=5,
                           emotion_indices=[0, 1, 2, 3, 4, 5, 6, 7], message="채널 재사용 측정")
    req = pb.SpeechUploadPacketRequest(packet=packet)

    def old_style():  # 이전 동작: 호출마다 새 채널 (핸드셰이크 매번)
        ch = conn._build_channel()
        try:
            pbg.SpeechRelayStub(ch).SendPacket(req, timeout=10)
        finally:
            ch.close()

    def cached():  # 현재 동작: 공유 채널 재사용
        conn.set_connection().SendPacket(req, timeout=10)

    print(f"N={N} SendPacket calls per row, TLS on, packet={len(packet)}B")
    try:
        for delay in (0.0, 0.1):
            conn.close_connection()
            measure("per-call", old_style, delay)
            conn.close_connection()
            measure("cached", cached, delay)
            conn.close_connection()
    finally:
        conn.close_connection()
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        log.close()


if __name__ == "__main__":
    main()
