"""SubscribeMessages(실시간 전달) 확인 스크립트. 실서버 또는 endpoint/mock_relay.py에 접속한다.

확인 항목
  1. 받는 사람(--receiver)과 보낸 사람(--sender)이 각각 구독하고, 구독 등록 신호(initial metadata)를 받는지
  2. SendPacket 후 두 구독자 모두에게 같은 메시지가 전달되는지와 걸린 시간
  3. 같은 사용자의 두 번째 구독(다른 기기/탭 가정)에도 전달되는지

예시 (저장소 루트에서)
  .venv\\Scripts\\python.exe tests\\subscribe_check.py --ip 127.0.0.1 --port 50061 --plain
  .venv\\Scripts\\python.exe tests\\subscribe_check.py --ip 203.255.92.244 --port 8001 --ca grpc_getting_started\\certs\\ca.pem
"""
import argparse
import os
import queue
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import grpc

from src.utils.bitpacking import encode_packet
from src.utils.text_compress import compress_text


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    p = argparse.ArgumentParser(description="SubscribeMessages 확인")
    p.add_argument("--ip", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--ca", help="CA 인증서 (TLS)")
    p.add_argument("--plain", action="store_true", help="평문 접속")
    p.add_argument("--sender", type=int, default=2)
    p.add_argument("--receiver", type=int, default=0)
    p.add_argument("--timeout", type=float, default=10.0, help="전달 대기 시간(초)")
    args = p.parse_args()
    if not args.plain and not args.ca:
        p.error("TLS 접속에는 --ca가 필요합니다 (평문은 --plain)")

    os.environ.update(SERV_IP=args.ip, SERV_PORT=str(args.port), SERV_TLS="0" if args.plain else "1")
    if args.ca:
        os.environ["CA_CERT_PATH"] = os.path.abspath(args.ca)
    from grpc_getting_started import server_communicate_pb2 as pb
    from grpc_getting_started.server_communicate_connect import close_connection, set_connection

    stub = set_connection()
    results: list[tuple[str, bool, str]] = []

    def record(step, ok, detail=""):
        results.append((step, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {step}" + (f" - {detail}" if detail else ""))

    # 구독 3개: 받는 사람, 보낸 사람, 받는 사람의 두 번째 기기
    subs = {"receiver": args.receiver, "sender": args.sender, "receiver#2": args.receiver}
    calls, inboxes, threads = {}, {}, []
    for name, uid in subs.items():
        call = stub.SubscribeMessages(pb.UserIdentifier(user_id=str(uid)))  # 타임아웃 없음
        calls[name], inboxes[name] = call, queue.Queue()

        def reader(call=call, box=inboxes[name]):
            try:
                for item in call:
                    box.put((time.perf_counter(), item))
            except grpc.RpcError as e:
                if e.code() != grpc.StatusCode.CANCELLED:
                    box.put((time.perf_counter(), e))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        threads.append(t)

    # 1. 구독 등록 신호
    for name, call in calls.items():
        try:
            md = dict(call.initial_metadata())  # 등록될 때까지 대기
            record(f"1. 구독 등록 ({name}, user {subs[name]})", md.get("subscribed") == "1", f"metadata={md}")
        except grpc.RpcError as e:
            record(f"1. 구독 등록 ({name}, user {subs[name]})", False, f"{e.code()} {e.details()}")

    # 2. 전송 후 전달 확인
    text = f"실시간 전달 확인 {datetime.now():%H:%M:%S}"
    packet = encode_packet(send_id=args.sender, receiver_id=args.receiver, emo_type=1,
                           emotion_indices=[0, 1, 2, 3, 4, 5, 6, 7], message=compress_text(text))
    t0 = time.perf_counter()
    accepted = stub.SendPacket(pb.SpeechUploadPacketRequest(packet=packet), timeout=10).accepted
    record("2. SendPacket", accepted)

    ids = set()
    for name in subs:
        try:
            ts, item = inboxes[name].get(timeout=args.timeout)
        except queue.Empty:
            record(f"3. 전달 ({name})", False, f"{args.timeout:.0f}초 안에 받지 못함")
            continue
        if isinstance(item, grpc.RpcError):
            record(f"3. 전달 ({name})", False, f"{item.code()} {item.details()}")
            continue
        ok = item.message == text and item.sender_id == str(args.sender) and item.receiver_id == str(args.receiver)
        ids.add(item.message_id)
        record(f"3. 전달 ({name})", ok,
               f"{(ts - t0) * 1000:.0f}ms (전송 시작부터), message_id={item.message_id[:8]}, send_time={item.send_time.ToDatetime()}")
    record("4. 같은 message_id", len(ids) == 1, f"{len(ids)}개")

    for call in calls.values():
        call.cancel()
    for t in threads:
        t.join(timeout=3)
    close_connection()

    failed = [s for s, ok, _ in results if not ok]
    print("-" * 60)
    print(f"결과: {len(results) - len(failed)}/{len(results)} 통과" + (f", 실패: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
