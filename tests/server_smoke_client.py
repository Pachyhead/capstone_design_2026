"""GPU 서버(또는 목 서버) 스모크 테스트 클라이언트.

엔드포인트 UI 없이 엔드포인트와 같은 연결 경로(set_connection)로 서버를 확인한다.
녹음/STT는 쓰지 않으므로 torch, mlx-whisper 없이 실행된다.

단계
  1. TLS 연결 + GetPendingMessages
  2. SendPacket (compress_text로 인코딩한 메시지)
  3. 저장 확인: 수신자 목록에서 원문이 복원되어 저장됐는지
  4. (--voice) GetVoice 스트리밍
  5. (--negative) 평문 접속 / 잘못된 CA 접속이 거부되는지

예시 (저장소 루트에서)
  .venv\\Scripts\\python.exe tests\\server_smoke_client.py --ip 203.255.92.244 --port 50051 ^
      --ca grpc_getting_started\\certs\\ca.pem --voice --negative --wrong-ca endpoint\\mock_certs\\ca.pem
"""
import argparse
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import grpc

from src.utils.bitpacking import encode_packet
from src.utils.text_compress import compress_text

results: list[tuple[str, bool, str]] = []


def record(step: str, ok: bool, detail: str = "") -> bool:
    results.append((step, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {step}" + (f" - {detail}" if detail else ""))
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpeechRelay 서버 스모크 테스트")
    p.add_argument("--ip", required=True, help="서버 IP (인증서 SAN과 같아야 함)")
    p.add_argument("--port", required=True, help="서버 포트")
    p.add_argument("--ca", help="CA 인증서 경로 (TLS일 때 필수)")
    p.add_argument("--plain", action="store_true", help="평문으로 접속 (SERV_TLS=0, 디버깅용)")
    p.add_argument("--sender", type=int, default=0, help="발신자 ID (0~3)")
    p.add_argument("--receiver", type=int, default=1, help="수신자 ID (0~3)")
    p.add_argument("--message", help="보낼 메시지 (기본: 시각이 들어간 한글 문장)")
    p.add_argument("--timeout", type=float, default=30.0, help="RPC 타임아웃(초)")
    p.add_argument("--voice", action="store_true", help="저장된 메시지로 GetVoice 스트리밍 확인 (TTS 필요)")
    p.add_argument("--voice-timeout", type=float, default=120.0, help="GetVoice 타임아웃(초)")
    p.add_argument("--save-wav", help="GetVoice 결과를 저장할 경로")
    p.add_argument("--negative", action="store_true", help="평문/잘못된 CA 접속이 거부되는지 확인")
    p.add_argument("--wrong-ca", help="--negative에서 쓸 다른 CA 경로 (예: endpoint/mock_certs/ca.pem)")
    args = p.parse_args()
    if not args.plain and not args.ca:
        p.error("TLS 접속에는 --ca가 필요합니다 (평문은 --plain)")
    return args


def configure_env(args: argparse.Namespace) -> None:
    # set_connection()은 load_dotenv()를 호출하지만 이미 있는 환경변수는 덮어쓰지 않는다
    os.environ["SERV_IP"] = args.ip
    os.environ["SERV_PORT"] = str(args.port)
    os.environ["SERV_TLS"] = "0" if args.plain else "1"
    if args.ca:
        os.environ["CA_CERT_PATH"] = os.path.abspath(args.ca)


def find_items(resp, text: str):
    return [it for lst in resp.lists for it in lst.items if it.message == text]


def main() -> int:
    # Windows 콘솔(cp949)에서 이모지 등 출력 시 실패하지 않도록
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    configure_env(args)

    from grpc_getting_started import server_communicate_pb2 as pb
    from grpc_getting_started.server_communicate_connect import close_connection, set_connection
    from grpc_getting_started.server_communicate_pb2_grpc import SpeechRelayStub

    addr = f"{args.ip}:{args.port}"
    print(f"target={addr} mode={'plaintext' if args.plain else 'TLS'} "
          f"sender={args.sender} receiver={args.receiver}")

    stub = set_connection()

    # 1. 연결 + 목록
    try:
        t0 = time.perf_counter()
        resp = stub.GetPendingMessages(pb.UserIdentifier(user_id=str(args.sender)), timeout=args.timeout)
        ms = (time.perf_counter() - t0) * 1000
        counts = [len(lst.items) for lst in resp.lists]
        record("1. 연결 + GetPendingMessages", True, f"{ms:.0f}ms, 채팅방별 개수={counts}")
    except grpc.RpcError as e:
        record("1. 연결 + GetPendingMessages", False, f"{e.code()} {e.details()}")
        close_connection()
        return summary()

    # 2. SendPacket
    text = args.message or f"서버 스모크 테스트 {datetime.now():%Y-%m-%d %H:%M:%S}"
    blob = compress_text(text)
    packet = encode_packet(send_id=args.sender, receiver_id=args.receiver, emo_type=2,
                           emotion_indices=[0, 1, 2, 3, 4, 5, 6, 7], message=blob)
    try:
        t0 = time.perf_counter()
        accepted = stub.SendPacket(pb.SpeechUploadPacketRequest(packet=packet), timeout=args.timeout).accepted
        ms = (time.perf_counter() - t0) * 1000
        record("2. SendPacket", accepted,
               f"{ms:.0f}ms, utf8={len(text.encode())}B compressed={len(blob)}B packet={len(packet)}B"
               + ("" if accepted else " (서버가 거부: 서버 로그 확인)"))
    except grpc.RpcError as e:
        record("2. SendPacket", False, f"{e.code()} {e.details()}")

    # 3. 저장 확인 (원문 복원)
    message_id = None
    try:
        resp = stub.GetPendingMessages(pb.UserIdentifier(user_id=str(args.receiver)), timeout=args.timeout)
        found = find_items(resp, text)
        if found:
            message_id = found[-1].message_id
            record("3. 저장 확인 (원문 복원)", True, f"message_id={message_id}")
        else:
            record("3. 저장 확인 (원문 복원)", False,
                   "수신자 목록에서 원문을 찾지 못함 (복원 실패 또는 목록 조건 차이: 서버 DB 직접 확인)")
    except grpc.RpcError as e:
        record("3. 저장 확인 (원문 복원)", False, f"{e.code()} {e.details()}")

    # 4. GetVoice
    if args.voice:
        if not message_id:
            record("4. GetVoice", False, "3단계에서 message_id를 얻지 못해 건너뜀")
        else:
            try:
                t0 = time.perf_counter()
                first_ms, frames, chunks = None, 0, []
                for frame in stub.GetVoice(pb.MessageIdentifier(message_id=message_id), timeout=args.voice_timeout):
                    if first_ms is None and frame.audio_content:
                        first_ms = (time.perf_counter() - t0) * 1000
                    frames += 1
                    chunks.append(frame.audio_content)
                    if frame.is_final:
                        break
                body = b"".join(chunks)
                is_wav = body[:4] == b"RIFF" and body[8:12] == b"WAVE"
                total_ms = (time.perf_counter() - t0) * 1000
                first = f"{first_ms:.0f}ms" if first_ms is not None else "-"
                record("4. GetVoice", is_wav and len(body) > 44,
                       f"frames={frames} bytes={len(body):,} 첫 데이터={first} 전체={total_ms:.0f}ms WAV헤더={is_wav}")
                if args.save_wav and body:
                    with open(args.save_wav, "wb") as f:
                        f.write(body)
                    print(f"      saved: {os.path.abspath(args.save_wav)}")
            except grpc.RpcError as e:
                record("4. GetVoice", False, f"{e.code()} {e.details()}")

    close_connection()

    # 5. 거부되어야 하는 접속
    if args.negative:
        if args.plain:
            print("[SKIP] 5. 거부 확인 - 평문 모드에서는 의미가 없어 건너뜀")
        else:
            ch = grpc.insecure_channel(addr)
            try:
                SpeechRelayStub(ch).GetPendingMessages(pb.UserIdentifier(user_id=str(args.sender)), timeout=5)
                record("5a. 평문 접속 거부", False, "평문 요청이 성공함: 서버가 평문 포트로 떠 있음 (SERV_TLS 확인)")
            except grpc.RpcError as e:
                record("5a. 평문 접속 거부", True, f"{e.code()}")
            finally:
                ch.close()

            if args.wrong_ca:
                with open(args.wrong_ca, "rb") as f:
                    creds = grpc.ssl_channel_credentials(root_certificates=f.read())
                ch = grpc.secure_channel(addr, creds)
                try:
                    SpeechRelayStub(ch).GetPendingMessages(pb.UserIdentifier(user_id=str(args.sender)), timeout=5)
                    record("5b. 잘못된 CA 거부", False, "다른 CA로도 접속됨: 인증서 체인 확인 필요")
                except grpc.RpcError as e:
                    record("5b. 잘못된 CA 거부", True, f"{e.code()}")
                finally:
                    ch.close()

    return summary()


def summary() -> int:
    failed = [s for s, ok, _ in results if not ok]
    print("-" * 60)
    print(f"결과: {len(results) - len(failed)}/{len(results)} 통과" + (f", 실패: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
