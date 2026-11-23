"""MessageBroker와 SpeechRelayServicer의 실시간 전달 단위 테스트 (DB, TTS 없이 실행)."""
import threading
from datetime import datetime

import pytest

from grpc_getting_started import server_communicate_pb2 as pb
from grpc_getting_started.message_broker import SUBSCRIBED_METADATA, MessageBroker
from grpc_getting_started.server_communicate_server import SpeechRelayServicer
from src.utils.bitpacking import encode_packet
from src.utils.text_compress import compress_text


class FakeContext:
    """grpc.ServicerContext 대신 쓰는 최소 구현"""

    def __init__(self):
        self.active = True
        self.initial_metadata = None

    def is_active(self):
        return self.active

    def send_initial_metadata(self, metadata):
        self.initial_metadata = metadata

    def abort(self, code, details):
        raise RuntimeError(f"{code}: {details}")


class FakeHandler:
    """FileSpeechHandler 대신 메모리에 저장하는 핸들러"""

    def __init__(self, fail=False):
        self.fail = fail
        self.rows = {}

    def save_incoming_speech(self, sender_id, receiver_id, message, emo_type, emotion_vector):
        if self.fail:
            return None
        message_id = f"m{len(self.rows)}"
        self.rows[message_id] = {
            "message_id": message_id, "sender_id": str(sender_id), "receiver_id": str(receiver_id),
            "message": message, "emo_type": emo_type, "send_time": datetime(2026, 11, 23, 12, 0, 0),
        }
        return message_id

    def get_message_metadata(self, message_id):
        return self.rows.get(message_id)


def packet(sender, receiver, text="안녕"):
    return pb.SpeechUploadPacketRequest(packet=encode_packet(
        send_id=sender, receiver_id=receiver, emo_type=1,
        emotion_indices=[0, 1, 2, 3, 4, 5, 6, 7], message=compress_text(text)))


class TestMessageBroker:
    def test_publish_reaches_every_subscriber_of_user(self):
        broker = MessageBroker()
        q1, q2 = broker.subscribe("0"), broker.subscribe("0")
        other = broker.subscribe("1")
        assert broker.publish("0", "item") == 2
        assert q1.get_nowait() == "item" and q2.get_nowait() == "item"
        assert other.empty()

    def test_publish_without_subscribers(self):
        assert MessageBroker().publish("3", "item") == 0

    def test_unsubscribe_removes_queue(self):
        broker = MessageBroker()
        q = broker.subscribe("0")
        broker.unsubscribe("0", q)
        assert broker.subscriber_count("0") == 0
        assert broker.publish("0", "item") == 0
        broker.unsubscribe("0", q)  # 두 번 해제해도 오류 없음

    def test_full_queue_is_skipped(self):
        broker = MessageBroker(maxsize=1)
        q = broker.subscribe("0")
        assert broker.publish("0", "a") == 1
        assert broker.publish("0", "b") == 0  # 가득 차면 건너뜀
        assert q.get_nowait() == "a"

    def test_int_and_str_user_id_are_same(self):
        broker = MessageBroker()
        q = broker.subscribe(0)
        assert broker.publish("0", "item") == 1
        assert q.get_nowait() == "item"

    def test_stream_sends_metadata_and_unsubscribes_on_close(self):
        broker = MessageBroker(poll_interval=0.05)
        ctx = FakeContext()
        got = []

        def consume():
            for item in broker.stream("0", ctx):
                got.append(item)

        t = threading.Thread(target=consume)
        t.start()
        for _ in range(100):  # 구독 등록 대기
            if broker.subscriber_count("0"):
                break
            threading.Event().wait(0.01)
        assert ctx.initial_metadata == SUBSCRIBED_METADATA
        broker.publish("0", "hello")
        for _ in range(100):
            if got:
                break
            threading.Event().wait(0.01)
        ctx.active = False  # 클라이언트 연결 종료
        t.join(timeout=2)
        assert got == ["hello"]
        assert not t.is_alive()
        assert broker.subscriber_count("0") == 0


class TestServicerNotify:
    def test_send_packet_notifies_receiver_and_sender(self):
        broker = MessageBroker()
        servicer = SpeechRelayServicer(FakeHandler(), broker)
        receiver_q, sender_q = broker.subscribe("0"), broker.subscribe("2")
        status = servicer.SendPacket(packet(2, 0, "2번이 0번에게"), FakeContext())
        assert status.accepted
        for q in (receiver_q, sender_q):
            item = q.get_nowait()
            assert (item.sender_id, item.receiver_id, item.message) == ("2", "0", "2번이 0번에게")
            assert item.message_id == "m0"
            assert item.send_time.ToDatetime() == datetime(2026, 11, 23, 12, 0, 0)

    def test_failed_save_does_not_notify(self):
        broker = MessageBroker()
        servicer = SpeechRelayServicer(FakeHandler(fail=True), broker)
        q = broker.subscribe("0")
        assert not servicer.SendPacket(packet(2, 0), FakeContext()).accepted
        assert q.empty()

    def test_notify_error_does_not_fail_send(self):
        handler = FakeHandler()
        handler.get_message_metadata = lambda message_id: (_ for _ in ()).throw(RuntimeError("db down"))
        servicer = SpeechRelayServicer(handler, MessageBroker())
        assert servicer.SendPacket(packet(2, 0), FakeContext()).accepted

    def test_subscribe_requires_user_id(self):
        servicer = SpeechRelayServicer(FakeHandler(), MessageBroker())
        with pytest.raises(RuntimeError):
            next(servicer.SubscribeMessages(pb.UserIdentifier(user_id=""), FakeContext()))
