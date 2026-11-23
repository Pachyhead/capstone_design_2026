"""새 메시지를 구독 중인 클라이언트에게 전달하는 구독자 관리.

사용자 ID마다 구독 큐 집합을 둔다. 같은 사용자가 여러 기기/탭에서 접속하면 큐가 여러 개가 된다.
상태는 서버 프로세스 메모리에만 있으므로 gRPC 서버가 하나일 때만 동작한다.
"""

import queue
import threading
from collections.abc import Iterator
from typing import Any

# 구독이 등록되었음을 알리는 initial metadata. 클라이언트는 이를 받은 뒤 목록을 다시 받아(resync)
# 구독 직전에 저장된 메시지를 놓치지 않는다.
SUBSCRIBED_METADATA = (("subscribed", "1"),)


class MessageBroker:
    def __init__(self, maxsize: int = 100, poll_interval: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[queue.Queue]] = {}
        self._maxsize = maxsize
        self._poll_interval = poll_interval  # 연결 종료를 확인하는 주기(초)

    def subscribe(self, user_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.setdefault(str(user_id), set()).add(q)
        return q

    def unsubscribe(self, user_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(str(user_id))
            if subs is None:
                return
            subs.discard(q)
            if not subs:
                del self._subscribers[str(user_id)]

    def publish(self, user_id: str, item: Any) -> int:
        """user_id의 모든 구독 큐에 item을 넣고, 전달한 큐 수를 반환한다.

        큐가 가득 찬(오래 읽지 않은) 구독자는 건너뛴다. 놓친 메시지는 클라이언트가 재접속할 때
        GetPendingMessages로 다시 맞춘다.
        """
        with self._lock:
            targets = list(self._subscribers.get(str(user_id), ()))
        delivered = 0
        for q in targets:
            try:
                q.put_nowait(item)
                delivered += 1
            except queue.Full:
                pass
        return delivered

    def subscriber_count(self, user_id: str) -> int:
        with self._lock:
            return len(self._subscribers.get(str(user_id), ()))

    def stream(self, user_id: str, context) -> Iterator[Any]:
        """서버 스트리밍 RPC 본문. 연결이 끊기면 구독을 해제하고 끝난다.

        context는 grpc.ServicerContext (is_active, send_initial_metadata를 사용).
        """
        q = self.subscribe(user_id)
        try:
            context.send_initial_metadata(SUBSCRIBED_METADATA)
            while context.is_active():
                try:
                    yield q.get(timeout=self._poll_interval)
                except queue.Empty:
                    continue
        finally:
            self.unsubscribe(user_id, q)
