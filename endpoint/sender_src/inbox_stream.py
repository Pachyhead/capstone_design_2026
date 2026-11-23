"""서버의 새 메시지 구독(SubscribeMessages)을 받아 브라우저(SSE /events)로 넘긴다.

InboxSubscriber: 백그라운드 스레드에서 현재 사용자로 구독하고, 끊기면 다시 접속한다.
Broadcaster: 접속 중인 브라우저 탭(SSE 연결)마다 큐를 두고 이벤트를 나눠 준다.

브라우저로 보내는 이벤트
  message: 새 메시지 1건 (/set_my_id 목록 항목과 같은 형식)
  resync:  구독이 (다시) 등록됨. 끊겨 있던 동안의 메시지를 위해 목록을 다시 받아야 함
"""

import asyncio
import logging
import threading
from typing import Any

import grpc
from fastapi.encoders import jsonable_encoder

import config  # noqa: F401 - 저장소 루트를 sys.path에 추가
from grpc_getting_started.server_communicate_receiver import SubscribeMessages, item_to_dict

logger = logging.getLogger(__name__)


class Broadcaster:
    """SSE 연결마다 asyncio.Queue를 둔다. publish()는 어느 스레드에서든 호출할 수 있다."""

    def __init__(self, maxsize: int = 100) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.Queue] = set()
        self._maxsize = maxsize

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def connect(self) -> asyncio.Queue:
        """SSE 연결 하나를 등록한다. 이벤트 루프 안에서 호출한다."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._clients.add(q)
        return q

    def disconnect(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def client_count(self) -> int:
        return len(self._clients)

    def publish(self, event: str, data: Any = None) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._fanout, event, data)

    def _fanout(self, event: str, data: Any) -> None:
        for q in list(self._clients):
            try:
                q.put_nowait((event, data))
            except asyncio.QueueFull:
                # 오래 읽지 않은 탭: 쌓인 이벤트를 버리고 목록을 다시 받게 한다
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(("resync", None))


class InboxSubscriber:
    """현재 사용자의 SubscribeMessages 스트림을 유지하고, 받은 메시지를 Broadcaster로 넘긴다."""

    RETRY_MIN = 1.0   # 재접속 대기(초), 실패할 때마다 두 배
    RETRY_MAX = 10.0

    def __init__(self, broadcaster: Broadcaster) -> None:
        self._broadcaster = broadcaster
        self._lock = threading.Lock()
        self._user_id: int | None = None
        self._call = None
        self._connected = False
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="inbox-subscriber", daemon=True)

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        self._thread.start()

    def switch_user(self, user_id: int) -> None:
        """구독할 사용자를 바꾼다 (/set_my_id). 같은 사용자면 기존 구독을 유지한다."""
        with self._lock:
            if self._user_id == user_id:
                return
            self._user_id = user_id
            call = self._call
        if call is not None:
            call.cancel()
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            call = self._call
        if call is not None:
            call.cancel()
        self._wake.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        delay = self.RETRY_MIN
        while not self._stop.is_set():
            with self._lock:
                user_id = self._user_id
            if user_id is None:  # /set_my_id 전에는 구독하지 않는다
                self._wake.wait()
                self._wake.clear()
                continue
            self._wake.clear()

            call = SubscribeMessages(str(user_id))
            with self._lock:
                stale = self._stop.is_set() or self._user_id != user_id
                if not stale:
                    self._call = call
            if stale:  # 여는 사이 사용자가 바뀌었거나 종료됨
                call.cancel()
                continue

            try:
                metadata = dict(call.initial_metadata() or ())  # 구독 등록 신호까지 대기
                if metadata.get("subscribed") == "1":
                    self._connected = True
                    delay = self.RETRY_MIN
                    logger.info("inbox subscribed: user %s", user_id)
                    self._broadcaster.publish("resync", {"user_id": user_id})
                for item in call:
                    self._broadcaster.publish("message", jsonable_encoder(item_to_dict(item)))
                logger.warning("inbox stream ended by server: user %s", user_id)
            except grpc.RpcError as e:
                switched = self._stop.is_set() or self._user_id != user_id
                if e.code() == grpc.StatusCode.CANCELLED and switched:
                    continue  # 사용자 전환 또는 종료로 직접 끊은 경우: 바로 다음 구독으로
                logger.warning("inbox stream error (user %s): %s %s", user_id, e.code(), e.details())
            finally:
                self._connected = False
                with self._lock:
                    if self._call is call:
                        self._call = None

            # 재접속 대기. 사용자 전환이나 종료가 있으면 바로 깬다
            if self._wake.wait(delay):
                self._wake.clear()
            delay = min(delay * 2, self.RETRY_MAX)
