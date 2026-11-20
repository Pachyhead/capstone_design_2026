"""Bounded producer bridge; independent of Torch/FastAPI for lifecycle tests."""
from queue import Queue, Full, Empty
from threading import Event, Thread


class AudioWorker:
    def __init__(self, produce, capacity=8):
        self.queue = Queue(maxsize=capacity)
        self.cancelled = Event()
        self.done = Event()
        self.error = None
        self.thread = Thread(target=self._run, args=(produce,), daemon=True)

    def _emit(self, item):
        if item is None:
            raise ValueError("None is reserved for end of stream")
        while not self.cancelled.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except Full:
                continue
        raise InterruptedError("Audio stream disconnected")

    def _run(self, produce):
        try:
            produce(self._emit, self.cancelled.is_set)
        except InterruptedError as exc:
            if not self.cancelled.is_set():
                self.error = exc
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    def start(self):
        self.thread.start()
        return self

    def read(self):
        while not self.cancelled.is_set():
            try:
                return self.queue.get(timeout=0.1)
            except Empty:
                if self.done.is_set():
                    if self.error is not None:
                        raise self.error
                    return None
        return None

    def close(self):
        self.cancelled.set()
