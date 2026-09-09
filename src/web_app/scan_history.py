"""Bounded in-memory analysis history."""

from collections import deque
from threading import Lock


class AnalysisHistory:
    """Thread-safe recent-result store for the single-process web app."""

    def __init__(self, max_items: int = 10) -> None:
        self._items: deque[dict] = deque(maxlen=max(1, max_items))
        self._lock = Lock()

    def add(self, item: dict) -> None:
        with self._lock:
            self._items.appendleft(item)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._items)
