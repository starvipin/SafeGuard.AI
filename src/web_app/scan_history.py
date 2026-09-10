# Recent scans are stored in RAM and disappear when the application process restarts.
"""Bounded in-memory analysis history."""

from collections import deque
from threading import Lock


# A lock serializes concurrent reads and writes to the history store.
class AnalysisHistory:
    """Thread-safe recent-result store for the single-process web app."""

    # When the bounded deque fills up, adding a new result automatically removes the oldest one.
    def __init__(self, max_items: int = 10) -> None:
        self._items: deque[dict] = deque(maxlen=max(1, max_items))
        self._lock = Lock()

    # Add results on the left so the most recent scan appears first.
    def add(self, item: dict) -> None:
        with self._lock:
            self._items.appendleft(item)

    # Remove all recent results from this application's history store.
    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    # Return a list copy so callers cannot directly modify the internal deque.
    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._items)
