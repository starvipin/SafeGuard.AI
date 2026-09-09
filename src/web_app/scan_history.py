# Recent scan results RAM mein rehte hain; process restart par history clear ho jati hai.
"""Bounded in-memory analysis history."""

from collections import deque
from threading import Lock


# Lock concurrent requests ke dauran history read/write ko ek-ek karke hone deta hai.
class AnalysisHistory:
    """Thread-safe recent-result store for the single-process web app."""

    # deque ki length limit bharne par sabse purana result automatically nikalta hai.
    def __init__(self, max_items: int = 10) -> None:
        self._items: deque[dict] = deque(maxlen=max(1, max_items))
        self._lock = Lock()

    # Naya result left side par rakho, isliye latest result list mein sabse pehle dikhega.
    def add(self, item: dict) -> None:
        with self._lock:
            self._items.appendleft(item)

    # Isi app process ki poori recent history khaali karo.
    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    # Internal deque ki list copy do, taaki caller seedhe internal collection na badle.
    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._items)
