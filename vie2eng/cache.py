"""A bounded, thread-safe cache shared by translation requests."""

from collections import OrderedDict
from threading import Lock

from .config import CACHE_CAPACITY


class TranslationCache:
    def __init__(self, capacity=CACHE_CAPACITY):
        if capacity < 1:
            raise ValueError("Cache capacity must be positive")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()

    def __len__(self):
        with self._lock:
            return len(self._entries)
