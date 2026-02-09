from __future__ import annotations

import time
from collections import OrderedDict
from typing import Generic, Optional, Tuple, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    def __init__(self, capacity: int, ttl_seconds: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._store: "OrderedDict[K, Tuple[V, float]]" = OrderedDict()
        self._capacity = int(capacity)
        self._ttl = float(ttl_seconds)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired_keys: list[K] = []
        for key, (_, exp) in self._store.items():
            if exp < now:
                expired_keys.append(key)
            else:
                break  # order preserved; newer at end
        for k in expired_keys:
            self._store.pop(k, None)

    def get(self, key: K) -> Optional[V]:
        self._purge_expired()
        item = self._store.get(key)
        if item is None:
            return None
        value, exp = item
        if exp < time.monotonic():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        self._purge_expired()
        self._store[key] = (value, time.monotonic() + self._ttl)
        self._store.move_to_end(key)
        if len(self._store) > self._capacity:
            self._store.popitem(last=False)


