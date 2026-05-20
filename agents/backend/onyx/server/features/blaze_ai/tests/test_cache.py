from __future__ import annotations

from agents.backend.onyx.server.features.blaze_ai.utils.cache import LRUCache


def test_lru_cache_evicts_oldest():
    cache: LRUCache[str, int] = LRUCache(capacity=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)  # should evict "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


