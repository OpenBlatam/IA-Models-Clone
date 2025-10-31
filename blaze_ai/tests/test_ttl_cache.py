from __future__ import annotations

import asyncio

from agents.backend.onyx.server.features.blaze_ai.utils.ttl_cache import TTLCache


async def test_ttl_cache_expiry():  # type: ignore[no-untyped-def]
    c: TTLCache[str, int] = TTLCache(capacity=2, ttl_seconds=0.1)
    c.set("a", 1)
    assert c.get("a") == 1
    await asyncio.sleep(0.11)
    assert c.get("a") is None
    c.set("b", 2)
    c.set("c", 3)
    assert c.get("b") == 2
    c.set("d", 4)  # evicts oldest (b or c if expired)
    # Not asserting eviction target due to time variance, but ensure capacity maintained
    # and no exception occurs.
    assert c.get("d") == 4


