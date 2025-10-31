from typing import Any, Dict

import pytest

from agents.backend.onyx.server.features.ads.infrastructure.cache import (
    CacheManager,
    CacheConfig,
    CacheType,
)
from agents.backend.onyx.server.features.ads.infrastructure.external_services import (
    ExternalServiceManager,
)


@pytest.mark.asyncio
async def test_cache_manager_memory_set_get_roundtrip() -> None:
    # Use memory backend to avoid external dependencies
    cm = CacheManager(CacheConfig(cache_type=CacheType.MEMORY))
    key = cm.generate_key("unit", "cache", n=1)

    payload: Dict[str, Any] = {"a": 1, "b": [1, 2, 3], "ok": True}
    ok = await cm.set(key, payload, ttl=5)
    assert ok is True

    value = await cm.get(key)
    assert isinstance(value, dict)
    assert value == payload

    # Clean up
    deleted = await cm.strategy.delete(key)
    assert deleted is True


def test_external_service_health_snapshot_shape() -> None:
    mgr = ExternalServiceManager()
    snap = mgr.get_health_snapshot()
    assert "services" in snap
    assert "total" in snap
    assert isinstance(snap["services"], dict)







