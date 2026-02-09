import pytest


@pytest.mark.asyncio
async def test_cache_stats(async_client):
    resp = await async_client.get("/cache/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True
    assert "data" in data and isinstance(data["data"], dict)
    assert "hit_rate" in data["data"]


@pytest.mark.asyncio
async def test_rate_limit_status(async_client):
    resp = await async_client.get("/rate-limit/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True
    assert "data" in data and isinstance(data["data"], dict)
    assert "windows" in data["data"]







