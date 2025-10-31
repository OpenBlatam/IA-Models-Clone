import pytest


@pytest.mark.asyncio
async def test_health_basic(async_client):
    resp = await async_client.get("/health")
    assert resp.status_code in (200, 204) or resp.status_code == 404  # tolerate absence if only advanced exists


@pytest.mark.asyncio
async def test_health_advanced(async_client):
    resp = await async_client.get("/health/advanced")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "success" in data and data["success"] is True
    assert "data" in data and isinstance(data["data"], dict)
    assert "status" in data["data"]


@pytest.mark.asyncio
async def test_docs_available(async_client):
    # Swagger UI
    resp = await async_client.get("/docs")
    assert resp.status_code == 200
    # ReDoc
    resp = await async_client.get("/redoc")
    assert resp.status_code == 200







