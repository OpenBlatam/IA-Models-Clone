import pytest


@pytest.mark.asyncio
async def test_analyze_endpoint(async_client):
    payload = {
        "content": "Texto de ejemplo para análisis de redundancia",
        "analysis_type": "redundancy",
        "language": "es",
        "threshold": 0.7
    }
    resp = await async_client.post("/api/v1/analyze", json=payload)
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert data.get("success") is True
    assert "data" in data


@pytest.mark.asyncio
async def test_similarity_endpoint(async_client):
    payload = {
        "text1": "El rápido zorro marrón salta",
        "text2": "Un zorro veloz de color marrón brinca",
        "algorithm": "cosine",
        "threshold": 0.6
    }
    resp = await async_client.post("/api/v1/similarity", json=payload)
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert data.get("success") is True
    assert "data" in data


@pytest.mark.asyncio
async def test_quality_endpoint(async_client):
    payload = {
        "content": "Artículo bien escrito con gramática y estructura clara.",
        "quality_metrics": ["readability", "grammar"],
        "target_audience": "general"
    }
    resp = await async_client.post("/api/v1/quality", json=payload)
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert data.get("success") is True
    assert "data" in data







