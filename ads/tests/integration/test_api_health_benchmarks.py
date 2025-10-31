from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.backend.onyx.server.features.ads.api import router as ads_router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(ads_router)
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/ads/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data.get("routers"), list)
    assert "/core" in data["routers"]


def test_capabilities_endpoint(client: TestClient) -> None:
    r = client.get("/ads/capabilities")
    assert r.status_code == 200
    data = r.json()
    assert data["feature"] == "ads"
    assert isinstance(data.get("capabilities"), dict)
    for key in ["domain", "application", "infrastructure", "optimization", "training", "api"]:
        assert key in data["capabilities"]


def test_benchmarks_endpoint(client: TestClient) -> None:
    r = client.get("/ads/benchmarks")
    assert r.status_code == 200
    data = r.json()
    assert "json_encode_ms" in data
    assert isinstance(data["json_encode_ms"], (int, float))


def test_benchmarks_deep_endpoint(client: TestClient) -> None:
    r = client.get("/ads/benchmarks/deep")
    assert r.status_code == 200
    data: Dict[str, Any] = r.json()
    assert "timings" in data and isinstance(data["timings"], dict)
    assert "errors" in data and isinstance(data["errors"], dict)


def test_services_health_endpoint(client: TestClient) -> None:
    r = client.get("/ads/health/services")
    assert r.status_code == 200
    data = r.json()
    assert "services" in data
    assert "total" in data







