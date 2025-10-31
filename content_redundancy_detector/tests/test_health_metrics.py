import asyncio
import pytest

try:
    from httpx import AsyncClient
except Exception:  # pragma: no cover
    AsyncClient = None  # type: ignore

from ..api.main import create_app


pytestmark = pytest.mark.asyncio


async def _get_client():
    app = create_app()
    if AsyncClient is None:
        pytest.skip("httpx not installed")
    return AsyncClient(app=app, base_url="http://test")


async def test_prometheus_metrics_endpoint_cache_control():
    async with await _get_client() as client:
        resp = await client.get("/api/v1/metrics/prometheus")
        assert resp.headers.get("cache-control") is not None
        assert "no-store" in resp.headers.get("cache-control", "").lower()
        assert resp.status_code in (200, 503)
        # Content type should be text/plain for Prometheus
        assert "text/plain" in resp.headers.get("content-type", "")


async def test_health_endpoints_no_store():
    async with await _get_client() as client:
        # /health
        h = await client.get("/api/v1/health/")
        assert h.status_code == 200
        assert "no-store" in h.headers.get("cache-control", "").lower()
        data = h.json()
        assert data.get("success") is True

        # /ready may be 200 or 503 depending on init, but must be no-store
        r = await client.get("/api/v1/health/ready")
        assert r.status_code in (200, 503)
        assert "no-store" in r.headers.get("cache-control", "").lower()

        # /live must be 200 and no-store
        l = await client.get("/api/v1/health/live")
        assert l.status_code == 200
        assert "no-store" in l.headers.get("cache-control", "").lower()


async def test_metrics_json_routes():
    async with await _get_client() as client:
        m = await client.get("/api/v1/metrics")
        assert m.status_code == 200
        payload = m.json()
        assert "success" in payload and "data" in payload

        mw = await client.get("/api/v1/metrics/webhooks")
        assert mw.status_code == 200
        payload_w = mw.json()
        assert payload_w.get("success") is True
        assert "data" in payload_w







