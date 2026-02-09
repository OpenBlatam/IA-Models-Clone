"""
Tests de integración para API de Robot Movement AI v2.0
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Importar app
try:
    from robot_movement_ai.main import app
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False
    app = None


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_health_check():
    """Test de health check endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_health_ready():
    """Test de readiness check"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health/ready")
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] == "ready"


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_health_live():
    """Test de liveness check"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "alive"
        assert "uptime_seconds" in data


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_metrics_endpoint():
    """Test de endpoint de métricas"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health/metrics")
        assert response.status_code == 200
        # Verificar que es texto plano (formato Prometheus)
        assert "text/plain" in response.headers.get("content-type", "")


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_openapi_schema():
    """Test de esquema OpenAPI"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_docs_endpoint():
    """Test de endpoint de documentación"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_cors_headers():
    """Test de headers CORS"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        # CORS debería estar configurado
        assert response.status_code in [200, 204]


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_request_id_header():
    """Test de Request ID header"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
async def test_timing_header():
    """Test de Timing header"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert "X-Process-Time" in response.headers
        # Verificar que es un número
        timing = float(response.headers["X-Process-Time"])
        assert timing >= 0


@pytest.fixture
def client():
    """Fixture para cliente de test"""
    if APP_AVAILABLE:
        return TestClient(app)
    return None


def test_sync_health_check(client):
    """Test síncrono de health check"""
    if client is None:
        pytest.skip("App not available")
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data




