"""
Basic smoke tests for Suno Clone AI API
"""
import pytest
import httpx
import os

API_URL = os.getenv("API_URL", "http://localhost:8020")
TIMEOUT = 10.0


@pytest.fixture
def client():
    """Create HTTP client"""
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


def test_health_endpoint(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_api_root(client):
    """Test API root endpoint"""
    response = client.get("/")
    assert response.status_code in [200, 404]  # May redirect or 404


def test_docs_endpoint(client):
    """Test API documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test OpenAPI schema endpoint"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data or "swagger" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint (may require auth)"""
    response = client.get("/metrics")
    # May return 200 or 401/403
    assert response.status_code in [200, 401, 403]


def test_response_time(client):
    """Test that response time is acceptable"""
    import time
    
    start = time.time()
    response = client.get("/health")
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 2.0, f"Response time too slow: {elapsed}s"




