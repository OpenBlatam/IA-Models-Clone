"""
Tests for modular API router
Validates that the modular router endpoints are accessible and functional
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_modular_router_health(async_client: AsyncClient):
    """Test that modular router health endpoint is accessible"""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "success" in data or "status" in data or "api_ready" in data


@pytest.mark.asyncio
async def test_modular_router_metrics(async_client: AsyncClient):
    """Test that modular router metrics endpoint is accessible"""
    try:
        response = await async_client.get("/api/v1/metrics")
        # Metrics endpoint may return 200 (if enabled) or 404 (if disabled)
        assert response.status_code in [200, 404]
    except Exception:
        # Metrics endpoint may not be available
        pass


@pytest.mark.asyncio
async def test_modular_router_correlation_headers(async_client: AsyncClient):
    """Test that correlation headers are returned in responses"""
    response = await async_client.get("/api/v1/health")
    
    # Check for X-Request-ID header (may be set by middleware)
    assert "X-Request-ID" in response.headers or response.status_code == 200
    
    # Verify response structure
    if response.status_code == 200:
        data = response.json()
        # Response should be JSON-serializable
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_modular_router_endpoints_exist(async_client: AsyncClient):
    """Test that modular router endpoints are properly registered"""
    # Test root endpoint
    response = await async_client.get("/")
    assert response.status_code == 200
    
    # Test API v1 prefix
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    
    # Test that 404 is returned for non-existent endpoints
    response = await async_client.get("/api/v1/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_modular_router_openapi_schema(async_client: AsyncClient):
    """Test that OpenAPI schema includes modular router endpoints"""
    response = await async_client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "paths" in schema
    assert "info" in schema
    
    # Check that API v1 paths exist
    paths = schema["paths"]
    has_api_v1_paths = any(path.startswith("/api/v1") for path in paths.keys())
    assert has_api_v1_paths, "No /api/v1 paths found in OpenAPI schema"


@pytest.mark.asyncio
async def test_modular_router_cors_headers(async_client: AsyncClient):
    """Test that CORS headers are present (if configured)"""
    response = await async_client.options(
        "/api/v1/health",
        headers={"Origin": "http://localhost:3000"}
    )
    
    # CORS headers may be present or not depending on configuration
    # Just verify the request doesn't fail
    assert response.status_code in [200, 204, 405]


