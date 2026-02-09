"""
Tests for health check endpoints
"""

import pytest
from fastapi import status


def test_root_endpoint(client):
    """Test root endpoint returns API information"""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["service"] == "Logistics AI Platform"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data


def test_readiness_check(client):
    """Test readiness check endpoint"""
    response = client.get("/ready")
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ready", "not_ready"]


def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get("content-type") is not None


def test_metrics_info_endpoint(client):
    """Test metrics info endpoint"""
    response = client.get("/metrics/info")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "enabled" in data
    assert "metrics" in data

