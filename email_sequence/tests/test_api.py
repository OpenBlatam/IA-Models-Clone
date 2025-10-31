"""
API Tests for Email Sequence System

This module contains tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client: TestClient):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestSequenceEndpoints:
    """Test sequence management endpoints."""
    
    def test_create_sequence(self, client: TestClient, sample_sequence_data):
        """Test sequence creation."""
        response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_sequence_data["name"]
        assert "id" in data
        assert data["status"] == "draft"
    
    def test_get_sequence(self, client: TestClient, sample_sequence_data):
        """Test getting a sequence."""
        # Create sequence first
        create_response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        sequence_id = create_response.json()["id"]
        
        # Get sequence
        response = client.get(f"/api/v1/email-sequences/{sequence_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sequence_id
        assert data["name"] == sample_sequence_data["name"]
    
    def test_list_sequences(self, client: TestClient, sample_sequence_data):
        """Test listing sequences."""
        # Create a sequence
        client.post("/api/v1/email-sequences", json=sample_sequence_data)
        
        # List sequences
        response = client.get("/api/v1/email-sequences")
        assert response.status_code == 200
        data = response.json()
        assert "sequences" in data
        assert "total_count" in data
        assert len(data["sequences"]) >= 1
    
    def test_update_sequence(self, client: TestClient, sample_sequence_data):
        """Test updating a sequence."""
        # Create sequence
        create_response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        sequence_id = create_response.json()["id"]
        
        # Update sequence
        update_data = {"name": "Updated Sequence Name"}
        response = client.put(f"/api/v1/email-sequences/{sequence_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
    
    def test_delete_sequence(self, client: TestClient, sample_sequence_data):
        """Test deleting a sequence."""
        # Create sequence
        create_response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        sequence_id = create_response.json()["id"]
        
        # Delete sequence
        response = client.delete(f"/api/v1/email-sequences/{sequence_id}")
        assert response.status_code == 204
        
        # Verify deletion
        get_response = client.get(f"/api/v1/email-sequences/{sequence_id}")
        assert get_response.status_code == 404
    
    def test_activate_sequence(self, client: TestClient, sample_sequence_data):
        """Test activating a sequence."""
        # Create sequence
        create_response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        sequence_id = create_response.json()["id"]
        
        # Activate sequence
        response = client.post(f"/api/v1/email-sequences/{sequence_id}/activate")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"


class TestSubscriberEndpoints:
    """Test subscriber management endpoints."""
    
    def test_add_subscribers_to_sequence(self, client: TestClient, sample_sequence_data, sample_subscriber_data):
        """Test adding subscribers to a sequence."""
        # Create sequence
        create_response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        sequence_id = create_response.json()["id"]
        
        # Add subscribers
        bulk_data = {
            "subscribers": [sample_subscriber_data],
            "sequence_id": sequence_id
        }
        response = client.post(f"/api/v1/email-sequences/{sequence_id}/subscribers", json=bulk_data)
        assert response.status_code == 201
        data = response.json()
        assert data["total_items"] == 1
        assert data["successful_items"] == 1


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""
    
    def test_get_sequence_analytics(self, client: TestClient, sample_sequence_data):
        """Test getting sequence analytics."""
        # Create sequence
        create_response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        sequence_id = create_response.json()["id"]
        
        # Get analytics
        response = client.get(f"/api/v1/email-sequences/{sequence_id}/analytics")
        assert response.status_code == 200
        data = response.json()
        assert "sequence_id" in data
        assert "metrics" in data
        assert "time_range" in data


class TestErrorHandling:
    """Test error handling."""
    
    def test_sequence_not_found(self, client: TestClient):
        """Test 404 error for non-existent sequence."""
        response = client.get("/api/v1/email-sequences/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
    
    def test_invalid_sequence_data(self, client: TestClient):
        """Test validation error for invalid sequence data."""
        invalid_data = {
            "name": "",  # Empty name should fail validation
            "target_audience": "Test audience"
        }
        response = client.post("/api/v1/email-sequences", json=invalid_data)
        assert response.status_code == 422
    
    def test_invalid_subscriber_email(self, client: TestClient, sample_sequence_data):
        """Test validation error for invalid email."""
        # Create sequence
        create_response = client.post("/api/v1/email-sequences", json=sample_sequence_data)
        sequence_id = create_response.json()["id"]
        
        # Add subscriber with invalid email
        bulk_data = {
            "subscribers": [{"email": "invalid-email"}],
            "sequence_id": sequence_id
        }
        response = client.post(f"/api/v1/email-sequences/{sequence_id}/subscribers", json=bulk_data)
        assert response.status_code == 422


class TestAsyncEndpoints:
    """Test async endpoints."""
    
    @pytest.mark.asyncio
    async def test_async_health_check(self, async_client: AsyncClient):
        """Test async health check."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_async_sequence_creation(self, async_client: AsyncClient, sample_sequence_data):
        """Test async sequence creation."""
        response = await async_client.post("/api/v1/email-sequences", json=sample_sequence_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_sequence_data["name"]


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiting(self, client: TestClient):
        """Test rate limiting (if enabled)."""
        # Make multiple requests quickly
        responses = []
        for _ in range(70):  # Exceed default rate limit of 60/min
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should eventually get rate limited
        assert 429 in responses


class TestCORS:
    """Test CORS functionality."""
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are present."""
        response = client.options("/api/v1/email-sequences")
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers






























