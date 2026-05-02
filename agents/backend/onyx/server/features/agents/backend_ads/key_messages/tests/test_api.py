import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
import sys
import os
import asyncio
import aiohttp

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from key_messages.api import router
from key_messages.models import KeyMessageRequest, MessageType, MessageTone

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_generate_response(client: TestClient):
    """Test message generation endpoint."""
    request_data = {
        "message": "Test message",
        "message_type": MessageType.MARKETING,
        "tone": MessageTone.PROFESSIONAL,
        "target_audience": "Business professionals",
        "keywords": ["Point 1", "Point 2"]
    }
    
    response = client.post("/key-messages/generate", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"] is not None
    assert "original_message" in data["data"]
    assert "processing_time" in data
    assert "success" in data

def test_analyze_message(client: TestClient):
    """Test message analysis endpoint."""
    request_data = {
        "message": "Test message for analysis",
        "message_type": MessageType.INFORMATIONAL,
        "tone": MessageTone.CASUAL,
        "target_audience": "General audience",
        "keywords": ["Analysis point 1"]
    }
    
    response = client.post("/key-messages/analyze", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"] is not None
    assert "original_message" in data["data"]
    assert "processing_time" in data
    assert "success" in data

def test_get_message_types(client: TestClient):
    """Test getting message types."""
    response = client.get("/key-messages/types")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(item, str) for item in data)

def test_get_message_tones(client: TestClient):
    """Test getting message tones."""
    response = client.get("/key-messages/tones")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(item, str) for item in data)

def test_clear_cache(client: TestClient):
    """Test cache clearing endpoint."""
    response = client.delete("/key-messages/cache")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "message" in data

def test_invalid_request(client: TestClient):
    """Test handling of invalid requests."""
    response = client.post("/key-messages/generate", json={})
    assert response.status_code == 422

def test_response_compression(client: TestClient):
    """Test response compression (cache hit)."""
    import pytest
    pytest.skip("GZipMiddleware cannot be reliably tested with FastAPI TestClient. Run as integration test against live server.")
    request_data = {
        "message": "Test message",
        "message_type": MessageType.MARKETING,
        "tone": MessageTone.PROFESSIONAL,
        "target_audience": "Test audience",
        "keywords": ["Test point"]
    }
    # First request (cache miss)
    response1 = client.post(
        "/key-messages/generate",
        json=request_data,
        headers={"Accept-Encoding": "gzip"}
    )
    assert response1.status_code == 200
    # Second request (should be cache hit and compressed)
    response2 = client.post(
        "/key-messages/generate",
        json=request_data,
        headers={"Accept-Encoding": "gzip"}
    )
    assert response2.status_code == 200
    assert "Content-Encoding" in response2.headers

def test_cache_headers(client: TestClient):
    """Test cache headers."""
    # First request
    response = client.get("/key-messages/types")
    assert response.status_code == 200
    
    # Second request (should be cached)
    response = client.get("/key-messages/types")
    assert response.status_code == 200
    assert response.headers.get("X-Cache") == "HIT"

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent requests handling."""
    async def make_request(session, request_data):
        async with session.post(
            "http://localhost:8000/key-messages/generate",
            json=request_data
        ) as response:
            return await response.json()
    
    request_data = {
        "message": "Concurrent test message",
        "message_type": MessageType.MARKETING,
        "tone": MessageTone.PROFESSIONAL,
        "target_audience": "Test audience",
        "keywords": ["Test point"]
    }
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, request_data) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            assert "response" in result
            assert "processing_time" in result
            assert "cache_status" in result 