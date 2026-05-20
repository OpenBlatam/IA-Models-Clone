import pytest
from fastapi.testclient import TestClient
from ..api import router
from ..models import AdditionalContentRequest

@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

def test_generate_additional_content(client: TestClient):
    """Test generating additional content."""
    request_data = {
        "text": "This is a test post about AI and machine learning",
        "platform": "twitter",
        "content_type": "tweet",
        "tone": "professional",
        "max_hashtags": 3,
        "include_cta": True
    }
    
    response = client.post("/additional-content/generate", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "hashtags" in data
    assert "call_to_action" in data
    assert "suggested_links" in data
    assert "full_text" in data
    assert "metadata" in data

def test_get_platforms(client: TestClient):
    """Test getting supported platforms."""
    response = client.get("/additional-content/platforms")
    assert response.status_code == 200
    data = response.json()
    assert "platforms" in data
    assert isinstance(data["platforms"], list)
    assert len(data["platforms"]) > 0

def test_get_content_types(client: TestClient):
    """Test getting supported content types."""
    response = client.get("/additional-content/content-types")
    assert response.status_code == 200
    data = response.json()
    assert "content_types" in data
    assert isinstance(data["content_types"], list)
    assert len(data["content_types"]) > 0

def test_get_cta_types(client: TestClient):
    """Test getting supported CTA types."""
    response = client.get("/additional-content/cta-types")
    assert response.status_code == 200
    data = response.json()
    assert "cta_types" in data
    assert isinstance(data["cta_types"], list)
    assert len(data["cta_types"]) > 0

def test_clear_cache(client: TestClient):
    """Test clearing the cache."""
    response = client.delete("/additional-content/cache")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Cache cleared successfully" 