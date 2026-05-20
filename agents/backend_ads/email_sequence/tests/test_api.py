import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from ..api import router
from ..models import (
    EmailSequenceRequest,
    EmailTemplate,
    BrandVoice,
    AudienceProfile,
    ProjectContext
)

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture
def sample_request_data():
    return {
        "type": "email-sequence",
        "prompt": "Test prompt",
        "target_audience": "Test audience",
        "goals": ["goal1", "goal2"],
        "brand_voice": {
            "tone": "professional",
            "style": "conversational"
        },
        "audience_profile": {
            "demographics": {
                "age_range": "25-35",
                "location": "US"
            },
            "customer_stage": "awareness"
        },
        "project_context": {
            "project_name": "Test Project",
            "project_description": "Test Description",
            "industry": "Technology"
        }
    }

@pytest.fixture
def sample_template_data():
    return {
        "subject": "Test Subject",
        "body": "Test Body",
        "delay_days": 1,
        "template_type": "welcome"
    }

def test_create_sequence(sample_request_data):
    response = client.post("/email-sequence/create", json=sample_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "sequence_id" in data
    assert data["status"] == "draft"

def test_get_sequence(sample_request_data):
    # First create a sequence
    create_response = client.post("/email-sequence/create", json=sample_request_data)
    sequence_id = create_response.json()["sequence_id"]
    
    # Then get it
    response = client.get(f"/email-sequence/{sequence_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_id"] == sequence_id

def test_get_nonexistent_sequence():
    response = client.get("/email-sequence/nonexistent-id")
    assert response.status_code == 404

def test_update_sequence_status(sample_request_data):
    # First create a sequence
    create_response = client.post("/email-sequence/create", json=sample_request_data)
    sequence_id = create_response.json()["sequence_id"]
    
    # Then update its status
    response = client.put(f"/email-sequence/{sequence_id}/status", params={"status": "active"})
    assert response.status_code == 200
    
    # Verify the update
    get_response = client.get(f"/email-sequence/{sequence_id}")
    assert get_response.json()["status"] == "active"

def test_add_template(sample_request_data, sample_template_data):
    # First create a sequence
    create_response = client.post("/email-sequence/create", json=sample_request_data)
    sequence_id = create_response.json()["sequence_id"]
    
    # Then add a template
    response = client.post(f"/email-sequence/{sequence_id}/templates", json=sample_template_data)
    assert response.status_code == 200
    
    # Verify the template was added
    get_response = client.get(f"/email-sequence/{sequence_id}")
    assert len(get_response.json()["templates"]) == 1

def test_list_sequences(sample_request_data):
    # Create multiple sequences
    client.post("/email-sequence/create", json=sample_request_data)
    client.post("/email-sequence/create", json=sample_request_data)
    
    # Test listing all sequences
    response = client.get("/email-sequence/list")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    
    # Test filtering
    response = client.get("/email-sequence/list", params={"status": "draft"})
    assert response.status_code == 200
    data = response.json()
    assert all(seq["status"] == "draft" for seq in data)

def test_delete_sequence(sample_request_data):
    # First create a sequence
    create_response = client.post("/email-sequence/create", json=sample_request_data)
    sequence_id = create_response.json()["sequence_id"]
    
    # Then delete it
    response = client.delete(f"/email-sequence/{sequence_id}")
    assert response.status_code == 200
    
    # Verify it's deleted
    get_response = client.get(f"/email-sequence/{sequence_id}")
    assert get_response.status_code == 404

def test_duplicate_sequence(sample_request_data, sample_template_data):
    # First create and populate a sequence
    create_response = client.post("/email-sequence/create", json=sample_request_data)
    sequence_id = create_response.json()["sequence_id"]
    client.post(f"/email-sequence/{sequence_id}/templates", json=sample_template_data)
    
    # Then duplicate it
    response = client.post(f"/email-sequence/{sequence_id}/duplicate")
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_id"] != sequence_id
    assert data["sequence_name"].startswith("Copy of")
    assert len(data["templates"]) == 1 