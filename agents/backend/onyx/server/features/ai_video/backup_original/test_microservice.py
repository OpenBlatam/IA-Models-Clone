from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from fastapi.testclient import TestClient
from fastapi_microservice import app

    from fastapi_microservice import process_video_task
from typing import Any, List, Dict, Optional
import logging
import asyncio
client = TestClient(app)

TOKEN = "supersecrettoken"

@pytest.fixture
def auth_header():
    
    """auth_header function."""
return {"Authorization": f"Bearer {TOKEN}"}

def test_health():
    
    """test_health function."""
resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_video_post(auth_header, monkeypatch) -> Any:
    # Mock Celery task to avoid real async processing
    monkeypatch.setattr(process_video_task, "delay", lambda *a, **kw: None)
    payload = {"input_text": "test video", "user_id": "user1"}
    resp = client.post("/v1/video", json=payload, headers=auth_header)
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "processing"
    assert data["request_id"].startswith("req_")

def test_video_post_unauthorized():
    
    """test_video_post_unauthorized function."""
payload = {"input_text": "test video", "user_id": "user1"}
    resp = client.post("/v1/video", json=payload)
    assert resp.status_code == 401

def test_metrics():
    
    """test_metrics function."""
resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text 