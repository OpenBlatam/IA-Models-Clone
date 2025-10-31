from typing import Any, List, Dict, Optional
import pytest
def test_generate_copywriting_variants(client):
    payload = {
        "product_description": "Zapatos deportivos de alta gama",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "target_audience": "Jóvenes activos",
        "key_points": ["Comodidad", "Estilo", "Durabilidad"],
        "instructions": "Enfatiza la innovación.",
        "restrictions": ["no mencionar precio"],
        "creativity_level": 0.8,
        "language": "es"
    }
    response = client.post("/copywriting/generate?model_name=gpt2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "variants" in data
    assert isinstance(data["variants"], list)
    assert len(data["variants"]) >= 1
    for variant in data["variants"]:
        assert "headline" in variant
        assert "primary_text" in variant
    assert "model_used" in data
    assert "generation_time" in data
    assert "extra_metadata" in data

def test_batch_generate_copywriting(client):
    payload = [
        {
            "product_description": "Zapatos deportivos de alta gama",
            "target_platform": "Instagram",
            "tone": "inspirational",
            "target_audience": "Jóvenes activos",
            "key_points": ["Comodidad", "Estilo", "Durabilidad"],
            "instructions": "Enfatiza la innovación.",
            "restrictions": ["no mencionar precio"],
            "creativity_level": 0.8,
            "language": "es"
        },
        {
            "product_description": "Reloj inteligente para fitness",
            "target_platform": "Facebook",
            "tone": "informative",
            "target_audience": "Adultos activos",
            "key_points": ["Monitor de ritmo cardíaco", "GPS", "Resistente al agua"],
            "instructions": "Enfatiza la tecnología.",
            "restrictions": ["no mencionar precio"],
            "creativity_level": 0.7,
            "language": "es"
        }
    ]
    response = client.post("/copywriting/batch-generate?wait=true", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert "variants" in result
    for result in data["results"]:
        for variant in result["variants"]:
            assert "headline" in variant
            assert "primary_text" in variant

def test_list_models(client):
    response = client.get("/copywriting/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert isinstance(data["available_models"], list)
    assert "gpt2" in data["available_models"]

def test_task_status(client, monkeypatch):
    # Simula un resultado de Celery
    class DummyResult:
        state = "SUCCESS"
        result = {"foo": "bar"}
        info = None
    monkeypatch.setattr("celery.result.AsyncResult", lambda task_id: DummyResult())
    response = client.get("/copywriting/task-status/dummy-task-id")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["result"] == {"foo": "bar"}

def test_task_status_failure(client, monkeypatch):
    class DummyResult:
        state = "FAILURE"
        result = None
        info = Exception("Test error")
    monkeypatch.setattr("celery.result.AsyncResult", lambda task_id: DummyResult())
    response = client.get("/copywriting/task-status/dummy-task-id")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FAILURE"
    assert "error" in data

def test_batch_task_status(client, monkeypatch):
    class DummyResult:
        def __init__(self, state, result=None, info=None):
            self.state = state
            self.result = result
            self.info = info
    # Simulate two tasks: one SUCCESS, one FAILURE
    dummy_results = {
        "task1": DummyResult("SUCCESS", {"foo": "bar"}),
        "task2": DummyResult("FAILURE", None, Exception("Test error")),
    }
    monkeypatch.setattr("celery.result.AsyncResult", lambda task_id: dummy_results[task_id])
    response = client.post("/copywriting/batch-status", json=["task1", "task2"])
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert len(data["tasks"]) == 2
    for task in data["tasks"]:
        assert "task_id" in task
        assert "status" in task
        if task["task_id"] == "task1":
            assert task["status"] == "SUCCESS"
            assert task["result"] == {"foo": "bar"}
            assert task["error"] is None
        elif task["task_id"] == "task2":
            assert task["status"] == "FAILURE"
            assert task["result"] is None
            assert "Test error" in task["error"]

def test_generate_invalid_model(client):
    payload = {
        "product_description": "Zapatos deportivos de alta gama",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "language": "es"
    }
    response = client.post("/copywriting/generate?model_name=invalid_model", json=payload)
    assert response.status_code == 400
    assert "Modelo" in response.json()["detail"]

def test_batch_too_large(client):
    payload = [{
        "product_description": f"Producto {i}",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "language": "es"
    } for i in range(25)]
    response = client.post("/copywriting/batch-generate", json=payload)
    assert response.status_code == 400
    assert "batch máximo" in response.json()["detail"]

def test_generate_invalid_input(client):
    # Falta campo obligatorio
    payload = {
        "target_platform": "Instagram",
        "tone": "inspirational",
        "language": "es"
    }
    response = client.post("/copywriting/generate?model_name=gpt2", json=payload)
    assert response.status_code == 422

def test_feedback_endpoint(client):
    feedback = {
        "type": "human",
        "score": 0.9,
        "comments": "Muy buen copy",
        "user_id": "user123",
        "timestamp": "2024-06-01T12:00:00Z"
    }
    response = client.post("/copywriting/feedback", json={"variant_id": "variant_1", "feedback": feedback})
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert data["variant_id"] == "variant_1" 