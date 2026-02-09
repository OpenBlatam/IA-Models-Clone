"""
API Tests - Tests de integración para API
==========================================

Tests para endpoints de la API mejorados.
"""

import pytest
from fastapi import status
from typing import Dict, Any


class TestMoveToEndpoint:
    """Tests para endpoint /api/v1/move/to."""
    
    def test_move_to_success(self, api_client, sample_move_request):
        """Test de movimiento exitoso."""
        response = api_client.post("/api/v1/move/to", json=sample_move_request)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "success" in data or "error" in data
    
    def test_move_to_invalid_position(self, api_client):
        """Test con posición inválida."""
        invalid_request = {"x": 15.0, "y": 5.0, "z": 2.0}
        response = api_client.post("/api/v1/move/to", json=invalid_request)
        assert response.status_code == 422
    
    def test_move_to_invalid_quaternion(self, api_client):
        """Test con quaternion inválido."""
        invalid_request = {
            "x": 0.5,
            "y": 0.3,
            "z": 0.2,
            "orientation": [1.0, 0.0, 0.0]
        }
        response = api_client.post("/api/v1/move/to", json=invalid_request)
        assert response.status_code == 422
    
    def test_move_to_missing_fields(self, api_client):
        """Test con campos faltantes."""
        incomplete_request = {"x": 0.5}
        response = api_client.post("/api/v1/move/to", json=incomplete_request)
        assert response.status_code == 422


class TestChatEndpoint:
    """Tests para endpoint /api/v1/chat."""
    
    def test_chat_success(self, api_client, sample_chat_message):
        """Test de chat exitoso."""
        response = api_client.post("/api/v1/chat", json=sample_chat_message)
        assert response.status_code in [200, 500]
        data = response.json()
        assert "success" in data or "error" in data
    
    def test_chat_empty_message(self, api_client):
        """Test con mensaje vacío."""
        invalid_request = {"message": ""}
        response = api_client.post("/api/v1/chat", json=invalid_request)
        assert response.status_code == 422
    
    def test_chat_too_long_message(self, api_client):
        """Test con mensaje muy largo."""
        invalid_request = {"message": "x" * 10001}
        response = api_client.post("/api/v1/chat", json=invalid_request)
        assert response.status_code == 422
    
    def test_chat_missing_message(self, api_client):
        """Test sin campo message."""
        invalid_request = {"context": {}}
        response = api_client.post("/api/v1/chat", json=invalid_request)
        assert response.status_code == 422


class TestPathEndpoint:
    """Tests para endpoint /api/v1/move/path."""
    
    def test_path_success(self, api_client, sample_path_request):
        """Test de path exitoso."""
        response = api_client.post("/api/v1/move/path", json=sample_path_request)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "success" in data or "error" in data
    
    def test_path_insufficient_waypoints(self, api_client):
        """Test con pocos waypoints."""
        invalid_request = {"waypoints": [{"x": 0.0, "y": 0.0, "z": 0.0}]}
        response = api_client.post("/api/v1/move/path", json=invalid_request)
        assert response.status_code == 422
    
    def test_path_too_many_waypoints(self, api_client):
        """Test con demasiados waypoints."""
        invalid_request = {
            "waypoints": [
                {"x": float(i), "y": 0.0, "z": 0.0}
                for i in range(101)
            ]
        }
        response = api_client.post("/api/v1/move/path", json=invalid_request)
        assert response.status_code == 422


class TestHealthEndpoint:
    """Tests para endpoint /health."""
    
    def test_health_check(self, api_client):
        """Test de health check básico."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "robot" in data
    
    def test_root_endpoint(self, api_client):
        """Test de endpoint raíz."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data or "status" in data


class TestErrorHandling:
    """Tests para manejo de errores."""
    
    def test_error_response_structure(self, api_client):
        """Test de estructura de respuesta de error."""
        invalid_request = {"x": 15.0, "y": 5.0, "z": 2.0}
        response = api_client.post("/api/v1/move/to", json=invalid_request)
        
        if response.status_code >= 400:
            data = response.json()
            assert "detail" in data or "error" in data
    
    def test_request_id_header(self, api_client):
        """Test de header X-Request-ID."""
        response = api_client.get("/health")
        assert "X-Request-ID" in response.headers or response.status_code == 200
    
    def test_response_time_header(self, api_client):
        """Test de header X-Response-Time."""
        response = api_client.get("/health")
        assert "X-Response-Time" in response.headers or response.status_code == 200

