"""
Tests para las rutas de búsqueda avanzada
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
import json

from api.routes.search_advanced import router
from services.search_engine import SearchEngine


@pytest.fixture
def mock_search_engine():
    """Mock del motor de búsqueda"""
    engine = Mock(spec=SearchEngine)
    engine.index_document = Mock(return_value=True)
    engine.search = Mock(return_value=[
        {"id": "doc-1", "score": 0.9, "content": "Test content 1"},
        {"id": "doc-2", "score": 0.8, "content": "Test content 2"}
    ])
    engine.fuzzy_search = Mock(return_value=[
        {"id": "doc-1", "score": 0.85}
    ])
    engine.autocomplete = Mock(return_value=["pop", "pop rock", "pop music"])
    return engine


@pytest.fixture
def client(mock_search_engine):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.search_advanced.get_search_engine', return_value=mock_search_engine):
        with patch('api.routes.search_advanced.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestIndexDocument:
    """Tests para indexar documento"""
    
    def test_index_document_success(self, client, mock_search_engine):
        """Test de indexación exitosa"""
        response = client.post(
            "/search/index",
            params={"doc_id": "doc-123"},
            json={
                "content": {
                    "title": "Test Song",
                    "artist": "Test Artist",
                    "genre": "pop"
                }
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["doc_id"] == "doc-123"
    
    def test_index_document_with_text_fields(self, client, mock_search_engine):
        """Test con campos de texto especificados"""
        response = client.post(
            "/search/index",
            params={"doc_id": "doc-123"},
            json={
                "content": {"title": "Test", "description": "Test desc"},
                "text_fields": ["title", "description"]
            }
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestSearch:
    """Tests para búsqueda"""
    
    def test_search_success(self, client, mock_search_engine):
        """Test de búsqueda exitosa"""
        response = client.get("/search/query?q=pop")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data or isinstance(data, list)
    
    def test_search_with_filters(self, client, mock_search_engine):
        """Test con filtros"""
        filters = json.dumps({"genre": "pop", "year": 2024})
        response = client.get(f"/search/query?q=pop&filters={filters}")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_search_with_fuzzy(self, client, mock_search_engine):
        """Test con búsqueda fuzzy"""
        response = client.get("/search/query?q=pop&fuzzy=true")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_search_invalid_filters(self, client):
        """Test con filtros inválidos"""
        response = client.get("/search/query?q=pop&filters=invalid_json")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid filters JSON" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestFuzzySearch:
    """Tests para búsqueda fuzzy"""
    
    def test_fuzzy_search_success(self, client, mock_search_engine):
        """Test de búsqueda fuzzy exitosa"""
        response = client.get("/search/fuzzy?q=pop")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data or isinstance(data, list)


@pytest.mark.unit
@pytest.mark.api
class TestAutocomplete:
    """Tests para autocompletado"""
    
    def test_autocomplete_success(self, client, mock_search_engine):
        """Test de autocompletado exitoso"""
        response = client.get("/search/autocomplete?q=po")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "suggestions" in data or isinstance(data, list)


@pytest.mark.integration
@pytest.mark.api
class TestSearchAdvancedIntegration:
    """Tests de integración para búsqueda avanzada"""
    
    def test_full_search_workflow(self, client, mock_search_engine):
        """Test del flujo completo de búsqueda"""
        # 1. Indexar documento
        index_response = client.post(
            "/search/index",
            params={"doc_id": "doc-123"},
            json={"content": {"title": "Test Song"}}
        )
        assert index_response.status_code == status.HTTP_200_OK
        
        # 2. Búsqueda básica
        search_response = client.get("/search/query?q=test")
        assert search_response.status_code == status.HTTP_200_OK
        
        # 3. Búsqueda fuzzy
        fuzzy_response = client.get("/search/fuzzy?q=test")
        assert fuzzy_response.status_code == status.HTTP_200_OK
        
        # 4. Autocompletado
        autocomplete_response = client.get("/search/autocomplete?q=te")
        assert autocomplete_response.status_code == status.HTTP_200_OK



