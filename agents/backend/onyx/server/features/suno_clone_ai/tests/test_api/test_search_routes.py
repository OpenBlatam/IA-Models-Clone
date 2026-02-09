"""
Tests para las rutas de búsqueda
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.search import router
from services.search_engine import SearchEngine


@pytest.fixture
def mock_search_engine():
    """Mock del motor de búsqueda"""
    engine = Mock(spec=SearchEngine)
    engine.search = Mock(return_value=[
        {"id": "song-1", "title": "Test Song 1", "score": 0.9},
        {"id": "song-2", "title": "Test Song 2", "score": 0.8}
    ])
    engine.autocomplete = Mock(return_value=["pop", "pop rock", "pop music"])
    engine.suggest = Mock(return_value=["rock", "jazz", "blues"])
    return engine


@pytest.fixture
def client(mock_search_engine):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.search.get_search_engine', return_value=mock_search_engine):
        with patch('api.routes.search.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestSearch:
    """Tests para búsqueda"""
    
    def test_search_success(self, client, mock_search_engine):
        """Test de búsqueda exitosa"""
        response = client.get("/search?query=pop")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data or isinstance(data, list)
    
    def test_search_with_filters(self, client, mock_search_engine):
        """Test de búsqueda con filtros"""
        response = client.get(
            "/search",
            params={
                "query": "pop",
                "genre": "rock",
                "min_duration": 30,
                "max_duration": 300
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_search_pagination(self, client, mock_search_engine):
        """Test de búsqueda con paginación"""
        response = client.get(
            "/search",
            params={
                "query": "pop",
                "page": 1,
                "page_size": 10
            }
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestAutocomplete:
    """Tests para autocompletado"""
    
    def test_autocomplete_success(self, client, mock_search_engine):
        """Test de autocompletado exitoso"""
        response = client.get("/search/autocomplete?query=po")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "suggestions" in data or isinstance(data, list)


@pytest.mark.unit
@pytest.mark.api
class TestSuggest:
    """Tests para sugerencias"""
    
    def test_suggest_success(self, client, mock_search_engine):
        """Test de sugerencias exitosas"""
        response = client.get("/search/suggest?query=pop")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "suggestions" in data or isinstance(data, list)


@pytest.mark.integration
@pytest.mark.api
class TestSearchIntegration:
    """Tests de integración para búsqueda"""
    
    def test_full_search_workflow(self, client, mock_search_engine):
        """Test del flujo completo de búsqueda"""
        # 1. Búsqueda básica
        search_response = client.get("/search?query=pop")
        assert search_response.status_code == status.HTTP_200_OK
        
        # 2. Autocompletado
        autocomplete_response = client.get("/search/autocomplete?query=po")
        assert autocomplete_response.status_code == status.HTTP_200_OK
        
        # 3. Sugerencias
        suggest_response = client.get("/search/suggest?query=pop")
        assert suggest_response.status_code == status.HTTP_200_OK
