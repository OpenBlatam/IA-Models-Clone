"""
Tests para las rutas de estadísticas
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from api.routes.stats import router
from services.song_service import SongService
from services.metrics_service import MetricsService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.list_songs = Mock(return_value=[
        {
            "song_id": "song-1",
            "user_id": "user-1",
            "status": "completed",
            "genre": "pop",
            "created_at": datetime.now().isoformat()
        },
        {
            "song_id": "song-2",
            "user_id": "user-2",
            "status": "completed",
            "genre": "rock",
            "created_at": datetime.now().isoformat()
        }
    ])
    return service


@pytest.fixture
def mock_metrics_service():
    """Mock del servicio de métricas"""
    service = Mock(spec=MetricsService)
    service.get_stats = Mock(return_value={
        "total_songs": 100,
        "total_users": 50
    })
    return service


@pytest.fixture
def client(mock_song_service, mock_metrics_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    from api.dependencies import SongServiceDep, MetricsServiceDep
    
    app = FastAPI()
    app.include_router(router)
    
    def get_song_service():
        return mock_song_service
    
    def get_metrics_service():
        return mock_metrics_service
    
    app.dependency_overrides[SongServiceDep] = get_song_service
    app.dependency_overrides[MetricsServiceDep] = get_metrics_service
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.api
class TestGetOverviewStats:
    """Tests para obtener estadísticas generales"""
    
    def test_get_overview_stats_success(self, client, mock_song_service, mock_metrics_service):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/stats/overview")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_songs" in data or "songs" in data
        assert "period" in data or "days" in data
    
    def test_get_overview_stats_with_days(self, client, mock_song_service, mock_metrics_service):
        """Test con número de días personalizado"""
        response = client.get("/stats/overview?days=30")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data.get("days") == 30 or data.get("period") is not None
    
    def test_get_overview_stats_with_trends(self, client, mock_song_service, mock_metrics_service):
        """Test con tendencias incluidas"""
        response = client.get("/stats/overview?include_trends=true")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Puede incluir información de tendencias
        assert isinstance(data, dict)
    
    def test_get_overview_stats_days_validation(self, client):
        """Test de validación de días"""
        # Días muy bajo
        response = client.get("/stats/overview?days=0")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Días muy alto
        response = client.get("/stats/overview?days=366")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.unit
@pytest.mark.api
class TestGetTopSongs:
    """Tests para obtener top canciones"""
    
    def test_get_top_songs_success(self, client, mock_song_service):
        """Test de obtención exitosa de top canciones"""
        response = client.get("/stats/top-songs")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "songs" in data or isinstance(data, list)
    
    def test_get_top_songs_with_limit(self, client, mock_song_service):
        """Test con límite personalizado"""
        response = client.get("/stats/top-songs?limit=10")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        if isinstance(data, dict):
            assert "limit" in data or len(data.get("songs", [])) <= 10


@pytest.mark.unit
@pytest.mark.api
class TestGetGenreStats:
    """Tests para obtener estadísticas por género"""
    
    def test_get_genre_stats_success(self, client, mock_song_service):
        """Test de obtención exitosa de estadísticas por género"""
        response = client.get("/stats/genres")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "genres" in data or isinstance(data, dict)


@pytest.mark.integration
@pytest.mark.api
class TestStatsIntegration:
    """Tests de integración para estadísticas"""
    
    def test_full_stats_workflow(self, client, mock_song_service, mock_metrics_service):
        """Test del flujo completo de estadísticas"""
        # 1. Overview
        overview_response = client.get("/stats/overview?days=7")
        assert overview_response.status_code == status.HTTP_200_OK
        
        # 2. Top songs
        top_response = client.get("/stats/top-songs?limit=10")
        assert top_response.status_code == status.HTTP_200_OK
        
        # 3. Genre stats
        genre_response = client.get("/stats/genres")
        assert genre_response.status_code == status.HTTP_200_OK



