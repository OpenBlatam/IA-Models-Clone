"""
Tests para las rutas de recomendaciones
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.recommendations import router
from services.recommendation_engine import RecommendationEngine


@pytest.fixture
def mock_recommendation_engine():
    """Mock del motor de recomendaciones"""
    engine = Mock(spec=RecommendationEngine)
    engine.get_content_based_recommendations = Mock(return_value=[
        {"song_id": "song-1", "similarity": 0.85},
        {"song_id": "song-2", "similarity": 0.78}
    ])
    engine.get_collaborative_recommendations = Mock(return_value=[
        {"song_id": "song-3", "score": 0.92},
        {"song_id": "song-4", "score": 0.88}
    ])
    engine.get_hybrid_recommendations = Mock(return_value=[
        {"song_id": "song-5", "score": 0.90},
        {"song_id": "song-6", "score": 0.85}
    ])
    engine.get_trending = Mock(return_value=[
        {"song_id": "trending-1", "play_count": 1000},
        {"song_id": "trending-2", "play_count": 950}
    ])
    engine.get_popular = Mock(return_value=[
        {"song_id": "popular-1", "rating": 4.8},
        {"song_id": "popular-2", "rating": 4.7}
    ])
    return engine


@pytest.fixture
def client(mock_recommendation_engine):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.recommendations.get_recommendation_engine', return_value=mock_recommendation_engine):
        with patch('api.routes.recommendations.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestContentBasedRecommendations:
    """Tests para recomendaciones basadas en contenido"""
    
    def test_get_content_based_success(self, client, mock_recommendation_engine):
        """Test de obtención exitosa de recomendaciones basadas en contenido"""
        response = client.get(
            "/recommendations/content-based",
            params={"user_id": "user-123", "limit": 10}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["type"] == "content_based"
        assert data["user_id"] == "user-123"
        assert "recommendations" in data
        assert len(data["recommendations"]) == 2
        assert data["count"] == 2
        
        mock_recommendation_engine.get_content_based_recommendations.assert_called_once_with(
            user_id="user-123",
            limit=10,
            min_similarity=0.3
        )
    
    def test_get_content_based_with_min_similarity(self, client, mock_recommendation_engine):
        """Test con similitud mínima personalizada"""
        response = client.get(
            "/recommendations/content-based",
            params={
                "user_id": "user-123",
                "limit": 5,
                "min_similarity": 0.7
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_recommendation_engine.get_content_based_recommendations.call_args[1]
        assert call_args["min_similarity"] == 0.7
    
    def test_get_content_based_with_user_from_token(self, client, mock_recommendation_engine):
        """Test usando user_id del token"""
        with patch('api.routes.recommendations.get_current_user', return_value={"user_id": "token_user"}):
            response = client.get(
                "/recommendations/content-based"
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["user_id"] == "token_user"
    
    def test_get_content_based_missing_user_id(self, client):
        """Test sin user_id"""
        with patch('api.routes.recommendations.get_current_user', return_value=None):
            response = client.get(
                "/recommendations/content-based"
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "user_id is required" in response.json()["detail"]
    
    def test_get_content_based_limit_validation(self, client):
        """Test de validación de límite"""
        # Límite muy bajo
        response = client.get(
            "/recommendations/content-based",
            params={"user_id": "user-123", "limit": 0}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Límite muy alto
        response = client.get(
            "/recommendations/content-based",
            params={"user_id": "user-123", "limit": 51}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_content_based_error_handling(self, client, mock_recommendation_engine):
        """Test de manejo de errores"""
        mock_recommendation_engine.get_content_based_recommendations.side_effect = Exception("Engine error")
        
        response = client.get(
            "/recommendations/content-based",
            params={"user_id": "user-123"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error getting recommendations" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestCollaborativeRecommendations:
    """Tests para recomendaciones colaborativas"""
    
    def test_get_collaborative_success(self, client, mock_recommendation_engine):
        """Test de obtención exitosa de recomendaciones colaborativas"""
        response = client.get(
            "/recommendations/collaborative",
            params={"user_id": "user-123", "limit": 10}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["type"] == "collaborative"
        assert data["user_id"] == "user-123"
        assert "recommendations" in data
        assert len(data["recommendations"]) == 2
    
    def test_get_collaborative_with_min_users(self, client, mock_recommendation_engine):
        """Test con mínimo de usuarios personalizado"""
        response = client.get(
            "/recommendations/collaborative",
            params={
                "user_id": "user-123",
                "min_users": 5
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_recommendation_engine.get_collaborative_recommendations.call_args[1]
        assert call_args["min_users"] == 5
    
    def test_get_collaborative_error_handling(self, client, mock_recommendation_engine):
        """Test de manejo de errores"""
        mock_recommendation_engine.get_collaborative_recommendations.side_effect = Exception("Error")
        
        response = client.get(
            "/recommendations/collaborative",
            params={"user_id": "user-123"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.unit
@pytest.mark.api
class TestHybridRecommendations:
    """Tests para recomendaciones híbridas"""
    
    def test_get_hybrid_success(self, client, mock_recommendation_engine):
        """Test de obtención exitosa de recomendaciones híbridas"""
        response = client.get(
            "/recommendations/hybrid",
            params={"user_id": "user-123", "limit": 10}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["type"] == "hybrid"
        assert "recommendations" in data
    
    def test_get_hybrid_with_weights(self, client, mock_recommendation_engine):
        """Test con pesos personalizados"""
        response = client.get(
            "/recommendations/hybrid",
            params={
                "user_id": "user-123",
                "content_weight": 0.6,
                "collaborative_weight": 0.4
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_recommendation_engine.get_hybrid_recommendations.call_args[1]
        assert call_args["content_weight"] == 0.6
        assert call_args["collaborative_weight"] == 0.4


@pytest.mark.unit
@pytest.mark.api
class TestTrendingRecommendations:
    """Tests para recomendaciones trending"""
    
    def test_get_trending_success(self, client, mock_recommendation_engine):
        """Test de obtención exitosa de trending"""
        response = client.get(
            "/recommendations/trending",
            params={"limit": 10, "timeframe": "week"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["type"] == "trending"
        assert "recommendations" in data
        assert len(data["recommendations"]) == 2
    
    def test_get_trending_different_timeframes(self, client, mock_recommendation_engine):
        """Test con diferentes timeframes"""
        timeframes = ["day", "week", "month"]
        
        for timeframe in timeframes:
            response = client.get(
                "/recommendations/trending",
                params={"timeframe": timeframe}
            )
            
            assert response.status_code == status.HTTP_200_OK
            call_args = mock_recommendation_engine.get_trending.call_args[1]
            assert call_args["timeframe"] == timeframe


@pytest.mark.unit
@pytest.mark.api
class TestPopularRecommendations:
    """Tests para recomendaciones populares"""
    
    def test_get_popular_success(self, client, mock_recommendation_engine):
        """Test de obtención exitosa de populares"""
        response = client.get(
            "/recommendations/popular",
            params={"limit": 10}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["type"] == "popular"
        assert "recommendations" in data
    
    def test_get_popular_with_min_rating(self, client, mock_recommendation_engine):
        """Test con rating mínimo"""
        response = client.get(
            "/recommendations/popular",
            params={"min_rating": 4.5}
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_recommendation_engine.get_popular.call_args[1]
        assert call_args["min_rating"] == 4.5


@pytest.mark.integration
@pytest.mark.api
class TestRecommendationsIntegration:
    """Tests de integración para recomendaciones"""
    
    def test_multiple_recommendation_types(self, client, mock_recommendation_engine):
        """Test de múltiples tipos de recomendaciones"""
        # Content-based
        response1 = client.get(
            "/recommendations/content-based",
            params={"user_id": "user-123"}
        )
        assert response1.status_code == status.HTTP_200_OK
        
        # Collaborative
        response2 = client.get(
            "/recommendations/collaborative",
            params={"user_id": "user-123"}
        )
        assert response2.status_code == status.HTTP_200_OK
        
        # Hybrid
        response3 = client.get(
            "/recommendations/hybrid",
            params={"user_id": "user-123"}
        )
        assert response3.status_code == status.HTTP_200_OK
        
        # Trending
        response4 = client.get("/recommendations/trending")
        assert response4.status_code == status.HTTP_200_OK
        
        # Popular
        response5 = client.get("/recommendations/popular")
        assert response5.status_code == status.HTTP_200_OK



