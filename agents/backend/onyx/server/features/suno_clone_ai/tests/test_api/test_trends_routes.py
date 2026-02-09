"""
Tests para las rutas de análisis de tendencias
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime

from api.routes.trends import router
from services.trend_analysis import TrendAnalysisService


@pytest.fixture
def mock_trend_service():
    """Mock del servicio de análisis de tendencias"""
    service = Mock(spec=TrendAnalysisService)
    
    # Mock de TrendReport
    genre_trend = Mock()
    genre_trend.genre = "pop"
    genre_trend.popularity = 0.85
    genre_trend.growth_rate = 0.15
    genre_trend.sample_count = 100
    
    trend_report = Mock()
    trend_report.period_start = datetime(2024, 1, 1)
    trend_report.period_end = datetime(2024, 1, 31)
    trend_report.genre_trends = [genre_trend]
    trend_report.bpm_trends = {"average": 120, "range": [100, 140]}
    trend_report.key_trends = {"C": 0.3, "G": 0.25}
    trend_report.top_tags = ["happy", "energetic", "upbeat"]
    
    service.analyze_trends = Mock(return_value=trend_report)
    service.predict_trends = Mock(return_value=trend_report)
    service.compare_periods = Mock(return_value={
        "current": trend_report,
        "previous": trend_report,
        "changes": {}
    })
    
    return service


@pytest.fixture
def client(mock_trend_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.trends.get_trend_analysis_service', return_value=mock_trend_service):
        with patch('api.routes.trends.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestAnalyzeTrends:
    """Tests para análisis de tendencias"""
    
    def test_analyze_trends_success(self, client, mock_trend_service):
        """Test de análisis exitoso"""
        songs = [
            {"genre": "pop", "bpm": 120, "tags": ["happy"]},
            {"genre": "rock", "bpm": 140, "tags": ["energetic"]}
        ]
        
        response = client.post(
            "/trends/analyze",
            json={
                "songs": songs,
                "period_days": 30
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "period_start" in data
        assert "period_end" in data
        assert "genre_trends" in data
        assert "bpm_trends" in data
        assert "key_trends" in data
        assert "top_tags" in data
    
    def test_analyze_trends_different_periods(self, client, mock_trend_service):
        """Test con diferentes períodos"""
        periods = [7, 30, 90, 365]
        
        for period in periods:
            response = client.post(
                "/trends/analyze",
                json={
                    "songs": [{"genre": "pop"}],
                    "period_days": period
                }
            )
            assert response.status_code == status.HTTP_200_OK
    
    def test_analyze_trends_empty_songs(self, client):
        """Test con lista vacía de canciones"""
        response = client.post(
            "/trends/analyze",
            json={
                "songs": [],
                "period_days": 30
            }
        )
        
        # Puede ser válido o inválido
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.unit
@pytest.mark.api
class TestPredictTrends:
    """Tests para predicción de tendencias"""
    
    def test_predict_trends_success(self, client, mock_trend_service):
        """Test de predicción exitosa"""
        current_trends = {
            "genre_trends": [{"genre": "pop", "popularity": 0.8}],
            "bpm_trends": {"average": 120}
        }
        
        response = client.post(
            "/trends/predict",
            json={"current_trends": current_trends}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predicted_trends" in data or "trends" in data


@pytest.mark.unit
@pytest.mark.api
class TestComparePeriods:
    """Tests para comparar períodos"""
    
    def test_compare_periods_success(self, client, mock_trend_service):
        """Test de comparación exitosa"""
        response = client.post(
            "/trends/compare",
            json={
                "current_period_days": 30,
                "previous_period_days": 30
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "current" in data or "comparison" in data


@pytest.mark.integration
@pytest.mark.api
class TestTrendsIntegration:
    """Tests de integración para análisis de tendencias"""
    
    def test_full_trends_workflow(self, client, mock_trend_service):
        """Test del flujo completo de análisis de tendencias"""
        songs = [
            {"genre": "pop", "bpm": 120, "tags": ["happy"]},
            {"genre": "rock", "bpm": 140, "tags": ["energetic"]}
        ]
        
        # 1. Analizar tendencias
        analyze_response = client.post(
            "/trends/analyze",
            json={"songs": songs, "period_days": 30}
        )
        assert analyze_response.status_code == status.HTTP_200_OK
        
        # 2. Predecir tendencias
        current_trends = analyze_response.json()
        predict_response = client.post(
            "/trends/predict",
            json={"current_trends": current_trends}
        )
        assert predict_response.status_code == status.HTTP_200_OK
        
        # 3. Comparar períodos
        compare_response = client.post(
            "/trends/compare",
            json={
                "current_period_days": 30,
                "previous_period_days": 30
            }
        )
        assert compare_response.status_code == status.HTTP_200_OK



