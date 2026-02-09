"""
Tests para endpoints de ML API
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


@pytest.fixture
def ml_client():
    """Fixture para crear cliente de test de ML API"""
    try:
        from ..api.ml_api import app
        return TestClient(app)
    except ImportError:
        pytest.skip("ML API not available")


class TestMLPredictionEndpoints:
    """Tests para endpoints de predicción ML"""
    
    @patch('music_analyzer_ai.services.ml_service.MLService')
    def test_predict_genre(self, mock_ml_service, ml_client):
        """Test de predicción de género"""
        if ml_client is None:
            pytest.skip("ML client not available")
        
        mock_service = Mock()
        mock_service.predict_genre.return_value = {
            "genre": "Rock",
            "confidence": 0.95
        }
        mock_ml_service.return_value = mock_service
        
        response = ml_client.post(
            "/ml/predict/genre",
            json={"audio_features": {"energy": 0.8, "tempo": 120}}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "genre" in data or "prediction" in data
    
    @patch('music_analyzer_ai.services.ml_service.MLService')
    def test_comprehensive_ml_analysis(self, mock_ml_service, ml_client):
        """Test de análisis ML comprehensivo"""
        if ml_client is None:
            pytest.skip("ML client not available")
        
        mock_service = Mock()
        mock_service.analyze_track_comprehensive.return_value = {
            "genre_prediction": {"genre": "Rock", "confidence": 0.9},
            "emotion_prediction": {"emotion": "Happy", "confidence": 0.8}
        }
        mock_ml_service.return_value = mock_service
        
        response = ml_client.post(
            "/ml/analyze/comprehensive",
            json={"track_id": "123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data is not None


class TestMLTrainingEndpoints:
    """Tests para endpoints de entrenamiento ML"""
    
    @patch('music_analyzer_ai.services.ml_service.MLService')
    def test_train_model(self, mock_ml_service, ml_client):
        """Test de entrenamiento de modelo"""
        if ml_client is None:
            pytest.skip("ML client not available")
        
        mock_service = Mock()
        mock_service.train_model.return_value = {
            "status": "success",
            "accuracy": 0.85
        }
        mock_ml_service.return_value = mock_service
        
        response = ml_client.post(
            "/ml/train",
            json={"model_type": "genre_classifier", "epochs": 10}
        )
        
        assert response.status_code in [200, 202]
    
    @patch('music_analyzer_ai.services.ml_service.MLService')
    def test_get_model_status(self, mock_ml_service, ml_client):
        """Test de estado del modelo"""
        if ml_client is None:
            pytest.skip("ML client not available")
        
        mock_service = Mock()
        mock_service.get_model_status.return_value = {
            "status": "ready",
            "version": "1.0.0"
        }
        mock_ml_service.return_value = mock_service
        
        response = ml_client.get("/ml/status")
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

