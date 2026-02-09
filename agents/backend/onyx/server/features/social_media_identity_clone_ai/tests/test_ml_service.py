"""
Tests para MLService
"""

import pytest
from ml.ml_service import MLService


def test_predict_performance():
    """Test predicción de rendimiento"""
    ml_service = MLService()
    
    prediction = ml_service.predict_content_performance(
        content="💪 Contenido motivacional con hashtags #fitness #motivation",
        platform="instagram",
        identity_id="test-identity"
    )
    
    assert "predicted_engagement" in prediction
    assert "confidence" in prediction
    assert "factors" in prediction
    assert "recommendation" in prediction
    
    assert 0.0 <= prediction["predicted_engagement"] <= 1.0
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert isinstance(prediction["factors"], list)


def test_analyze_trends():
    """Test análisis de tendencias"""
    ml_service = MLService()
    
    # Debe funcionar incluso sin datos
    trends = ml_service.analyze_content_trends("test-identity")
    
    assert "trends" in trends or "platform_stats" in trends
    assert isinstance(trends, dict)




