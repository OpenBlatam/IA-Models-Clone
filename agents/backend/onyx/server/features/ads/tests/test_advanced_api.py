from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from agents.backend.onyx.server.features.ads.api.advanced import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
    router,
    TrainingDataRequest,
    ContentOptimizationRequest,
    BrandVoiceAnalysisRequest,
    CompetitorAnalysisRequest,
    ContentVariationsRequest
)

# Test client fixture
@pytest.fixture
def client():
    
    """client function."""
return TestClient(router)

# Mock user fixture
@pytest.fixture
def mock_user():
    
    """mock_user function."""
return {
        "id": "test_user_id",
        "email": "test@example.com",
        "role": "admin"
    }

# Mock service fixture
@pytest.fixture
def mock_service():
    
    """mock_service function."""
return AsyncMock()

# Test training data endpoint
def test_train_ai_model(client, mock_user, mock_service) -> Any:
    """Test AI model training endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.train_ai_model.return_value = {"status": "success"}
        
        response = client.post(
            "/train-ai",
            json={
                "training_data": [
                    {
                        "content": "Sample content",
                        "metrics": {"engagement": 0.8}
                    }
                ]
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_service.train_ai_model.assert_called_once()

# Test content optimization endpoint
def test_optimize_content(client, mock_user, mock_service) -> Any:
    """Test content optimization endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.optimize_content.return_value = {"optimized_content": "Improved content"}
        
        response = client.post(
            "/optimize-content",
            json={
                "content": "Original content",
                "optimization_type": "engagement"
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "optimized_content" in response.json()
        mock_service.optimize_content.assert_called_once()

# Test audience analysis endpoint
def test_analyze_audience(client, mock_user, mock_service) -> Any:
    """Test audience analysis endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.analyze_audience.return_value = {
            "demographics": {"age": "25-34"},
            "interests": ["technology"]
        }
        
        response = client.get(
            "/audience/test_segment",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "demographics" in response.json()
        mock_service.analyze_audience.assert_called_once_with("test_segment")

# Test brand voice analysis endpoint
def test_analyze_brand_voice(client, mock_user, mock_service) -> Any:
    """Test brand voice analysis endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.analyze_brand_voice.return_value = {
            "tone": "professional",
            "style": "formal"
        }
        
        response = client.post(
            "/brand-voice",
            json={
                "content_samples": ["Sample 1", "Sample 2"]
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "tone" in response.json()
        mock_service.analyze_brand_voice.assert_called_once()

# Test content performance tracking endpoint
def test_track_content_performance(client, mock_user, mock_service) -> Any:
    """Test content performance tracking endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.track_content_performance.return_value = {
            "views": 1000,
            "engagement": 0.05
        }
        
        response = client.get(
            "/performance/test_content",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "views" in response.json()
        mock_service.track_content_performance.assert_called_once_with("test_content")

# Test AI recommendations endpoint
def test_generate_ai_recommendations(client, mock_user, mock_service) -> Any:
    """Test AI recommendations endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.generate_ai_recommendations.return_value = [
            "Recommendation 1",
            "Recommendation 2"
        ]
        
        response = client.post(
            "/recommendations",
            json={
                "content": "Test content",
                "context": {"platform": "social_media"}
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "recommendations" in response.json()
        mock_service.generate_ai_recommendations.assert_called_once()

# Test content impact analysis endpoint
def test_analyze_content_impact(client, mock_user, mock_service) -> Any:
    """Test content impact analysis endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.analyze_content_impact.return_value = {
            "reach": 5000,
            "engagement": 0.08
        }
        
        response = client.get(
            "/impact/test_content",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "reach" in response.json()
        mock_service.analyze_content_impact.assert_called_once_with("test_content")

# Test audience targeting optimization endpoint
def test_optimize_audience_targeting(client, mock_user, mock_service) -> Optional[Dict[str, Any]]:
    """Test audience targeting optimization endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.optimize_audience_targeting.return_value = {
            "optimized_segments": ["segment1", "segment2"]
        }
        
        response = client.post(
            "/audience/optimize/test_segment",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "optimized_segments" in response.json()
        mock_service.optimize_audience_targeting.assert_called_once_with("test_segment")

# Test content variations endpoint
def test_generate_content_variations(client, mock_user, mock_service) -> Any:
    """Test content variations endpoint."""
    with patch('onyx.server.features.ads.api.advanced.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.generate_content_variations.return_value = [
            "Variation 1",
            "Variation 2",
            "Variation 3"
        ]
        
        response = client.post(
            "/variations",
            json={
                "content": "Original content",
                "variations": 3
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "variations" in response.json()
        mock_service.generate_content_variations.assert_called_once()

# Test competitor analysis endpoint
def test_analyze_competitor_content(client, mock_user, mock_service) -> Any:
    """Test competitor analysis endpoint."""
    with patch('onyx.server.features.ads.advanced_api.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.analyze_competitor_content.return_value = {
            "similarities": ["tone", "style"],
            "differences": ["length", "focus"]
        }
        
        response = client.post(
            "/competitor",
            json={
                "competitor_urls": [
                    "https://competitor1.com",
                    "https://competitor2.com"
                ]
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "similarities" in response.json()
        mock_service.analyze_competitor_content.assert_called_once()

# Test error handling
def test_error_handling(client, mock_user, mock_service) -> Any:
    """Test error handling in endpoints."""
    with patch('onyx.server.features.ads.advanced_api.AdvancedAdsService') as mock_service_class:
        mock_service_class.return_value = mock_service
        mock_service.optimize_content.side_effect = Exception("Test error")
        
        response = client.post(
            "/optimize-content",
            json={
                "content": "Test content",
                "optimization_type": "engagement"
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 500
        assert "error" in response.json() 