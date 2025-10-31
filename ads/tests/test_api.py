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
import httpx
from datetime import datetime

from agents.backend.onyx.server.features.ads.api import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
    router,
    AdsGenerationRequest,
    BackgroundRemovalRequest,
    AdsAnalyticsRequest,
    BrandVoice,
    AudienceProfile,
    ProjectContext,
    ContentSource,
    EmailSequenceMetrics,
    EmailSequenceSettings
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
def mock_ads_service():
    
    """mock_ads_service function."""
return AsyncMock()

# Test ads generation
def test_generate_ads(client, mock_user, mock_ads_service) -> Any:
    """Test ads generation endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        with patch('onyx.server.features.ads.api.AdsDBService') as mock_db:
            mock_service_class.return_value = mock_ads_service
            mock_ads_service.generate_ads.return_value = {
                "url": "https://example.com/ads/1",
                "content": "Generated ad content"
            }
            mock_db.create_ads_generation.return_value = {"id": 1}
            
            response = client.post(
                "/generate",
                json={
                    "prompt": "Test prompt",
                    "type": "social_media",
                    "brand_voice": {
                        "tone": "professional",
                        "style": "conversational"
                    },
                    "audience_profile": {
                        "demographics": {"age_range": "25-34"}
                    }
                },
                headers={"Authorization": f"Bearer {mock_user['id']}"}
            )
            
            assert response.status_code == 200
            assert "url" in response.json()
            mock_ads_service.generate_ads.assert_called_once()
            mock_db.create_ads_generation.assert_called_once()

# Test background removal
def test_remove_background(client, mock_user, mock_ads_service) -> Any:
    """Test background removal endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.remove_background.return_value = {
            "image_url": "https://example.com/processed/1.png"
        }
        
        response = client.post(
            "/remove-background",
            json={
                "image_url": "https://example.com/image.jpg",
                "metadata": {"type": "product"}
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "image_url" in response.json()
        mock_ads_service.remove_background.assert_called_once()

# Test analytics tracking
def test_track_analytics(client, mock_user, mock_ads_service) -> Any:
    """Test analytics tracking endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.track_analytics.return_value = {
            "status": "success",
            "metrics": {"views": 1000}
        }
        
        response = client.post(
            "/analytics",
            json={
                "ads_generation_id": 1,
                "metrics": {"views": 1000},
                "email_metrics": {
                    "sequence_id": "seq1",
                    "total_sent": 100,
                    "opens": 50
                }
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert "status" in response.json()
        mock_ads_service.track_analytics.assert_called_once()

# Test list ads
def test_list_ads(client, mock_user, mock_ads_service) -> List[Any]:
    """Test list ads endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.list_ads.return_value = [
            {"id": 1, "type": "social_media"},
            {"id": 2, "type": "display"}
        ]
        
        response = client.get(
            "/list?type=social_media&limit=10&offset=0",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        mock_ads_service.list_ads.assert_called_once()

# Test get ads
def test_get_ads(client, mock_user, mock_ads_service) -> Optional[Dict[str, Any]]:
    """Test get ads endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.get_ads.return_value = {
            "id": 1,
            "type": "social_media",
            "content": "Ad content"
        }
        
        response = client.get(
            "/1",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert response.json()["id"] == 1
        mock_ads_service.get_ads.assert_called_once_with(1)

# Test delete ads
def test_delete_ads(client, mock_user, mock_ads_service) -> Any:
    """Test delete ads endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.delete_ads.return_value = {"status": "success"}
        
        response = client.delete(
            "/1",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_ads_service.delete_ads.assert_called_once_with(1)

# Test list background removals
def test_list_background_removals(client, mock_user, mock_ads_service) -> List[Any]:
    """Test list background removals endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.list_background_removals.return_value = [
            {"id": 1, "image_url": "https://example.com/1.png"},
            {"id": 2, "image_url": "https://example.com/2.png"}
        ]
        
        response = client.get(
            "/background-removals?limit=10&offset=0",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        mock_ads_service.list_background_removals.assert_called_once()

# Test get background removal
def test_get_background_removal(client, mock_user, mock_ads_service) -> Optional[Dict[str, Any]]:
    """Test get background removal endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.get_background_removal.return_value = {
            "id": 1,
            "image_url": "https://example.com/1.png"
        }
        
        response = client.get(
            "/background-removals/1",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert response.json()["id"] == 1
        mock_ads_service.get_background_removal.assert_called_once_with(1)

# Test delete background removal
def test_delete_background_removal(client, mock_user, mock_ads_service) -> Any:
    """Test delete background removal endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.delete_background_removal.return_value = {"status": "success"}
        
        response = client.delete(
            "/background-removals/1",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_ads_service.delete_background_removal.assert_called_once_with(1)

# Test list analytics
def test_list_analytics(client, mock_user, mock_ads_service) -> List[Any]:
    """Test list analytics endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.list_analytics.return_value = [
            {"id": 1, "metrics": {"views": 1000}},
            {"id": 2, "metrics": {"views": 2000}}
        ]
        
        response = client.get(
            "/analytics?ads_generation_id=1&limit=10&offset=0",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        mock_ads_service.list_analytics.assert_called_once()

# Test get analytics
def test_get_analytics(client, mock_user, mock_ads_service) -> Optional[Dict[str, Any]]:
    """Test get analytics endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.get_analytics.return_value = {
            "id": 1,
            "metrics": {"views": 1000}
        }
        
        response = client.get(
            "/analytics/1",
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert response.json()["id"] == 1
        mock_ads_service.get_analytics.assert_called_once_with(1)

# Test stream ads
def test_stream_ads(client, mock_user, mock_ads_service) -> Any:
    """Test stream ads endpoint."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.stream_ads.return_value = [
            "Ad 1",
            "Ad 2",
            "Ad 3"
        ]
        
        response = client.post(
            "/stream",
            json={
                "url": "https://example.com",
                "type": "social_media",
                "prompt": "Test prompt"
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        mock_ads_service.stream_ads.assert_called_once()

# Test error handling
def test_error_handling(client, mock_user, mock_ads_service) -> Any:
    """Test error handling in endpoints."""
    with patch('onyx.server.features.ads.api.AdsService') as mock_service_class:
        mock_service_class.return_value = mock_ads_service
        mock_ads_service.generate_ads.side_effect = Exception("Test error")
        
        response = client.post(
            "/generate",
            json={
                "prompt": "Test prompt",
                "type": "social_media"
            },
            headers={"Authorization": f"Bearer {mock_user['id']}"}
        )
        
        assert response.status_code == 500
        assert "error" in response.json() 