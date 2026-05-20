"""
Testing utilities for routers
"""

from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def create_test_client(router) -> TestClient:
    """
    Create a test client for a router
    
    Args:
        router: Router to test
    
    Returns:
        TestClient instance
    """
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def mock_service(service_name: str, methods: Dict[str, Any]) -> Mock:
    """
    Create a mock service
    
    Args:
        service_name: Name of the service
        methods: Dictionary of method names and return values
    
    Returns:
        Mock service instance
    """
    mock = Mock()
    for method_name, return_value in methods.items():
        if isinstance(return_value, dict) and return_value.get("async"):
            setattr(mock, method_name, AsyncMock(return_value=return_value["value"]))
        else:
            setattr(mock, method_name, Mock(return_value=return_value))
    return mock


def create_mock_track(track_id: str = "test_track_id") -> Dict[str, Any]:
    """Create a mock track response"""
    return {
        "id": track_id,
        "name": "Test Track",
        "artists": [{"name": "Test Artist"}],
        "album": {"name": "Test Album"},
        "duration_ms": 200000,
        "popularity": 50
    }


def create_mock_audio_features() -> Dict[str, Any]:
    """Create mock audio features"""
    return {
        "danceability": 0.7,
        "energy": 0.8,
        "key": 0,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.1,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.6,
        "tempo": 120.0,
        "time_signature": 4
    }

