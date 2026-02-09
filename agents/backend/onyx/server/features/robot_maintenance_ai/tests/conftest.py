"""
Pytest configuration and fixtures.
"""

import pytest
import os
from ..config.maintenance_config import MaintenanceConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = MaintenanceConfig()
    config.openrouter.api_key = os.getenv("OPENROUTER_API_KEY", "test-key")
    return config


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data for testing."""
    return {
        "temperature": 25.5,
        "pressure": 100.0,
        "vibration": 0.5,
        "current": 10.0,
        "voltage": 220.0,
        "rpm": 1500.0
    }






