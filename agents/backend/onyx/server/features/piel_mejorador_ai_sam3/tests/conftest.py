"""
Pytest configuration and fixtures.
"""

import pytest
import os
from pathlib import Path

# Set test environment variables
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("TRUTHGPT_ENDPOINT", "http://test-endpoint")


@pytest.fixture
def test_config():
    """Create test configuration."""
    from piel_mejorador_ai_sam3.config.piel_mejorador_config import PielMejoradorConfig
    
    config = PielMejoradorConfig()
    config.openrouter.api_key = "test-key"
    config.debug = True
    return config




