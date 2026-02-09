"""
Pytest configuration and fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_script():
    """Sample script for testing"""
    return {
        "text": "This is a test script for video generation. It contains multiple sentences.",
        "language": "en"
    }


@pytest.fixture
def sample_video_config():
    """Sample video config for testing"""
    return {
        "resolution": "1920x1080",
        "fps": 30,
        "duration": 10.0
    }

