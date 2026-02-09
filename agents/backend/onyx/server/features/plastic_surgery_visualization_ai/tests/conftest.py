"""Pytest configuration and fixtures."""

import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import io

from main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (500, 500), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def sample_image_small():
    """Create a small sample image."""
    img = Image.new('RGB', (200, 200), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "uploads").mkdir()
    (storage / "outputs").mkdir()
    return storage

