"""
Gamma App - Test Configuration
Pytest configuration and fixtures
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.main import app
from models.database import Base, init_database, get_db
from utils.config import get_settings

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_gamma_app.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

@pytest.fixture
def client(test_db):
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
async def async_client(test_db):
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }

@pytest.fixture
def sample_project_data():
    """Sample project data for testing"""
    return {
        "name": "Test Project",
        "description": "A test project",
        "is_public": False
    }

@pytest.fixture
def sample_content_data():
    """Sample content data for testing"""
    return {
        "title": "Test Content",
        "content_type": "presentation",
        "content_data": {
            "sections": [
                {
                    "title": "Introduction",
                    "content": "This is a test presentation"
                }
            ]
        }
    }

@pytest.fixture
def auth_headers(client, sample_user_data):
    """Get authentication headers for testing"""
    # Create user
    response = client.post("/api/auth/register", json=sample_user_data)
    assert response.status_code == 201
    
    # Login
    login_data = {
        "username": sample_user_data["username"],
        "password": sample_user_data["password"]
    }
    response = client.post("/api/auth/login", data=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def mock_ai_response():
    """Mock AI response for testing"""
    return {
        "content": "This is a generated test content",
        "metadata": {
            "model": "test-model",
            "tokens_used": 100,
            "generation_time": 1.5
        }
    }

@pytest.fixture
def mock_cache_data():
    """Mock cache data for testing"""
    return {
        "key": "test_key",
        "value": {"test": "data"},
        "ttl": 3600,
        "namespace": "test"
    }

@pytest.fixture
def mock_performance_metrics():
    """Mock performance metrics for testing"""
    return {
        "response_time": 0.5,
        "memory_usage": 50.0,
        "cpu_usage": 30.0,
        "error_rate": 0.0
    }

@pytest.fixture
def mock_security_event():
    """Mock security event for testing"""
    return {
        "event_type": "rate_limit_exceeded",
        "severity": "medium",
        "source_ip": "127.0.0.1",
        "description": "Rate limit exceeded for test user"
    }

# Async fixtures
@pytest.fixture
async def async_test_db():
    """Create async test database"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

@pytest.fixture
async def async_auth_headers(async_client, sample_user_data):
    """Get async authentication headers for testing"""
    # Create user
    response = await async_client.post("/api/auth/register", json=sample_user_data)
    assert response.status_code == 201
    
    # Login
    login_data = {
        "username": sample_user_data["username"],
        "password": sample_user_data["password"]
    }
    response = await async_client.post("/api/auth/login", data=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# Test utilities
class TestUtils:
    """Test utility functions"""
    
    @staticmethod
    def create_test_file(content: str, extension: str = ".txt") -> Path:
        """Create a test file with content"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def cleanup_test_file(file_path: Path):
        """Clean up test file"""
        if file_path.exists():
            file_path.unlink()
    
    @staticmethod
    def assert_response_success(response, expected_status: int = 200):
        """Assert response is successful"""
        assert response.status_code == expected_status
        assert "error" not in response.json()
    
    @staticmethod
    def assert_response_error(response, expected_status: int = 400):
        """Assert response is an error"""
        assert response.status_code == expected_status
        assert "error" in response.json() or "detail" in response.json()

@pytest.fixture
def test_utils():
    """Test utilities fixture"""
    return TestUtils
