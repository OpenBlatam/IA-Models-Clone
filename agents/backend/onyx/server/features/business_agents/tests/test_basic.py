"""Basic tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.fixture
def client():
    """Create test client."""
    from ..main import app
    return TestClient(app)


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = Mock()
    cache.get = Mock(return_value=None)
    cache.set = Mock(return_value=None)
    cache.init = Mock(return_value=None)
    return cache


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing."""
    http_client = Mock()
    http_client.get_json = Mock(return_value={"status": "ok"})
    http_client.close = Mock(return_value=None)
    return http_client


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_liveness(client):
    """Test liveness probe."""
    response = client.get("/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


def test_readiness(client, mock_cache, mock_http_client):
    """Test readiness probe with mocked dependencies."""
    from ..main import app
    
    with patch.object(app.state, "cache", mock_cache), \
         patch.object(app.state, "http", mock_http_client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data


def test_custom_exceptions():
    """Test custom exception classes."""
    from ..exceptions import ValidationError, NotFoundError, RateLimitError
    
    # Validation error
    error = ValidationError("Invalid input", field="email")
    assert error.status_code == 422
    assert error.error_code == "VALIDATION_ERROR"
    assert "email" in error.metadata.get("field", "")
    
    # Not found error
    error = NotFoundError("user", "123")
    assert error.status_code == 404
    assert error.error_code == "NOT_FOUND"
    
    # Rate limit error
    error = RateLimitError(retry_after=60)
    assert error.status_code == 429
    assert error.error_code == "RATE_LIMIT_EXCEEDED"


