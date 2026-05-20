"""
Refactored conftest.py with comprehensive fixtures and configuration.
"""
import pytest
import asyncio
import os
import sys
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from tests.base import BaseTestClass, MockAIService, TestAssertions, TestConfig
from tests.config.test_config import test_config_manager, test_data_config, mock_config
from tests.fixtures.test_fixtures import (
    ServiceFixtures,
    DataFixtures,
    PerformanceFixtures,
    SecurityFixtures,
    MonitoringFixtures,
    AsyncFixtures,
    DatabaseFixtures,
    CacheFixtures,
    TestEnvironmentFixtures
)

# Import service and models
try:
    from agents.backend.onyx.server.features.copywriting.service import CopywritingService
    from agents.backend.onyx.server.features.copywriting.models import (
        CopywritingRequest,
        CopywritingResponse,
        BatchCopywritingRequest,
        FeedbackRequest
    )
except ImportError:
    # Fallback for testing without actual modules
    CopywritingService = Mock
    CopywritingRequest = Mock
    CopywritingResponse = Mock
    BatchCopywritingRequest = Mock
    FeedbackRequest = Mock


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add custom markers
    markers = [
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
        "integration: marks tests as integration tests",
        "performance: marks tests as performance tests",
        "security: marks tests as security tests",
        "monitoring: marks tests as monitoring tests",
        "load: marks tests as load tests",
        "unit: marks tests as unit tests",
        "benchmark: marks tests as benchmark tests",
        "example: marks tests as example tests",
        "critical: marks tests as critical",
        "optional: marks tests as optional"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)
    
    # Set test configuration
    config.test_config = test_config_manager.get_config()
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=getattr(logging, config.test_config.log_level),
        format=config.test_config.log_format
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on configuration."""
    # Skip slow tests if not requested
    if not test_config_manager.should_run_slow_tests():
        skip_slow = pytest.mark.skip(reason="slow tests not enabled")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip performance tests if not requested
    if not test_config_manager.should_run_performance_tests():
        skip_performance = pytest.mark.skip(reason="performance tests not enabled")
        for item in items:
            if "performance" in item.keywords or "benchmark" in item.keywords:
                item.add_marker(skip_performance)
    
    # Skip load tests if not requested
    if not test_config_manager.should_run_load_tests():
        skip_load = pytest.mark.skip(reason="load tests not enabled")
        for item in items:
            if "load" in item.keywords:
                item.add_marker(skip_load)


# Global fixtures
@pytest.fixture(scope="session")
def test_environment():
    """Get test environment configuration."""
    return test_config_manager.get_config().environment


@pytest.fixture(scope="session")
def test_category():
    """Get test category configuration."""
    return test_config_manager.get_config().category


@pytest.fixture(scope="session")
def performance_thresholds():
    """Get performance thresholds."""
    return test_config_manager.get_performance_thresholds()


@pytest.fixture(scope="session")
def coverage_thresholds():
    """Get coverage thresholds."""
    return test_config_manager.get_coverage_thresholds()


@pytest.fixture(scope="session")
def security_thresholds():
    """Get security thresholds."""
    return test_config_manager.get_security_thresholds()


# Service fixtures
@pytest.fixture(scope="function")
def copywriting_service():
    """Create a copywriting service instance."""
    return CopywritingService()


@pytest.fixture(scope="function")
def mock_ai_service():
    """Create a mock AI service."""
    config = mock_config.get_ai_service_config()
    return MockAIService(
        delay=config["delay"],
        should_fail=config["failure_rate"] > 0,
        response_data=config["response_template"]
    )


@pytest.fixture(scope="function")
def mock_ai_service_failing():
    """Create a failing mock AI service."""
    return MockAIService(should_fail=True)


@pytest.fixture(scope="function")
def mock_ai_service_slow():
    """Create a slow mock AI service."""
    return MockAIService(delay=1.0)


# Data fixtures
@pytest.fixture(scope="function")
def sample_request_data():
    """Provide sample request data for testing."""
    return {
        "product_description": test_data_config.get_random_product(),
        "target_platform": test_data_config.get_random_platform(),
        "tone": test_data_config.get_random_tone(),
        "target_audience": test_data_config.get_random_audience(),
        "key_points": test_data_config.get_random_key_points(),
        "instructions": test_data_config.get_random_instructions(),
        "restrictions": test_data_config.get_random_restrictions(),
        "creativity_level": 0.8,
        "language": "es"
    }


@pytest.fixture(scope="function")
def sample_response_data():
    """Provide sample response data for testing."""
    return {
        "variants": [
            {
                "headline": "¡Descubre la Innovación!",
                "primary_text": "Producto revolucionario para tu vida",
                "call_to_action": "Compra ahora",
                "hashtags": ["#innovación", "#producto"]
            }
        ],
        "model_used": "gpt-3.5-turbo",
        "generation_time": 2.5,
        "extra_metadata": {"tokens_used": 150}
    }


@pytest.fixture(scope="function")
def sample_request(sample_request_data):
    """Create a sample copywriting request."""
    return CopywritingRequest(**sample_request_data)


@pytest.fixture(scope="function")
def sample_response(sample_response_data):
    """Create a sample copywriting response."""
    return CopywritingResponse(**sample_response_data)


@pytest.fixture(scope="function")
def batch_request_data(sample_request_data):
    """Create batch request data."""
    return [
        {**sample_request_data, "product_description": f"Producto {i}"}
        for i in range(3)
    ]


@pytest.fixture(scope="function")
def batch_request(batch_request_data):
    """Create a batch copywriting request."""
    requests = [CopywritingRequest(**data) for data in batch_request_data]
    return BatchCopywritingRequest(requests=requests)


@pytest.fixture(scope="function")
def feedback_data():
    """Create feedback data."""
    return {
        "type": "human",
        "score": 0.9,
        "comments": "Muy buen copy",
        "user_id": "user123",
        "timestamp": "2024-06-01T12:00:00Z"
    }


@pytest.fixture(scope="function")
def feedback_request(feedback_data):
    """Create a feedback request."""
    return FeedbackRequest(
        variant_id="variant_1",
        feedback=feedback_data
    )


# Performance fixtures
@pytest.fixture(scope="function")
def performance_requests(sample_request_data):
    """Create multiple requests for performance testing."""
    return [
        CopywritingRequest(
            **{**sample_request_data, "product_description": f"Performance test product {i}"}
        )
        for i in range(20)
    ]


@pytest.fixture(scope="function")
def load_test_requests(sample_request_data):
    """Create requests for load testing."""
    return [
        CopywritingRequest(
            **{**sample_request_data, "product_description": f"Load test product {i}"}
        )
        for i in range(100)
    ]


# Security fixtures
@pytest.fixture(scope="function")
def malicious_inputs():
    """Create malicious input strings for testing."""
    return [
        "'; DROP TABLE users; --",
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "../../../etc/passwd",
        "{{7*7}}",
        "${7*7}",
        "`id`",
        "$(id)"
    ]


@pytest.fixture(scope="function")
def sql_injection_inputs():
    """Create SQL injection test inputs."""
    return [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "'; INSERT INTO users VALUES ('hacker', 'password'); --"
    ]


@pytest.fixture(scope="function")
def xss_inputs():
    """Create XSS test inputs."""
    return [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "';alert('XSS');//"
    ]


# Monitoring fixtures
@pytest.fixture(scope="function")
def mock_metrics():
    """Create a mock metrics collector."""
    mock_metrics = Mock()
    mock_metrics.increment_request_count.return_value = None
    mock_metrics.record_request_duration.return_value = None
    mock_metrics.record_response_time.return_value = None
    mock_metrics.record_tokens_used.return_value = None
    mock_metrics.record_model_usage.return_value = None
    mock_metrics.increment_error_count.return_value = None
    mock_metrics.record_error_type.return_value = None
    return mock_metrics


@pytest.fixture(scope="function")
def mock_logger():
    """Create a mock logger."""
    mock_logger = Mock()
    mock_logger.info.return_value = None
    mock_logger.debug.return_value = None
    mock_logger.warning.return_value = None
    mock_logger.error.return_value = None
    mock_logger.exception.return_value = None
    return mock_logger


@pytest.fixture(scope="function")
def mock_tracer():
    """Create a mock tracer."""
    mock_tracer = Mock()
    mock_span = Mock()
    mock_span.set_tag.return_value = None
    mock_span.log.return_value = None
    mock_span.set_error.return_value = None
    mock_tracer.start_span.return_value.__enter__.return_value = mock_span
    return mock_tracer


# Async fixtures
@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def async_service():
    """Create an async service instance."""
    service = CopywritingService()
    yield service
    # Cleanup if needed
    if hasattr(service, 'cleanup'):
        await service.cleanup()


# Database fixtures
@pytest.fixture(scope="function")
def mock_database():
    """Create a mock database."""
    mock_db = Mock()
    mock_db.is_healthy.return_value = True
    mock_db.save_request.return_value = "request_id_123"
    mock_db.save_response.return_value = "response_id_456"
    mock_db.get_request.return_value = None
    mock_db.get_response.return_value = None
    return mock_db


# Cache fixtures
@pytest.fixture(scope="function")
def mock_cache():
    """Create a mock cache."""
    mock_cache = Mock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.is_healthy.return_value = True
    return mock_cache


# Test environment fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Cleanup
    os.environ.pop('TESTING', None)
    os.environ.pop('LOG_LEVEL', None)


@pytest.fixture(scope="function")
def test_config():
    """Get test configuration."""
    return test_config_manager.get_config()


# Utility fixtures
@pytest.fixture(scope="function")
def test_assertions():
    """Get test assertions utility."""
    return TestAssertions()


@pytest.fixture(scope="function")
def test_data_manager():
    """Get test data manager."""
    from tests.base import test_data_manager
    yield test_data_manager
    test_data_manager.clear_cache()


# Pytest hooks
def pytest_runtest_setup(item):
    """Setup for each test."""
    # Add any test-specific setup here
    pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown for each test."""
    # Add any test-specific teardown here
    pass


def pytest_sessionstart(session):
    """Session start hook."""
    print(f"\n🚀 Starting test session with {len(session.items)} tests")
    print(f"📊 Test environment: {test_config_manager.get_config().environment.value}")
    print(f"📈 Test category: {test_config_manager.get_config().category.value}")


def pytest_sessionfinish(session, exitstatus):
    """Session finish hook."""
    print(f"\n✅ Test session completed with exit status: {exitstatus}")
    if exitstatus == 0:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom configuration
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "monitoring: marks tests as monitoring tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )
    config.addinivalue_line(
        "markers", "example: marks tests as example tests"
    )
    config.addinivalue_line(
        "markers", "critical: marks tests as critical"
    )
    config.addinivalue_line(
        "markers", "optional: marks tests as optional"
    )
