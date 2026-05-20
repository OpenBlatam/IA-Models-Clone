"""
Comprehensive test fixtures for copywriting service tests.
"""
import pytest
import asyncio
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from tests.base import BaseTestClass, MockAIService, TestAssertions
from tests.config.test_config import test_config_manager, test_data_config, mock_config
from agents.backend.onyx.server.features.copywriting.service import CopywritingService
from agents.backend.onyx.server.features.copywriting.models import (
    CopywritingRequest,
    CopywritingResponse,
    BatchCopywritingRequest,
    FeedbackRequest
)


class ServiceFixtures:
    """Fixtures for service testing."""
    
    @pytest.fixture(scope="function")
    def copywriting_service(self):
        """Create a copywriting service instance."""
        return CopywritingService()
    
    @pytest.fixture(scope="function")
    def mock_ai_service(self):
        """Create a mock AI service."""
        config = mock_config.get_ai_service_config()
        return MockAIService(
            delay=config["delay"],
            should_fail=config["failure_rate"] > 0,
            response_data=config["response_template"]
        )
    
    @pytest.fixture(scope="function")
    def mock_ai_service_failing(self):
        """Create a failing mock AI service."""
        return MockAIService(should_fail=True)
    
    @pytest.fixture(scope="function")
    def mock_ai_service_slow(self):
        """Create a slow mock AI service."""
        return MockAIService(delay=1.0)
    
    @pytest.fixture(scope="function")
    def mock_database(self):
        """Create a mock database."""
        config = mock_config.get_database_config()
        mock_db = Mock()
        mock_db.is_healthy.return_value = True
        mock_db.save_request.return_value = "request_id_123"
        mock_db.save_response.return_value = "response_id_456"
        mock_db.get_request.return_value = None
        mock_db.get_response.return_value = None
        return mock_db
    
    @pytest.fixture(scope="function")
    def mock_cache(self):
        """Create a mock cache."""
        config = mock_config.get_cache_config()
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.is_healthy.return_value = True
        return mock_cache
    
    @pytest.fixture(scope="function")
    def mock_metrics(self):
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
    def mock_logger(self):
        """Create a mock logger."""
        mock_logger = Mock()
        mock_logger.info.return_value = None
        mock_logger.debug.return_value = None
        mock_logger.warning.return_value = None
        mock_logger.error.return_value = None
        mock_logger.exception.return_value = None
        return mock_logger
    
    @pytest.fixture(scope="function")
    def mock_tracer(self):
        """Create a mock tracer."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.set_tag.return_value = None
        mock_span.log.return_value = None
        mock_span.set_error.return_value = None
        mock_tracer.start_span.return_value.__enter__.return_value = mock_span
        return mock_tracer


class DataFixtures:
    """Fixtures for test data."""
    
    @pytest.fixture(scope="function")
    def sample_request_data(self):
        """Provide sample request data."""
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
    def sample_response_data(self):
        """Provide sample response data."""
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
    def sample_request(self, sample_request_data):
        """Create a sample copywriting request."""
        return CopywritingRequest(**sample_request_data)
    
    @pytest.fixture(scope="function")
    def sample_response(self, sample_response_data):
        """Create a sample copywriting response."""
        return CopywritingResponse(**sample_response_data)
    
    @pytest.fixture(scope="function")
    def batch_request_data(self, sample_request_data):
        """Create batch request data."""
        return [
            {**sample_request_data, "product_description": f"Producto {i}"}
            for i in range(3)
        ]
    
    @pytest.fixture(scope="function")
    def batch_request(self, batch_request_data):
        """Create a batch copywriting request."""
        requests = [CopywritingRequest(**data) for data in batch_request_data]
        return BatchCopywritingRequest(requests=requests)
    
    @pytest.fixture(scope="function")
    def feedback_data(self):
        """Create feedback data."""
        return {
            "type": "human",
            "score": 0.9,
            "comments": "Muy buen copy",
            "user_id": "user123",
            "timestamp": "2024-06-01T12:00:00Z"
        }
    
    @pytest.fixture(scope="function")
    def feedback_request(self, feedback_data):
        """Create a feedback request."""
        return FeedbackRequest(
            variant_id="variant_1",
            feedback=feedback_data
        )


class PerformanceFixtures:
    """Fixtures for performance testing."""
    
    @pytest.fixture(scope="function")
    def performance_requests(self, sample_request_data):
        """Create multiple requests for performance testing."""
        return [
            CopywritingRequest(
                **{**sample_request_data, "product_description": f"Performance test product {i}"}
            )
            for i in range(20)
        ]
    
    @pytest.fixture(scope="function")
    def load_test_requests(self, sample_request_data):
        """Create requests for load testing."""
        return [
            CopywritingRequest(
                **{**sample_request_data, "product_description": f"Load test product {i}"}
            )
            for i in range(100)
        ]
    
    @pytest.fixture(scope="function")
    def performance_config(self):
        """Get performance test configuration."""
        return test_config_manager.get_performance_thresholds()
    
    @pytest.fixture(scope="function")
    def memory_monitor(self):
        """Create a memory monitor for testing."""
        import psutil
        import os
        
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.initial_memory = None
            
            def start(self):
                self.initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            def get_usage(self):
                current_memory = self.process.memory_info().rss / 1024 / 1024
                return current_memory - (self.initial_memory or 0)
        
        return MemoryMonitor()


class SecurityFixtures:
    """Fixtures for security testing."""
    
    @pytest.fixture(scope="function")
    def malicious_inputs(self):
        """Create malicious input strings for testing."""
        return [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "../../../etc/passwd",
            "{{7*7}}",
            "${7*7}",
            "`id`",
            "$(id)",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "<img src=x onerror=alert('XSS')>"
        ]
    
    @pytest.fixture(scope="function")
    def sql_injection_inputs(self):
        """Create SQL injection test inputs."""
        return [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "UNION SELECT * FROM users--"
        ]
    
    @pytest.fixture(scope="function")
    def xss_inputs(self):
        """Create XSS test inputs."""
        return [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>"
        ]
    
    @pytest.fixture(scope="function")
    def path_traversal_inputs(self):
        """Create path traversal test inputs."""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
    
    @pytest.fixture(scope="function")
    def injection_inputs(self):
        """Create injection test inputs."""
        return [
            "{{7*7}}",
            "${7*7}",
            "#{7*7}",
            "<%=7*7%>",
            "{{config.items()}}",
            "`id`",
            "$(id)"
        ]


class MonitoringFixtures:
    """Fixtures for monitoring and observability testing."""
    
    @pytest.fixture(scope="function")
    def mock_health_checks(self):
        """Create mock health check responses."""
        return {
            "database": {"status": "healthy", "response_time": 0.01},
            "cache": {"status": "healthy", "response_time": 0.005},
            "ai_service": {"status": "healthy", "response_time": 0.1},
            "overall": {"status": "healthy", "timestamp": "2024-06-01T12:00:00Z"}
        }
    
    @pytest.fixture(scope="function")
    def mock_metrics_data(self):
        """Create mock metrics data."""
        return {
            "request_count": 1000,
            "error_count": 5,
            "average_response_time": 1.5,
            "tokens_used": 50000,
            "model_usage": {
                "gpt-3.5-turbo": 800,
                "gpt-4": 200
            }
        }
    
    @pytest.fixture(scope="function")
    def mock_log_entries(self):
        """Create mock log entries."""
        return [
            {"level": "INFO", "message": "Request processed", "timestamp": "2024-06-01T12:00:00Z"},
            {"level": "ERROR", "message": "AI service error", "timestamp": "2024-06-01T12:00:01Z"},
            {"level": "DEBUG", "message": "Cache hit", "timestamp": "2024-06-01T12:00:02Z"}
        ]
    
    @pytest.fixture(scope="function")
    def mock_trace_data(self):
        """Create mock trace data."""
        return {
            "trace_id": "trace_123",
            "span_id": "span_456",
            "operation": "generate_copywriting",
            "duration": 1.5,
            "tags": {
                "model": "gpt-3.5-turbo",
                "platform": "Instagram",
                "tone": "inspirational"
            }
        }


class AsyncFixtures:
    """Fixtures for async testing."""
    
    @pytest.fixture(scope="function")
    def event_loop(self):
        """Create an event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture(scope="function")
    async def async_service(self):
        """Create an async service instance."""
        service = CopywritingService()
        yield service
        # Cleanup if needed
        if hasattr(service, 'cleanup'):
            await service.cleanup()
    
    @pytest.fixture(scope="function")
    async def async_mock_ai(self):
        """Create an async mock AI service."""
        return MockAIService(delay=0.1)


class DatabaseFixtures:
    """Fixtures for database testing."""
    
    @pytest.fixture(scope="function")
    def temp_database(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture(scope="function")
    def mock_database_connection(self, temp_database):
        """Create a mock database connection."""
        mock_conn = Mock()
        mock_conn.execute.return_value = Mock()
        mock_conn.commit.return_value = None
        mock_conn.rollback.return_value = None
        mock_conn.close.return_value = None
        return mock_conn


class CacheFixtures:
    """Fixtures for cache testing."""
    
    @pytest.fixture(scope="function")
    def mock_cache_backend(self):
        """Create a mock cache backend."""
        mock_backend = Mock()
        mock_backend.get.return_value = None
        mock_backend.set.return_value = True
        mock_backend.delete.return_value = True
        mock_backend.clear.return_value = True
        mock_backend.keys.return_value = []
        return mock_backend
    
    @pytest.fixture(scope="function")
    def cache_data(self):
        """Create sample cache data."""
        return {
            "request_123": {"status": "completed", "result": "test_result"},
            "request_456": {"status": "processing", "result": None},
            "request_789": {"status": "failed", "error": "test_error"}
        }


class TestEnvironmentFixtures:
    """Fixtures for test environment setup."""
    
    @pytest.fixture(scope="session", autouse=True)
    def setup_test_environment(self):
        """Setup test environment."""
        # Set test environment variables
        os.environ['TESTING'] = 'true'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
        yield
        
        # Cleanup
        os.environ.pop('TESTING', None)
        os.environ.pop('LOG_LEVEL', None)
    
    @pytest.fixture(scope="function")
    def test_config(self):
        """Get test configuration."""
        return test_config_manager.get_config()
    
    @pytest.fixture(scope="function")
    def test_data_manager(self):
        """Get test data manager."""
        from tests.base import test_data_manager
        yield test_data_manager
        test_data_manager.clear_cache()


# Export all fixtures
__all__ = [
    'ServiceFixtures',
    'DataFixtures', 
    'PerformanceFixtures',
    'SecurityFixtures',
    'MonitoringFixtures',
    'AsyncFixtures',
    'DatabaseFixtures',
    'CacheFixtures',
    'TestEnvironmentFixtures'
]
