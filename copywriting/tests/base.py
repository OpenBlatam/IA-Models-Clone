"""
Base test classes and shared utilities for copywriting service tests.
"""
import pytest
import asyncio
from typing import Dict, Any, List, Optional, Type, Union
from unittest.mock import Mock, patch, AsyncMock
import time
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (
    CopywritingInput,
    CopywritingOutput,
    Feedback,
    SectionFeedback,
    CopyVariantHistory,
    get_settings
)


class BaseTestClass:
    """Base class for all copywriting service tests."""
    
    @pytest.fixture
    def sample_request_data(self) -> Dict[str, Any]:
        """Provide sample request data for testing."""
        return {
            "product_description": "Zapatos deportivos de alta gama",
            "target_platform": "Instagram",
            "tone": "inspirational",
            "target_audience": "Jóvenes activos",
            "key_points": ["Comodidad", "Estilo", "Durabilidad"],
            "instructions": "Enfatiza la innovación",
            "restrictions": ["no mencionar precio"],
            "creativity_level": 0.8,
            "language": "es"
        }
    
    @pytest.fixture
    def sample_response_data(self) -> Dict[str, Any]:
        """Provide sample response data for testing."""
        return {
            "variants": [
                {
                    "headline": "¡Descubre la Comodidad Perfecta!",
                    "primary_text": "Zapatos deportivos diseñados para tu máximo rendimiento",
                    "call_to_action": "Compra ahora",
                    "hashtags": ["#deportes", "#comodidad"]
                }
            ],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 2.5,
            "extra_metadata": {"tokens_used": 150}
        }
    
    def create_request(self, **overrides) -> CopywritingInput:
        """Create a copywriting request with optional overrides."""
        base_data = self.sample_request_data.copy()
        base_data.update(overrides)
        # Add required fields for CopywritingInput
        if 'use_case' not in base_data:
            base_data['use_case'] = 'product_launch'
        if 'content_type' not in base_data:
            base_data['content_type'] = 'social_post'
        return CopywritingInput(**base_data)
    
    def create_response(self, **overrides) -> CopywritingOutput:
        """Create a copywriting response with optional overrides."""
        base_data = self.sample_response_data.copy()
        base_data.update(overrides)
        return CopywritingOutput(**base_data)
    
    def create_batch_request(self, count: int = 3) -> List[CopywritingInput]:
        """Create a batch copywriting request."""
        requests = [
            self.create_request(
                product_description=f"Producto {i}",
                target_platform=["Instagram", "Facebook", "Twitter"][i % 3],
                tone=["inspirational", "informative", "playful"][i % 3]
            )
            for i in range(count)
        ]
        return requests
    
    def create_feedback_request(self, **overrides) -> Feedback:
        """Create a feedback request with optional overrides."""
        feedback_data = {
            "type": "human",
            "score": 0.9,
            "comments": "Muy buen copy",
            "user_id": "user123"
        }
        feedback_data.update(overrides.get('feedback', {}))
        
        return Feedback(**feedback_data)


class MockAIService:
    """Mock AI service for testing."""
    
    def __init__(self, delay: float = 0.1, should_fail: bool = False, response_data: Optional[Dict] = None):
        self.delay = delay
        self.should_fail = should_fail
        self.call_count = 0
        self.response_data = response_data or {
            "variants": [{"headline": "Mock Headline", "primary_text": "Mock Content"}],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 0.1,
            "extra_metadata": {"tokens_used": 50}
        }
    
    async def mock_call(self, request: CopywritingInput, model: str) -> Dict[str, Any]:
        """Mock AI model call."""
        self.call_count += 1
        await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise Exception("Mock AI service error")
        
        response = self.response_data.copy()
        response["model_used"] = model
        response["generation_time"] = self.delay
        response["extra_metadata"]["call_count"] = self.call_count
        
        # Customize response based on request
        if hasattr(request, 'product_description'):
            response["variants"][0]["headline"] = f"Mock Headline for {request.product_description}"
            response["variants"][0]["primary_text"] = f"Mock content for {request.target_platform}"
        
        return response


class TestAssertions:
    """Custom assertions for copywriting tests."""
    
    @staticmethod
    def assert_valid_copywriting_response(response: CopywritingOutput):
        """Assert that a copywriting response is valid."""
        assert isinstance(response, CopywritingOutput)
        assert hasattr(response, 'variants') or hasattr(response, 'results')
        
        # Handle both single response and batch response formats
        if hasattr(response, 'variants'):
            variants = response.variants
        else:
            variants = response.results if hasattr(response, 'results') else []
        
        assert isinstance(variants, list)
        assert len(variants) > 0
        
        for variant in variants:
            assert isinstance(variant, dict)
            assert "headline" in variant
            assert "primary_text" in variant
            assert isinstance(variant["headline"], str)
            assert isinstance(variant["primary_text"], str)
            assert len(variant["headline"]) > 0
            assert len(variant["primary_text"]) > 0
    
    @staticmethod
    def assert_valid_batch_response(batch_response, expected_count: int):
        """Assert that a batch response is valid."""
        assert hasattr(batch_response, 'results')
        assert isinstance(batch_response.results, list)
        assert len(batch_response.results) == expected_count
        
        for result in batch_response.results:
            TestAssertions.assert_valid_copywriting_response(result)
    
    @staticmethod
    def assert_error_response(response, expected_status_code: int, expected_error_contains: str = None):
        """Assert that an error response is valid."""
        assert response.status_code == expected_status_code
        
        if expected_error_contains:
            response_data = response.json()
            assert expected_error_contains in response_data.get("detail", "")


class PerformanceMixin:
    """Mixin for performance testing utilities."""
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    async def measure_async_execution_time(self, coro):
        """Measure execution time of an async function."""
        start_time = time.time()
        result = await coro
        end_time = time.time()
        return result, end_time - start_time
    
    def assert_performance_threshold(self, execution_time: float, max_time: float):
        """Assert that execution time is within threshold."""
        assert execution_time <= max_time, f"Execution time {execution_time}s exceeds threshold {max_time}s"


class SecurityMixin:
    """Mixin for security testing utilities."""
    
    def get_malicious_inputs(self) -> List[str]:
        """Get list of malicious input strings for testing."""
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
    
    def assert_input_sanitized(self, input_value: str, sanitized_value: str):
        """Assert that input was properly sanitized."""
        # Basic sanitization checks
        assert "<script>" not in sanitized_value.lower()
        assert "javascript:" not in sanitized_value.lower()
        assert "../" not in sanitized_value
        assert "{{" not in sanitized_value
        assert "${" not in sanitized_value


class MonitoringMixin:
    """Mixin for monitoring and observability testing."""
    
    def mock_metrics_collection(self):
        """Mock metrics collection for testing."""
        return patch('agents.backend.onyx.server.features.copywriting.service.metrics')
    
    def mock_logging(self):
        """Mock logging for testing."""
        return patch('agents.backend.onyx.server.features.copywriting.service.logger')
    
    def mock_tracing(self):
        """Mock tracing for testing."""
        return patch('agents.backend.onyx.server.features.copywriting.service.tracer')
    
    def assert_metrics_collected(self, mock_metrics, expected_calls: List[str]):
        """Assert that expected metrics were collected."""
        for call_name in expected_calls:
            assert hasattr(mock_metrics, call_name)
            getattr(mock_metrics, call_name).assert_called()


class TestDataManager:
    """Centralized test data management."""
    
    def __init__(self):
        self._test_data_cache = {}
    
    def get_test_data(self, data_type: str, **overrides) -> Dict[str, Any]:
        """Get test data with optional overrides."""
        if data_type not in self._test_data_cache:
            self._test_data_cache[data_type] = self._create_test_data(data_type)
        
        data = self._test_data_cache[data_type].copy()
        data.update(overrides)
        return data
    
    def _create_test_data(self, data_type: str) -> Dict[str, Any]:
        """Create test data based on type."""
        data_types = {
            'request': {
                "product_description": "Test product",
                "target_platform": "Instagram",
                "tone": "inspirational",
                "language": "es"
            },
            'response': {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0,
                "extra_metadata": {}
            },
            'feedback': {
                "type": "human",
                "score": 0.9,
                "comments": "Test feedback",
                "user_id": "test_user"
            }
        }
        
        return data_types.get(data_type, {})
    
    def clear_cache(self):
        """Clear test data cache."""
        self._test_data_cache.clear()


# Global test data manager instance
test_data_manager = TestDataManager()


class TestConfig:
    """Test configuration and constants."""
    
    # Performance thresholds
    SINGLE_REQUEST_MAX_TIME = 1.0
    BATCH_REQUEST_MAX_TIME = 5.0
    CONCURRENT_REQUEST_MAX_TIME = 10.0
    LOAD_TEST_MAX_TIME = 60.0
    
    # Memory thresholds
    MAX_MEMORY_INCREASE_MB = 200.0
    
    # Coverage thresholds
    MIN_LINE_COVERAGE = 90.0
    MIN_BRANCH_COVERAGE = 85.0
    MIN_FUNCTION_COVERAGE = 95.0
    
    # Load testing
    MAX_CONCURRENT_REQUESTS = 100
    MAX_BATCH_SIZE = 20
    
    # Security testing
    MAX_INPUT_LENGTH = 1000
    MALICIOUS_INPUT_PATTERNS = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'\.\./',
        r'\{\{.*?\}\}',
        r'\$\{.*?\}'
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "monitoring: marks tests as monitoring tests")
    config.addinivalue_line("markers", "load: marks tests as load tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "example: marks tests as example tests")
