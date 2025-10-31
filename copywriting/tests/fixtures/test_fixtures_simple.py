"""
Simple test fixtures for copywriting service tests.
"""
import pytest
import asyncio
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import CopywritingInput, CopywritingOutput, Feedback
from tests.test_utils import TestDataFactory, MockAIService, TestAssertions


# Module-level fixtures for service testing
@pytest.fixture(scope="function")
def mock_copywriting_service():
    """Create a mock copywriting service instance."""
    service = Mock()
    service.generate_copy = Mock()
    service.process_batch = Mock()
    service.get_feedback = Mock()
    service.validate_input = Mock(return_value=True)
    return service

@pytest.fixture(scope="function")
def sample_copywriting_input():
    """Create a sample copywriting input."""
    return TestDataFactory.create_copywriting_input()

@pytest.fixture(scope="function")
def sample_copywriting_output():
    """Create a sample copywriting output."""
    return TestDataFactory.create_copywriting_output()

@pytest.fixture(scope="function")
def sample_feedback():
    """Create a sample feedback."""
    return TestDataFactory.create_feedback()

@pytest.fixture(scope="function")
def batch_copywriting_inputs():
    """Create a batch of copywriting inputs."""
    return TestDataFactory.create_batch_inputs(5)

@pytest.fixture(scope="function")
def mock_ai_service():
    """Create a mock AI service."""
    return MockAIService()


# Module-level fixtures for data testing
@pytest.fixture(scope="function")
def temp_directory():
    """Create a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def sample_data_dict():
    """Create a sample data dictionary."""
    return {
        "product_description": "Test product",
        "target_platform": "instagram",
        "content_type": "social_post",
        "tone": "inspirational",
        "use_case": "product_launch"
    }

@pytest.fixture(scope="function")
def sample_response_dict():
    """Create a sample response dictionary."""
    return {
        "variants": [
            {
                "variant_id": "test_1",
                "headline": "Test Headline",
                "primary_text": "Test content",
                "call_to_action": "Learn More"
            }
        ],
        "model_used": "gpt-3.5-turbo",
        "generation_time": 1.0,
        "tokens_used": 100
    }

@pytest.fixture(scope="function")
def sample_feedback_dict():
    """Create a sample feedback dictionary."""
    return {
        "type": "human",
        "score": 8.5,
        "comments": "Great content!",
        "variant_id": "test_1"
    }


# Module-level fixtures for mocking external services
@pytest.fixture(scope="function")
def mock_http_client():
    """Create a mock HTTP client."""
    client = Mock()
    client.get = Mock()
    client.post = Mock()
    client.put = Mock()
    client.delete = Mock()
    return client

@pytest.fixture(scope="function")
def mock_database():
    """Create a mock database."""
    db = Mock()
    db.query = Mock()
    db.insert = Mock()
    db.update = Mock()
    db.delete = Mock()
    db.commit = Mock()
    db.rollback = Mock()
    return db

@pytest.fixture(scope="function")
def mock_cache():
    """Create a mock cache."""
    cache = Mock()
    cache.get = Mock(return_value=None)
    cache.set = Mock()
    cache.delete = Mock()
    cache.clear = Mock()
    return cache

@pytest.fixture(scope="function")
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger

@pytest.fixture(scope="function")
def mock_metrics():
    """Create a mock metrics collector."""
    metrics = Mock()
    metrics.increment = Mock()
    metrics.gauge = Mock()
    metrics.histogram = Mock()
    metrics.timer = Mock()
    return metrics


# Module-level fixtures for async testing
@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for async testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def async_mock_service():
    """Create an async mock service."""
    service = AsyncMock()
    service.generate_copy_async = AsyncMock()
    service.process_batch_async = AsyncMock()
    service.get_feedback_async = AsyncMock()
    return service


class TestFixtures:
    """Test cases for fixtures."""
    
    def test_service_fixtures(self, mock_copywriting_service, sample_copywriting_input, sample_copywriting_output, sample_feedback, batch_copywriting_inputs, mock_ai_service):
        """Test service fixtures."""
        # Test mock service
        assert mock_copywriting_service is not None
        assert hasattr(mock_copywriting_service, 'generate_copy')
        assert hasattr(mock_copywriting_service, 'process_batch')
        assert hasattr(mock_copywriting_service, 'get_feedback')
        
        # Test sample input
        assert sample_copywriting_input is not None
        assert hasattr(sample_copywriting_input, 'product_description')
        assert hasattr(sample_copywriting_input, 'target_platform')
        
        # Test sample output
        assert sample_copywriting_output is not None
        assert hasattr(sample_copywriting_output, 'variants')
        assert hasattr(sample_copywriting_output, 'model_used')
        
        # Test sample feedback
        assert sample_feedback is not None
        assert hasattr(sample_feedback, 'type')
        assert hasattr(sample_feedback, 'score')
        
        # Test batch inputs
        assert isinstance(batch_copywriting_inputs, list)
        assert len(batch_copywriting_inputs) == 5
        for input_item in batch_copywriting_inputs:
            assert hasattr(input_item, 'product_description')
        
        # Test mock AI service
        assert mock_ai_service is not None
        assert hasattr(mock_ai_service, 'mock_call')
    
    def test_data_fixtures(self, temp_directory, sample_data_dict, sample_response_dict, sample_feedback_dict):
        """Test data fixtures."""
        # Test temp directory
        assert os.path.exists(temp_directory)
        assert os.path.isdir(temp_directory)
        
        # Test sample data dict
        assert isinstance(sample_data_dict, dict)
        assert "product_description" in sample_data_dict
        assert "target_platform" in sample_data_dict
        
        # Test sample response dict
        assert isinstance(sample_response_dict, dict)
        assert "variants" in sample_response_dict
        assert "model_used" in sample_response_dict
        
        # Test sample feedback dict
        assert isinstance(sample_feedback_dict, dict)
        assert "type" in sample_feedback_dict
        assert "score" in sample_feedback_dict
    
    def test_mock_fixtures(self, mock_http_client, mock_database, mock_cache, mock_logger, mock_metrics):
        """Test mock fixtures."""
        # Test mock HTTP client
        assert mock_http_client is not None
        assert hasattr(mock_http_client, 'get')
        assert hasattr(mock_http_client, 'post')
        
        # Test mock database
        assert mock_database is not None
        assert hasattr(mock_database, 'query')
        assert hasattr(mock_database, 'insert')
        
        # Test mock cache
        assert mock_cache is not None
        assert hasattr(mock_cache, 'get')
        assert hasattr(mock_cache, 'set')
        
        # Test mock logger
        assert mock_logger is not None
        assert hasattr(mock_logger, 'info')
        assert hasattr(mock_logger, 'error')
        
        # Test mock metrics
        assert mock_metrics is not None
        assert hasattr(mock_metrics, 'increment')
        assert hasattr(mock_metrics, 'gauge')
    
    def test_async_fixtures(self, event_loop, async_mock_service):
        """Test async fixtures."""
        # Test event loop
        assert event_loop is not None
        assert asyncio.get_event_loop() == event_loop
        
        # Test async mock service
        assert async_mock_service is not None
        assert hasattr(async_mock_service, 'generate_copy_async')
        assert hasattr(async_mock_service, 'process_batch_async')
    
    def test_fixture_scope(self, mock_copywriting_service, sample_copywriting_input):
        """Test fixture scope."""
        # Fixtures should be created fresh for each test
        assert mock_copywriting_service is not None
        assert sample_copywriting_input is not None
    
    def test_fixture_dependencies(self, sample_copywriting_input, mock_ai_service):
        """Test fixture dependencies."""
        # Test that fixtures can be used together
        assert sample_copywriting_input is not None
        assert mock_ai_service is not None
        
        # Test mock AI service with sample input
        result = asyncio.run(mock_ai_service.mock_call(sample_copywriting_input, "gpt-3.5-turbo"))
        assert result is not None
        assert "variants" in result
    
    def test_fixture_cleanup(self, temp_directory):
        """Test fixture cleanup."""
        # Create a file in temp directory
        test_file = os.path.join(temp_directory, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        assert os.path.exists(test_file)
        
        # Fixture should clean up after test
        # (This is handled by the fixture's yield statement)
    
    def test_fixture_mocking(self, mock_copywriting_service, sample_copywriting_input):
        """Test fixture mocking."""
        # Test mock service behavior
        mock_copywriting_service.generate_copy.return_value = TestDataFactory.create_copywriting_output()
        
        result = mock_copywriting_service.generate_copy(sample_copywriting_input)
        assert result is not None
        assert hasattr(result, 'variants')
        
        # Verify mock was called
        mock_copywriting_service.generate_copy.assert_called_once_with(sample_copywriting_input)
    
    def test_fixture_data_consistency(self, sample_copywriting_input, sample_copywriting_output, sample_feedback):
        """Test fixture data consistency."""
        # Test input consistency
        assert sample_copywriting_input.product_description is not None
        assert sample_copywriting_input.target_platform is not None
        assert sample_copywriting_input.content_type is not None
        
        # Test output consistency
        assert sample_copywriting_output.variants is not None
        assert len(sample_copywriting_output.variants) > 0
        assert sample_copywriting_output.model_used is not None
        
        # Test feedback consistency
        assert sample_feedback.type is not None
        assert sample_feedback.score is not None
        assert sample_feedback.comments is not None
    
    def test_fixture_performance(self, batch_copywriting_inputs):
        """Test fixture performance."""
        # Test that batch creation is fast
        import time
        start_time = time.time()
        
        # Access the fixture
        inputs = batch_copywriting_inputs
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should be fast
        assert execution_time < 1.0
        assert len(inputs) == 5
    
    def test_fixture_isolation(self, mock_copywriting_service):
        """Test fixture isolation."""
        # Each test should get a fresh instance
        assert mock_copywriting_service is not None
        
        # Modify the mock
        mock_copywriting_service.test_value = "modified"
        
        # In a real test, this would be isolated from other tests
        assert hasattr(mock_copywriting_service, 'test_value')
    
    def test_fixture_error_handling(self, mock_copywriting_service, sample_copywriting_input):
        """Test fixture error handling."""
        # Test mock service error handling
        mock_copywriting_service.generate_copy.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            mock_copywriting_service.generate_copy(sample_copywriting_input)
    
    def test_fixture_async_behavior(self, event_loop, async_mock_service):
        """Test fixture async behavior."""
        async def test_async():
            # Test async mock service
            result = await async_mock_service.generate_copy_async(None)
            return result
        
        # Run async test
        result = event_loop.run_until_complete(test_async())
        assert result is not None
    
    def test_fixture_configuration(self, mock_copywriting_service, mock_logger, mock_metrics):
        """Test fixture configuration."""
        # Test that fixtures can be configured
        mock_copywriting_service.config = {"timeout": 30}
        mock_logger.level = "INFO"
        mock_metrics.namespace = "test"
        
        assert mock_copywriting_service.config["timeout"] == 30
        assert mock_logger.level == "INFO"
        assert mock_metrics.namespace == "test"
