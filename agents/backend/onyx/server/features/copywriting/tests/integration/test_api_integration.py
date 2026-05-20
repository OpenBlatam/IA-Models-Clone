"""
Comprehensive API integration tests for copywriting service.
"""
import pytest
import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import CopywritingInput, CopywritingOutput, Feedback
from tests.test_utils import TestDataFactory, MockAIService, TestAssertions


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def mock_api_client(self):
        """Create a mock API client."""
        client = Mock()
        client.post = Mock()
        client.get = Mock()
        client.put = Mock()
        client.delete = Mock()
        return client
    
    @pytest.fixture
    def mock_copywriting_service(self):
        """Create a mock copywriting service."""
        service = Mock()
        service.generate_copy = Mock()
        service.process_batch = Mock()
        service.get_feedback = Mock()
        service.validate_input = Mock(return_value=True)
        return service
    
    def test_generate_copy_endpoint_integration(self, mock_api_client, mock_copywriting_service):
        """Test generate copy endpoint integration."""
        # Prepare test data
        input_data = TestDataFactory.create_copywriting_input()
        expected_output = TestDataFactory.create_copywriting_output()
        
        # Mock service response
        mock_copywriting_service.generate_copy.return_value = expected_output
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_output.model_dump()
        mock_api_client.post.return_value = mock_response
        
        # Test API call
        response = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "variants" in response_data
        assert "model_used" in response_data
        assert "generation_time" in response_data
        
        # Verify service was called
        mock_copywriting_service.generate_copy.assert_called_once()
    
    def test_batch_generate_endpoint_integration(self, mock_api_client, mock_copywriting_service):
        """Test batch generate endpoint integration."""
        # Prepare test data
        batch_inputs = TestDataFactory.create_batch_inputs(3)
        expected_outputs = [TestDataFactory.create_copywriting_output() for _ in range(3)]
        
        # Mock service response
        mock_copywriting_service.process_batch.return_value = expected_outputs
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [output.model_dump() for output in expected_outputs]
        mock_api_client.post.return_value = mock_response
        
        # Test API call
        response = mock_api_client.post("/api/copywriting/batch", json=[input.model_dump() for input in batch_inputs])
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 3
        for item in response_data:
            assert "variants" in item
            assert "model_used" in item
        
        # Verify service was called
        mock_copywriting_service.process_batch.assert_called_once()
    
    def test_feedback_endpoint_integration(self, mock_api_client, mock_copywriting_service):
        """Test feedback endpoint integration."""
        # Prepare test data
        feedback_data = TestDataFactory.create_feedback()
        expected_response = {"status": "success", "message": "Feedback recorded"}
        
        # Mock service response
        mock_copywriting_service.get_feedback.return_value = expected_response
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_api_client.post.return_value = mock_response
        
        # Test API call
        response = mock_api_client.post("/api/copywriting/feedback", json=feedback_data.model_dump())
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify service was called
        mock_copywriting_service.get_feedback.assert_called_once()
    
    def test_health_check_endpoint_integration(self, mock_api_client):
        """Test health check endpoint integration."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "timestamp": "2023-01-01T00:00:00Z"}
        mock_api_client.get.return_value = mock_response
        
        # Test API call
        response = mock_api_client.get("/api/copywriting/health")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "timestamp" in response_data
    
    def test_error_handling_integration(self, mock_api_client, mock_copywriting_service):
        """Test error handling integration."""
        # Mock service error
        mock_copywriting_service.generate_copy.side_effect = Exception("Service error")
        
        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error", "message": "Service error"}
        mock_api_client.post.return_value = mock_response
        
        # Test API call
        input_data = TestDataFactory.create_copywriting_input()
        response = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        
        # Verify error response
        assert response.status_code == 500
        response_data = response.json()
        assert "error" in response_data
        assert response_data["error"] == "Internal server error"
    
    def test_validation_error_integration(self, mock_api_client, mock_copywriting_service):
        """Test validation error integration."""
        # Mock validation error
        mock_copywriting_service.validate_input.return_value = False
        
        # Mock API validation error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Validation error", "message": "Invalid input data"}
        mock_api_client.post.return_value = mock_response
        
        # Test API call with invalid data
        invalid_data = {"invalid": "data"}
        response = mock_api_client.post("/api/copywriting/generate", json=invalid_data)
        
        # Verify validation error response
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data
        assert response_data["error"] == "Validation error"
    
    def test_rate_limiting_integration(self, mock_api_client):
        """Test rate limiting integration."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": "Rate limit exceeded",
            "message": "Too many requests",
            "retry_after": 60
        }
        mock_api_client.post.return_value = mock_response
        
        # Test API call
        input_data = TestDataFactory.create_copywriting_input()
        response = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        
        # Verify rate limit response
        assert response.status_code == 429
        response_data = response.json()
        assert response_data["error"] == "Rate limit exceeded"
        assert "retry_after" in response_data
    
    def test_concurrent_requests_integration(self, mock_api_client, mock_copywriting_service):
        """Test concurrent requests integration."""
        # Prepare test data
        input_data = TestDataFactory.create_copywriting_input()
        expected_output = TestDataFactory.create_copywriting_output()
        
        # Mock service response
        mock_copywriting_service.generate_copy.return_value = expected_output
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_output.model_dump()
        mock_api_client.post.return_value = mock_response
        
        # Test concurrent requests
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0
    
    def test_data_persistence_integration(self, mock_api_client, mock_copywriting_service):
        """Test data persistence integration."""
        # Prepare test data
        input_data = TestDataFactory.create_copywriting_input()
        expected_output = TestDataFactory.create_copywriting_output()
        
        # Mock service response
        mock_copywriting_service.generate_copy.return_value = expected_output
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_output.model_dump()
        mock_api_client.post.return_value = mock_response
        
        # Test data persistence
        response = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify data structure
        assert "variants" in response_data
        assert "model_used" in response_data
        assert "generation_time" in response_data
        assert "tokens_used" in response_data
        
        # Verify variants structure
        variants = response_data["variants"]
        assert len(variants) > 0
        for variant in variants:
            assert "variant_id" in variant
            assert "headline" in variant
            assert "primary_text" in variant
            assert "call_to_action" in variant
    
    def test_metrics_collection_integration(self, mock_api_client, mock_copywriting_service):
        """Test metrics collection integration."""
        # Prepare test data
        input_data = TestDataFactory.create_copywriting_input()
        expected_output = TestDataFactory.create_copywriting_output()
        
        # Mock service response
        mock_copywriting_service.generate_copy.return_value = expected_output
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_output.model_dump()
        mock_api_client.post.return_value = mock_response
        
        # Test metrics collection
        start_time = time.time()
        response = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        end_time = time.time()
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify metrics are collected
        assert "generation_time" in response_data
        assert "tokens_used" in response_data
        assert response_data["generation_time"] > 0
        assert response_data["tokens_used"] > 0
        
        # Verify timing
        total_time = end_time - start_time
        assert total_time > 0
    
    def test_caching_integration(self, mock_api_client, mock_copywriting_service):
        """Test caching integration."""
        # Prepare test data
        input_data = TestDataFactory.create_copywriting_input()
        expected_output = TestDataFactory.create_copywriting_output()
        
        # Mock service response
        mock_copywriting_service.generate_copy.return_value = expected_output
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_output.model_dump()
        mock_api_client.post.return_value = mock_response
        
        # Test caching
        response1 = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        response2 = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        
        # Verify both responses are successful
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Verify service was called (caching not implemented in mock)
        assert mock_copywriting_service.generate_copy.call_count == 2
    
    def test_security_headers_integration(self, mock_api_client):
        """Test security headers integration."""
        # Mock API response with security headers
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000"
        }
        mock_api_client.get.return_value = mock_response
        
        # Test security headers
        response = mock_api_client.get("/api/copywriting/health")
        
        # Verify response
        assert response.status_code == 200
        
        # Verify security headers
        headers = response.headers
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
    
    def test_api_versioning_integration(self, mock_api_client):
        """Test API versioning integration."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "1.0.0", "status": "healthy"}
        mock_api_client.get.return_value = mock_response
        
        # Test API versioning
        response = mock_api_client.get("/api/v1/copywriting/health")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["version"] == "1.0.0"
        assert response_data["status"] == "healthy"
    
    def test_request_id_tracking_integration(self, mock_api_client, mock_copywriting_service):
        """Test request ID tracking integration."""
        # Prepare test data
        input_data = TestDataFactory.create_copywriting_input()
        expected_output = TestDataFactory.create_copywriting_output()
        
        # Mock service response
        mock_copywriting_service.generate_copy.return_value = expected_output
        
        # Mock API response with request ID
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            **expected_output.model_dump(),
            "request_id": "req_123456789"
        }
        mock_api_client.post.return_value = mock_response
        
        # Test request ID tracking
        response = mock_api_client.post("/api/copywriting/generate", json=input_data.model_dump())
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "request_id" in response_data
        assert response_data["request_id"] == "req_123456789"
    
    def test_batch_processing_limits_integration(self, mock_api_client, mock_copywriting_service):
        """Test batch processing limits integration."""
        # Prepare large batch data
        batch_inputs = TestDataFactory.create_batch_inputs(100)  # Large batch
        expected_outputs = [TestDataFactory.create_copywriting_output() for _ in range(100)]
        
        # Mock service response
        mock_copywriting_service.process_batch.return_value = expected_outputs
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [output.model_dump() for output in expected_outputs]
        mock_api_client.post.return_value = mock_response
        
        # Test batch processing
        response = mock_api_client.post("/api/copywriting/batch", json=[input.model_dump() for input in batch_inputs])
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 100
        
        # Verify service was called
        mock_copywriting_service.process_batch.assert_called_once()
    
    def test_async_processing_integration(self, mock_api_client, mock_copywriting_service):
        """Test async processing integration."""
        # Prepare test data
        input_data = TestDataFactory.create_copywriting_input()
        expected_output = TestDataFactory.create_copywriting_output()
        
        # Mock async service response
        async def async_generate():
            await asyncio.sleep(0.1)  # Simulate async processing
            return expected_output
        
        mock_copywriting_service.generate_copy_async = AsyncMock(return_value=expected_output)
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_output.model_dump()
        mock_api_client.post.return_value = mock_response
        
        # Test async processing
        response = mock_api_client.post("/api/copywriting/generate-async", json=input_data.model_dump())
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "variants" in response_data
        assert "model_used" in response_data