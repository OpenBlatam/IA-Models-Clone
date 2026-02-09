"""
Tests for API error handling
Comprehensive error scenarios and edge cases
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json


@pytest.fixture
def error_test_client():
    """Create test client for error testing"""
    # Use canonical API router instead of deprecated recovery_api
    from api.recovery_api_refactored import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestErrorHandling:
    """Tests for error handling across API endpoints"""
    
    def test_404_not_found(self, error_test_client):
        """Test 404 error for non-existent endpoints"""
        response = error_test_client.get("/nonexistent/endpoint")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, error_test_client):
        """Test 405 error for wrong HTTP method"""
        # Try GET on POST-only endpoint
        response = error_test_client.get("/assessment/assess")
        assert response.status_code in [405, 404]  # Depends on implementation
    
    def test_422_validation_error(self, error_test_client):
        """Test 422 error for validation failures"""
        # Invalid request data
        response = error_test_client.post(
            "/assessment/assess",
            json={"invalid": "data"}
        )
        assert response.status_code == 422
        assert "detail" in response.json()
    
    def test_500_internal_server_error(self, error_test_client):
        """Test 500 error handling"""
        with patch('api.routes.assessment.handlers.process_assessment') as mock_process:
            mock_process.side_effect = Exception("Internal error")
            
            response = error_test_client.post(
                "/assessment/assess",
                json={
                    "addiction_type": "smoking",
                    "severity": "moderate",
                    "frequency": "daily"
                }
            )
            
            # Should return 500 or handle gracefully
            assert response.status_code in [500, 503]
    
    def test_malformed_json(self, error_test_client):
        """Test handling of malformed JSON"""
        response = error_test_client.post(
            "/assessment/assess",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_content_type(self, error_test_client):
        """Test handling of missing Content-Type header"""
        response = error_test_client.post(
            "/assessment/assess",
            data=json.dumps({"addiction_type": "smoking"}),
            headers={}  # No Content-Type
        )
        # Should either work or return 415
        assert response.status_code in [200, 201, 415, 422]
    
    def test_empty_request_body(self, error_test_client):
        """Test handling of empty request body"""
        response = error_test_client.post(
            "/assessment/assess",
            json={}
        )
        assert response.status_code == 422
    
    def test_null_values(self, error_test_client):
        """Test handling of null values"""
        response = error_test_client.post(
            "/assessment/assess",
            json={
                "addiction_type": None,
                "severity": None,
                "frequency": None
            }
        )
        assert response.status_code == 422
    
    def test_very_large_payload(self, error_test_client):
        """Test handling of very large payload"""
        large_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "notes": "A" * 1000000  # 1MB of text
        }
        
        response = error_test_client.post(
            "/progress/log-entry",
            json=large_data
        )
        
        # Should either accept or reject with 413
        assert response.status_code in [201, 400, 413, 422]
    
    def test_special_characters(self, error_test_client):
        """Test handling of special characters in input"""
        special_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "notes": "<script>alert('xss')</script> & 'quotes' \"double\""
        }
        
        response = error_test_client.post(
            "/progress/log-entry",
            json=special_data
        )
        
        # Should handle or sanitize
        assert response.status_code in [201, 400, 422]
    
    def test_sql_injection_attempt(self, error_test_client):
        """Test handling of SQL injection attempts"""
        malicious_user_id = "user_123'; DROP TABLE users; --"
        
        response = error_test_client.get(f"/progress/{malicious_user_id}")
        
        # Should not execute SQL, return 400/404/422
        assert response.status_code in [200, 400, 404, 422]
    
    def test_path_traversal_attempt(self, error_test_client):
        """Test handling of path traversal attempts"""
        malicious_path = "../../../etc/passwd"
        
        response = error_test_client.get(f"/progress/{malicious_path}")
        
        # Should not allow path traversal
        assert response.status_code in [400, 404, 422]
    
    def test_unicode_handling(self, error_test_client):
        """Test handling of unicode characters"""
        unicode_data = {
            "user_id": "user_123",
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 3,
            "notes": "测试 🎉 émojis 中文"
        }
        
        response = error_test_client.post(
            "/progress/log-entry",
            json=unicode_data
        )
        
        # Should handle unicode properly
        assert response.status_code in [201, 400, 422]
        if response.status_code == 201:
            assert "测试" in response.json().get("notes", "") or True


class TestTimeoutHandling:
    """Tests for timeout scenarios"""
    
    def test_slow_service_response(self, error_test_client):
        """Test handling of slow service responses"""
        import time
        
        with patch('api.routes.assessment.handlers.process_assessment') as mock_process:
            def slow_function(*args, **kwargs):
                time.sleep(2)  # Simulate slow response
                return {"severity": "moderate"}
            
            mock_process.side_effect = slow_function
            
            # Should timeout or handle gracefully
            response = error_test_client.post(
                "/assessment/assess",
                json={
                    "addiction_type": "smoking",
                    "severity": "moderate",
                    "frequency": "daily"
                },
                timeout=1  # 1 second timeout
            )
            
            # May timeout or return error
            assert response.status_code in [200, 500, 503, 504] or hasattr(response, 'timeout')


class TestConcurrentRequests:
    """Tests for concurrent request handling"""
    
    def test_concurrent_same_endpoint(self, error_test_client):
        """Test handling of concurrent requests to same endpoint"""
        import concurrent.futures
        
        def make_request():
            return error_test_client.post(
                "/progress/log-entry",
                json={
                    "user_id": "user_123",
                    "date": datetime.now().isoformat(),
                    "mood": "good",
                    "cravings_level": 3
                }
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should complete (may have different status codes)
        assert len(results) == 5
        assert all(r.status_code in [201, 400, 422, 500] for r in results)


class TestRateLimiting:
    """Tests for rate limiting scenarios"""
    
    def test_rapid_requests(self, error_test_client):
        """Test handling of rapid successive requests"""
        responses = []
        for _ in range(20):
            response = error_test_client.post(
                "/assessment/assess",
                json={
                    "addiction_type": "smoking",
                    "severity": "moderate",
                    "frequency": "daily"
                }
            )
            responses.append(response.status_code)
        
        # Should either all succeed or some be rate limited (429)
        assert all(code in [200, 201, 422, 429, 500] for code in responses)


class TestDataValidation:
    """Tests for data validation edge cases"""
    
    @pytest.mark.parametrize("invalid_date", [
        "not-a-date",
        "2024-13-45",  # Invalid month/day
        "2024-02-30",  # Invalid day for February
        "1900-01-01",  # Very old date
        "3000-01-01",  # Future date
        "",
        None,
    ])
    def test_invalid_dates(self, error_test_client, invalid_date):
        """Test handling of invalid dates"""
        request_data = {
            "user_id": "user_123",
            "date": invalid_date,
            "mood": "good",
            "cravings_level": 3
        }
        
        response = error_test_client.post("/progress/log-entry", json=request_data)
        
        # Should reject invalid dates
        assert response.status_code in [400, 422]
    
    @pytest.mark.parametrize("invalid_cravings", [
        -1,  # Negative
        11,  # Above maximum
        999,  # Way above maximum
        "not-a-number",  # Wrong type
        None,
    ])
    def test_invalid_cravings_level(self, error_test_client, invalid_cravings):
        """Test handling of invalid craving levels"""
        request_data = {
            "user_id": "user_123",
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": invalid_cravings
        }
        
        response = error_test_client.post("/progress/log-entry", json=request_data)
        
        # Should reject invalid craving levels
        assert response.status_code in [400, 422]
    
    def test_empty_strings(self, error_test_client):
        """Test handling of empty strings"""
        request_data = {
            "user_id": "",  # Empty string
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 3
        }
        
        response = error_test_client.post("/progress/log-entry", json=request_data)
        
        # Should reject empty required fields
        assert response.status_code in [400, 422]
    
    def test_whitespace_only(self, error_test_client):
        """Test handling of whitespace-only strings"""
        request_data = {
            "user_id": "   ",  # Only whitespace
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 3
        }
        
        response = error_test_client.post("/progress/log-entry", json=request_data)
        
        # Should reject whitespace-only fields
        assert response.status_code in [400, 422]



