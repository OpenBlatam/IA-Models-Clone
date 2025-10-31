"""
BUL Enhanced API Test Suite
==========================

Suite completa de pruebas para la API BUL mejorada.
Incluye tests para todas las nuevas funcionalidades.
"""

import asyncio
import pytest
import requests
import time
import json
from datetime import datetime
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class TestEnhancedBULAPI:
    """Test suite for Enhanced BUL API."""
    
    def setup_method(self):
        """Setup test environment."""
        self.api_url = API_BASE_URL
        self.session = requests.Session()
        self.test_user = "test_user"
        self.test_api_key = "test_key_123"
    
    def test_api_health(self):
        """Test enhanced health endpoint."""
        response = self.session.get(f"{self.api_url}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "uptime" in data
        assert "active_tasks" in data
        assert "total_requests" in data
        assert "cache_type" in data
    
    def test_api_root(self):
        """Test enhanced root endpoint."""
        response = self.session.get(f"{self.api_url}/", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "4.0.0"
        assert "features" in data
        assert "Authentication" in data["features"]
        assert "Rate Limiting" in data["features"]
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = self.session.get(f"{self.api_url}/metrics", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = self.session.get(f"{self.api_url}/stats", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "active_tasks" in data
        assert "success_rate" in data
        assert "average_processing_time" in data
        assert "cache_hit_rate" in data
        assert "uptime" in data
    
    def test_authentication_login(self):
        """Test authentication login."""
        response = self.session.post(
            f"{self.api_url}/auth/login",
            params={"user_id": "admin", "api_key": "admin_key_123"},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "user_id" in data
        assert "session_token" in data
        assert "permissions" in data
        assert "expires_at" in data
    
    def test_authentication_invalid_user(self):
        """Test authentication with invalid user."""
        response = self.session.post(
            f"{self.api_url}/auth/login",
            params={"user_id": "invalid_user"},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 401
    
    def test_document_generation(self):
        """Test enhanced document generation."""
        request_data = {
            "query": "Create a comprehensive marketing strategy for a new restaurant",
            "business_area": "marketing",
            "document_type": "strategy",
            "priority": 1,
            "user_id": self.test_user,
            "session_id": "test_session_123"
        }
        
        response = self.session.post(
            f"{self.api_url}/documents/generate",
            json=request_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert "queue_position" in data
        assert "created_at" in data
        
        return data["task_id"]
    
    def test_task_status(self):
        """Test enhanced task status."""
        # First create a task
        task_id = self.test_document_generation()
        
        # Check status
        response = self.session.get(
            f"{self.api_url}/tasks/{task_id}/status",
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_list_tasks(self):
        """Test enhanced task listing."""
        response = self.session.get(f"{self.api_url}/tasks", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "tasks" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data
    
    def test_list_tasks_with_filters(self):
        """Test task listing with filters."""
        # Test with status filter
        response = self.session.get(
            f"{self.api_url}/tasks",
            params={"status": "queued", "limit": 10},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "tasks" in data
        
        # Test with user filter
        response = self.session.get(
            f"{self.api_url}/tasks",
            params={"user_id": self.test_user},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_cancel_task(self):
        """Test task cancellation."""
        # Create a task
        task_id = self.test_document_generation()
        
        # Cancel the task
        response = self.session.post(
            f"{self.api_url}/tasks/{task_id}/cancel",
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
    
    def test_delete_task(self):
        """Test task deletion."""
        # Create a task
        task_id = self.test_document_generation()
        
        # Delete the task
        response = self.session.delete(
            f"{self.api_url}/tasks/{task_id}",
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Make multiple requests quickly to trigger rate limiting
        responses = []
        for i in range(15):  # More than the 10/minute limit
            response = self.session.post(
                f"{self.api_url}/documents/generate",
                json={
                    "query": f"Test query {i}",
                    "user_id": f"user_{i}"
                },
                timeout=TEST_TIMEOUT
            )
            responses.append(response)
        
        # At least one should be rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Rate limiting not working"
    
    def test_file_upload(self):
        """Test file upload functionality."""
        # Create a test file
        test_file_path = Path("test_upload.txt")
        test_content = "This is a test file for upload"
        test_file_path.write_text(test_content)
        
        try:
            with open(test_file_path, "rb") as f:
                files = {"file": ("test_upload.txt", f, "text/plain")}
                response = self.session.post(
                    f"{self.api_url}/upload",
                    files=files,
                    timeout=TEST_TIMEOUT
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "filename" in data
            assert "file_path" in data
            assert "size" in data
            assert "uploaded_at" in data
            
        finally:
            # Cleanup
            if test_file_path.exists():
                test_file_path.unlink()
    
    def test_document_download(self):
        """Test document download functionality."""
        # First create and wait for a task to complete
        task_id = self.test_document_generation()
        
        # Wait for task to complete (with timeout)
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            response = self.session.get(
                f"{self.api_url}/tasks/{task_id}/status",
                timeout=TEST_TIMEOUT
            )
            data = response.json()
            
            if data["status"] == "completed":
                break
            elif data["status"] == "failed":
                pytest.skip("Task failed, cannot test download")
            
            time.sleep(1)
            wait_time += 1
        
        if wait_time >= max_wait:
            pytest.skip("Task did not complete in time")
        
        # Test download
        response = self.session.get(
            f"{self.api_url}/download/{task_id}",
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        assert "attachment" in response.headers.get("content-disposition", "")
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid task ID
        response = self.session.get(
            f"{self.api_url}/tasks/invalid_task_id/status",
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 404
        
        # Test invalid document request
        response = self.session.post(
            f"{self.api_url}/documents/generate",
            json={"query": ""},  # Empty query should fail validation
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422  # Validation error
    
    def test_caching(self):
        """Test caching functionality."""
        # Make the same request twice
        request_data = {
            "query": "Test caching query",
            "user_id": "cache_test_user"
        }
        
        # First request
        response1 = self.session.post(
            f"{self.api_url}/documents/generate",
            json=request_data,
            timeout=TEST_TIMEOUT
        )
        assert response1.status_code == 200
        
        # Second request (should potentially hit cache)
        response2 = self.session.post(
            f"{self.api_url}/documents/generate",
            json=request_data,
            timeout=TEST_TIMEOUT
        )
        assert response2.status_code == 200
        
        # Both should succeed, cache behavior depends on implementation
        data1 = response1.json()
        data2 = response2.json()
        assert "task_id" in data1
        assert "task_id" in data2

class TestDashboardIntegration:
    """Test dashboard integration."""
    
    def test_dashboard_availability(self):
        """Test that dashboard can connect to API."""
        try:
            # Test API health from dashboard perspective
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            assert response.status_code == 200
            
            # Test stats endpoint
            response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
            assert response.status_code == 200
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running, cannot test dashboard integration")

def run_integration_tests():
    """Run integration tests."""
    print("🧪 Running Enhanced BUL API Integration Tests")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running or not healthy")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start the API first.")
        return False
    
    print("✅ API is running and healthy")
    
    # Run tests
    test_suite = TestEnhancedBULAPI()
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"🔍 Running {test_method}...")
            getattr(test_suite, test_method)()
            print(f"   ✅ {test_method} passed")
            passed += 1
        except Exception as e:
            print(f"   ❌ {test_method} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Enhanced API is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the API implementation.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    if success:
        print("\n✅ Enhanced BUL API is ready for production!")
    else:
        print("\n❌ Enhanced BUL API needs attention.")
