"""
AI Integration System - API Endpoints Tests
Test suite for REST API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import json

from ..main import app
from ..integration_engine import IntegrationStatus, ContentType

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def sample_integration_request():
    """Sample integration request data"""
    return {
        "content_id": "test_blog_001",
        "content_type": "blog_post",
        "content_data": {
            "title": "Test Blog Post",
            "content": "This is a test blog post content.",
            "author": "Test Author",
            "tags": ["test", "blog", "api"]
        },
        "target_platforms": ["wordpress", "hubspot"],
        "priority": 1,
        "max_retries": 3
    }

@pytest.fixture
def sample_bulk_request():
    """Sample bulk integration request data"""
    return {
        "requests": [
            {
                "content_id": "bulk_001",
                "content_type": "blog_post",
                "content_data": {
                    "title": "Bulk Test 1",
                    "content": "Content 1",
                    "author": "Test Author"
                },
                "target_platforms": ["wordpress"],
                "priority": 1
            },
            {
                "content_id": "bulk_002",
                "content_type": "email_campaign",
                "content_data": {
                    "title": "Bulk Test 2",
                    "content": "Content 2",
                    "author": "Test Author"
                },
                "target_platforms": ["mailchimp"],
                "priority": 2
            }
        ],
        "batch_id": "test_batch_001"
    }

class TestRootEndpoints:
    """Test root and health endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "AI Integration System"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "docs_url" in data
        assert "health_url" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"

class TestIntegrationEndpoints:
    """Test integration endpoints"""
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_create_integration_success(self, mock_engine, client, sample_integration_request):
        """Test successful integration creation"""
        mock_engine.add_integration_request = AsyncMock()
        mock_engine.process_single_request = AsyncMock()
        
        response = client.post("/ai-integration/integrate", json=sample_integration_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Integration request created successfully"
        assert data["content_id"] == sample_integration_request["content_id"]
        assert data["status"] == "queued"
    
    def test_create_integration_invalid_content_type(self, client, sample_integration_request):
        """Test integration creation with invalid content type"""
        sample_integration_request["content_type"] = "invalid_type"
        
        response = client.post("/ai-integration/integrate", json=sample_integration_request)
        assert response.status_code == 400
        
        data = response.json()
        assert "Invalid content type" in data["detail"]
    
    def test_create_integration_missing_fields(self, client):
        """Test integration creation with missing required fields"""
        incomplete_request = {
            "content_id": "test_001",
            "content_type": "blog_post"
            # Missing content_data and target_platforms
        }
        
        response = client.post("/ai-integration/integrate", json=incomplete_request)
        assert response.status_code == 422  # Validation error
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_create_bulk_integration(self, mock_engine, client, sample_bulk_request):
        """Test bulk integration creation"""
        mock_engine.add_integration_request = AsyncMock()
        mock_engine.process_integration_queue = AsyncMock()
        
        response = client.post("/ai-integration/integrate/bulk", json=sample_bulk_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Bulk integration request created successfully"
        assert data["processed_requests"] == 2
        assert data["total_requests"] == 2
        assert data["status"] == "queued"
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_get_integration_status_success(self, mock_engine, client):
        """Test getting integration status"""
        mock_status = {
            "content_id": "test_001",
            "overall_status": "completed",
            "results": [
                {
                    "platform": "wordpress",
                    "status": "completed",
                    "external_id": "wp_123"
                }
            ]
        }
        mock_engine.get_integration_status = AsyncMock(return_value=mock_status)
        
        response = client.get("/ai-integration/status/test_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["content_id"] == "test_001"
        assert data["status"] == "completed"
        assert len(data["results"]) == 1
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_get_integration_status_not_found(self, mock_engine, client):
        """Test getting status for non-existent integration"""
        mock_engine.get_integration_status = AsyncMock(return_value={"status": "not_found"})
        
        response = client.get("/ai-integration/status/non_existent")
        assert response.status_code == 404
        
        data = response.json()
        assert "Integration not found" in data["detail"]
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_get_all_integration_status(self, mock_engine, client):
        """Test getting all integration statuses"""
        mock_engine.results = {
            "test_001": [],
            "test_002": []
        }
        mock_engine.get_integration_status = AsyncMock(side_effect=[
            {"overall_status": "completed", "results": []},
            {"overall_status": "failed", "results": []}
        ])
        
        response = client.get("/ai-integration/status")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        assert data[0]["content_id"] == "test_001"
        assert data[1]["content_id"] == "test_002"
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_get_available_platforms(self, mock_engine, client):
        """Test getting available platforms"""
        mock_engine.get_available_platforms.return_value = ["wordpress", "hubspot", "mailchimp"]
        
        response = client.get("/ai-integration/platforms")
        assert response.status_code == 200
        
        data = response.json()
        assert "wordpress" in data
        assert "hubspot" in data
        assert "mailchimp" in data
        assert len(data) == 3
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_test_platform_connection_success(self, mock_engine, client):
        """Test successful platform connection test"""
        mock_engine.test_connection = AsyncMock(return_value=True)
        
        response = client.post("/ai-integration/platforms/wordpress/test")
        assert response.status_code == 200
        
        data = response.json()
        assert data["platform"] == "wordpress"
        assert data["status"] == True
        assert data["message"] == "Connection successful"
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_test_platform_connection_failure(self, mock_engine, client):
        """Test failed platform connection test"""
        mock_engine.test_connection = AsyncMock(return_value=False)
        
        response = client.post("/ai-integration/platforms/invalid/test")
        assert response.status_code == 200
        
        data = response.json()
        assert data["platform"] == "invalid"
        assert data["status"] == False
        assert data["message"] == "Connection failed"
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_get_queue_status(self, mock_engine, client):
        """Test getting queue status"""
        mock_engine.integration_queue = [
            Mock(content_id="test_001", content_type=ContentType.BLOG_POST, 
                 target_platforms=["wordpress"], priority=1, retry_count=0),
            Mock(content_id="test_002", content_type=ContentType.EMAIL_CAMPAIGN,
                 target_platforms=["mailchimp"], priority=2, retry_count=1)
        ]
        mock_engine.is_running = True
        mock_engine.results = {"test_001": [], "test_002": []}
        
        response = client.get("/ai-integration/queue/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["queue_length"] == 2
        assert data["engine_running"] == True
        assert data["total_results"] == 2
        assert len(data["pending_requests"]) == 2
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_process_queue(self, mock_engine, client):
        """Test manual queue processing"""
        mock_engine.process_integration_queue = AsyncMock()
        
        response = client.post("/ai-integration/queue/process")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Queue processing started"
        assert data["status"] == "processing"
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_delete_integration_status(self, mock_engine, client):
        """Test deleting integration status"""
        mock_engine.results = {"test_001": []}
        
        response = client.delete("/ai-integration/status/test_001")
        assert response.status_code == 200
        
        data = response.json()
        assert "Integration status deleted" in data["message"]
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_delete_integration_status_not_found(self, mock_engine, client):
        """Test deleting non-existent integration status"""
        mock_engine.results = {}
        
        response = client.delete("/ai-integration/status/non_existent")
        assert response.status_code == 404
        
        data = response.json()
        assert "Integration not found" in data["detail"]

class TestWebhookEndpoints:
    """Test webhook endpoints"""
    
    def test_handle_platform_webhook_success(self, client):
        """Test successful webhook handling"""
        webhook_payload = {
            "event_type": "content.created",
            "object_id": "12345",
            "platform": "salesforce",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        response = client.post("/ai-integration/webhooks/salesforce", json=webhook_payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Webhook processed for salesforce"
        assert data["status"] == "success"
    
    def test_handle_platform_webhook_error(self, client):
        """Test webhook handling with error"""
        # Send invalid payload to trigger error
        response = client.post("/ai-integration/webhooks/invalid", json={})
        assert response.status_code == 200  # Webhook endpoint should handle errors gracefully
        
        data = response.json()
        assert "Webhook processed for invalid" in data["message"]

class TestHealthEndpoints:
    """Test health and monitoring endpoints"""
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_health_check(self, mock_engine, client):
        """Test health check endpoint"""
        mock_engine.is_running = True
        mock_engine.get_available_platforms.return_value = ["wordpress", "hubspot"]
        mock_engine.integration_queue = []
        
        response = client.get("/ai-integration/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["engine_running"] == True
        assert data["available_platforms"] == 2
        assert data["queue_length"] == 0
        assert "timestamp" in data

class TestErrorHandling:
    """Test error handling in API endpoints"""
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_internal_server_error(self, mock_engine, client, sample_integration_request):
        """Test internal server error handling"""
        mock_engine.add_integration_request.side_effect = Exception("Internal error")
        
        response = client.post("/ai-integration/integrate", json=sample_integration_request)
        assert response.status_code == 500
        
        data = response.json()
        assert "Internal server error" in data["detail"]
    
    def test_validation_error(self, client):
        """Test validation error handling"""
        invalid_request = {
            "content_id": "test",
            "content_type": "blog_post",
            "content_data": "invalid_data",  # Should be dict
            "target_platforms": "invalid_platforms"  # Should be list
        }
        
        response = client.post("/ai-integration/integrate", json=invalid_request)
        assert response.status_code == 422  # Validation error

class TestContentTypes:
    """Test different content types"""
    
    @pytest.mark.parametrize("content_type", [
        "blog_post",
        "email_campaign", 
        "social_media_post",
        "product_description",
        "landing_page",
        "document",
        "presentation"
    ])
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_valid_content_types(self, mock_engine, client, content_type):
        """Test all valid content types"""
        mock_engine.add_integration_request = AsyncMock()
        mock_engine.process_single_request = AsyncMock()
        
        request_data = {
            "content_id": f"test_{content_type}",
            "content_type": content_type,
            "content_data": {
                "title": f"Test {content_type}",
                "content": f"Content for {content_type}"
            },
            "target_platforms": ["wordpress"]
        }
        
        response = client.post("/ai-integration/integrate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "queued"

class TestPriorityHandling:
    """Test priority handling in integration requests"""
    
    @pytest.mark.parametrize("priority", [1, 5, 10])
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_valid_priorities(self, mock_engine, client, priority):
        """Test valid priority values"""
        mock_engine.add_integration_request = AsyncMock()
        mock_engine.process_single_request = AsyncMock()
        
        request_data = {
            "content_id": f"test_priority_{priority}",
            "content_type": "blog_post",
            "content_data": {
                "title": f"Test Priority {priority}",
                "content": "Test content"
            },
            "target_platforms": ["wordpress"],
            "priority": priority
        }
        
        response = client.post("/ai-integration/integrate", json=request_data)
        assert response.status_code == 200
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_invalid_priority_too_low(self, mock_engine, client):
        """Test invalid priority (too low)"""
        request_data = {
            "content_id": "test_priority_low",
            "content_type": "blog_post",
            "content_data": {
                "title": "Test Low Priority",
                "content": "Test content"
            },
            "target_platforms": ["wordpress"],
            "priority": 0  # Below minimum of 1
        }
        
        response = client.post("/ai-integration/integrate", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('ai_integration_system.api_endpoints.integration_engine')
    def test_invalid_priority_too_high(self, mock_engine, client):
        """Test invalid priority (too high)"""
        request_data = {
            "content_id": "test_priority_high",
            "content_type": "blog_post",
            "content_data": {
                "title": "Test High Priority",
                "content": "Test content"
            },
            "target_platforms": ["wordpress"],
            "priority": 11  # Above maximum of 10
        }
        
        response = client.post("/ai-integration/integrate", json=request_data)
        assert response.status_code == 422  # Validation error



























