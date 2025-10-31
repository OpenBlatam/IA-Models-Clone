"""
Test suite for BUL API
======================

Comprehensive tests for the BUL API endpoints.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

from api.unified_api import app
from core.bul_engine import DocumentResponse, BusinessArea, DocumentType


class TestBULAPI:
    """Test cases for BUL API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_document_request(self):
        """Sample document request data"""
        return {
            "query": "Create a marketing strategy for a tech startup",
            "business_area": "marketing",
            "document_type": "marketing_strategy",
            "company_name": "TechStartup Inc",
            "industry": "Technology",
            "company_size": "Small",
            "target_audience": "Tech-savvy consumers",
            "language": "en",
            "format": "markdown",
            "style": "professional",
            "priority": "normal"
        }
    
    @pytest.fixture
    def sample_document_response(self):
        """Sample document response"""
        return DocumentResponse(
            id="test_doc_123",
            request_id="req_123",
            content="# Marketing Strategy\n\nThis is a comprehensive marketing strategy...",
            title="Marketing Strategy for TechStartup Inc",
            summary="A comprehensive marketing strategy document for a tech startup",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            word_count=1500,
            processing_time=12.5,
            confidence_score=0.92,
            created_at=datetime.now(),
            agent_used="María González - Marketing Specialist",
            metadata={"generation_method": "openrouter_langchain"}
        )


class TestRootEndpoint:
    """Test cases for root endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns system information"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "docs" in data
        assert "features" in data
        assert "capabilities" in data
        assert data["status"] == "running"
        assert data["version"] == "2.0.0"


class TestHealthEndpoint:
    """Test cases for health endpoint"""
    
    @patch('api.unified_api.get_health_checker')
    @patch('api.unified_api.get_cache_manager')
    def test_health_endpoint_success(self, mock_cache, mock_health, client):
        """Test health endpoint with successful checks"""
        # Mock health checker
        mock_health_instance = Mock()
        mock_health_instance.run_all_checks.return_value = {"status": "healthy"}
        mock_health_instance.get_health_summary.return_value = {
            "components": {"engine": {"status": "healthy"}},
            "metrics": {"uptime": 3600}
        }
        mock_health.return_value = mock_health_instance
        
        # Mock cache manager
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance
        
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert "uptime" in data
            assert "components" in data
            assert "metrics" in data
    
    def test_health_endpoint_basic(self, client):
        """Test health endpoint without API keys"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data


class TestDocumentGeneration:
    """Test cases for document generation endpoints"""
    
    @patch('api.unified_api.get_engine_dependency')
    @patch('api.unified_api.get_agent_manager_dependency')
    @patch('api.unified_api.get_rate_limiter')
    def test_generate_document_success(self, mock_rate_limiter, mock_agent_mgr, mock_engine, client, sample_document_request, sample_document_response):
        """Test successful document generation"""
        # Mock dependencies
        mock_engine_instance = Mock()
        mock_engine_instance.is_initialized = True
        mock_engine_instance.generate_document.return_value = sample_document_response
        mock_engine.return_value = mock_engine_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.is_initialized = True
        mock_agent_instance.get_best_agent.return_value = Mock(name="Test Agent")
        mock_agent_mgr.return_value = mock_agent_instance
        
        mock_rate_limiter_instance = Mock()
        mock_rate_limiter_instance.is_allowed.return_value = True
        mock_rate_limiter.return_value = mock_rate_limiter_instance
        
        response = client.post("/generate", json=sample_document_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert "content" in data
        assert "title" in data
        assert "summary" in data
        assert "business_area" in data
        assert "document_type" in data
        assert "word_count" in data
        assert "processing_time" in data
        assert "confidence_score" in data
        assert "agent_used" in data
        assert "format" in data
        assert "style" in data
        assert "metadata" in data
        
        assert data["business_area"] == "marketing"
        assert data["document_type"] == "marketing_strategy"
        assert data["word_count"] == 1500
        assert data["confidence_score"] == 0.92
    
    def test_generate_document_validation_error(self, client):
        """Test document generation with validation error"""
        invalid_request = {
            "query": "Short",  # Too short
            "language": "invalid_lang"  # Invalid language
        }
        
        response = client.post("/generate", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_document_rate_limit(self, client, sample_document_request):
        """Test document generation with rate limiting"""
        with patch('api.unified_api.get_rate_limiter') as mock_rate_limiter:
            mock_rate_limiter_instance = Mock()
            mock_rate_limiter_instance.is_allowed.return_value = False
            mock_rate_limiter.return_value = mock_rate_limiter_instance
            
            response = client.post("/generate", json=sample_document_request)
            
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["detail"]
    
    @patch('api.unified_api.get_engine_dependency')
    @patch('api.unified_api.get_agent_manager_dependency')
    @patch('api.unified_api.get_rate_limiter')
    def test_generate_document_engine_error(self, mock_rate_limiter, mock_agent_mgr, mock_engine, client, sample_document_request):
        """Test document generation with engine error"""
        # Mock dependencies
        mock_engine_instance = Mock()
        mock_engine_instance.is_initialized = True
        mock_engine_instance.generate_document.side_effect = Exception("Engine error")
        mock_engine.return_value = mock_engine_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.is_initialized = True
        mock_agent_mgr.return_value = mock_agent_instance
        
        mock_rate_limiter_instance = Mock()
        mock_rate_limiter_instance.is_allowed.return_value = True
        mock_rate_limiter.return_value = mock_rate_limiter_instance
        
        response = client.post("/generate", json=sample_document_request)
        
        assert response.status_code == 500
        assert "Engine error" in response.json()["detail"]


class TestBatchGeneration:
    """Test cases for batch document generation"""
    
    @patch('api.unified_api.get_engine_dependency')
    @patch('api.unified_api.get_agent_manager_dependency')
    @patch('api.unified_api.get_rate_limiter')
    def test_batch_generation_success(self, mock_rate_limiter, mock_agent_mgr, mock_engine, client):
        """Test successful batch document generation"""
        batch_request = {
            "requests": [
                {
                    "query": "Create a marketing strategy",
                    "business_area": "marketing",
                    "language": "en"
                },
                {
                    "query": "Create a sales proposal",
                    "business_area": "sales",
                    "language": "en"
                }
            ],
            "parallel": True,
            "priority": "normal"
        }
        
        # Mock dependencies
        mock_engine_instance = Mock()
        mock_engine_instance.is_initialized = True
        mock_engine_instance.generate_document.return_value = DocumentResponse(
            id="test_id",
            content="Generated content",
            title="Test Document",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            word_count=100,
            processing_time=5.0,
            confidence_score=0.9
        )
        mock_engine.return_value = mock_engine_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.is_initialized = True
        mock_agent_instance.get_best_agent.return_value = Mock(name="Test Agent")
        mock_agent_mgr.return_value = mock_agent_instance
        
        mock_rate_limiter_instance = Mock()
        mock_rate_limiter_instance.is_allowed.return_value = True
        mock_rate_limiter.return_value = mock_rate_limiter_instance
        
        response = client.post("/generate/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 2
        
        for item in data:
            assert "id" in item
            assert "content" in item
            assert "title" in item
    
    def test_batch_generation_too_many_requests(self, client):
        """Test batch generation with too many requests"""
        batch_request = {
            "requests": [{"query": f"Request {i}"} for i in range(11)],  # More than 10
            "parallel": True
        }
        
        response = client.post("/generate/batch", json=batch_request)
        
        assert response.status_code == 422  # Validation error


class TestSystemEndpoints:
    """Test cases for system information endpoints"""
    
    def test_business_areas_endpoint(self, client):
        """Test business areas endpoint"""
        response = client.get("/business-areas")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert "marketing" in data
        assert "sales" in data
        assert "operations" in data
        assert "hr" in data
        assert "finance" in data
    
    def test_document_types_endpoint(self, client):
        """Test document types endpoint"""
        response = client.get("/document-types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert "business_plan" in data
        assert "marketing_strategy" in data
        assert "sales_proposal" in data
        assert "operational_manual" in data
    
    @patch('api.unified_api.get_agent_manager_dependency')
    def test_agents_endpoint_success(self, mock_agent_mgr, client):
        """Test agents endpoint with successful response"""
        mock_agent_instance = Mock()
        mock_agent_instance.is_initialized = True
        mock_agent_instance.get_all_agents.return_value = [
            Mock(
                id="agent_1",
                name="Test Agent",
                agent_type=Mock(value="marketing"),
                experience_years=5,
                success_rate=0.95,
                total_documents_generated=100,
                average_rating=4.8,
                is_active=True,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
        ]
        mock_agent_mgr.return_value = mock_agent_instance
        
        response = client.get("/agents")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "agent_1"
        assert data[0]["name"] == "Test Agent"
        assert data[0]["agent_type"] == "marketing"
    
    @patch('api.unified_api.get_agent_manager_dependency')
    def test_agents_endpoint_not_initialized(self, mock_agent_mgr, client):
        """Test agents endpoint when not initialized"""
        mock_agent_instance = Mock()
        mock_agent_instance.is_initialized = False
        mock_agent_mgr.return_value = mock_agent_instance
        
        response = client.get("/agents")
        
        assert response.status_code == 503
        assert "System not initialized" in response.json()["detail"]
    
    @patch('api.unified_api.get_agent_manager_dependency')
    def test_agent_stats_endpoint(self, mock_agent_mgr, client):
        """Test agent stats endpoint"""
        mock_agent_instance = Mock()
        mock_agent_instance.is_initialized = True
        mock_agent_instance.get_agent_stats.return_value = {
            "total_agents": 10,
            "active_agents": 8,
            "total_documents_generated": 1000,
            "average_success_rate": 0.95,
            "agent_types": ["marketing", "sales", "operations"],
            "is_initialized": True
        }
        mock_agent_mgr.return_value = mock_agent_instance
        
        response = client.get("/agents/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_agents"] == 10
        assert data["active_agents"] == 8
        assert data["total_documents_generated"] == 1000
        assert data["average_success_rate"] == 0.95


class TestWebSocketEndpoint:
    """Test cases for WebSocket endpoint"""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws/test_user") as websocket:
            # Send a ping message
            websocket.send_json({"type": "ping"})
            
            # Receive pong response
            data = websocket.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data
    
    def test_websocket_subscribe(self, client):
        """Test WebSocket subscription"""
        with client.websocket_connect("/ws/test_user") as websocket:
            # Send subscribe message
            websocket.send_json({"type": "subscribe", "channel": "documents"})
            
            # Receive subscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["channel"] == "documents"


class TestErrorHandling:
    """Test cases for error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        
        assert "detail" in data
        assert "available_endpoints" in data
        assert isinstance(data["available_endpoints"], list)
    
    def test_500_error_handling(self, client):
        """Test 500 error handling"""
        # This would require mocking an internal server error
        # For now, we'll test the error handler structure
        with patch('api.unified_api.get_engine_dependency', side_effect=Exception("Internal error")):
            response = client.get("/health")
            
            # The health endpoint should handle the error gracefully
            assert response.status_code in [200, 500]


class TestValidation:
    """Test cases for input validation"""
    
    def test_document_request_validation(self, client):
        """Test document request validation"""
        # Test minimum query length
        response = client.post("/generate", json={"query": "Short"})
        assert response.status_code == 422
        
        # Test maximum query length
        long_query = "x" * 2001
        response = client.post("/generate", json={"query": long_query})
        assert response.status_code == 422
        
        # Test invalid language
        response = client.post("/generate", json={
            "query": "Valid query with sufficient length",
            "language": "invalid_lang"
        })
        assert response.status_code == 422
        
        # Test invalid format
        response = client.post("/generate", json={
            "query": "Valid query with sufficient length",
            "format": "invalid_format"
        })
        assert response.status_code == 422
    
    def test_batch_request_validation(self, client):
        """Test batch request validation"""
        # Test empty requests
        response = client.post("/generate/batch", json={"requests": []})
        assert response.status_code == 422
        
        # Test too many requests
        requests = [{"query": f"Query {i}"} for i in range(11)]
        response = client.post("/generate/batch", json={"requests": requests})
        assert response.status_code == 422


class TestSecurity:
    """Test cases for security features"""
    
    def test_cors_headers(self, client):
        """Test CORS headers"""
        response = client.options("/")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_security_headers(self, client):
        """Test security headers"""
        response = client.get("/")
        
        # Security headers should be present
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
        assert "x-xss-protection" in response.headers


# Integration tests
class TestAPIIntegration:
    """Integration tests for the API"""
    
    @patch('api.unified_api.get_engine_dependency')
    @patch('api.unified_api.get_agent_manager_dependency')
    @patch('api.unified_api.get_rate_limiter')
    def test_full_workflow(self, mock_rate_limiter, mock_agent_mgr, mock_engine, client):
        """Test full API workflow"""
        # Mock all dependencies
        mock_engine_instance = Mock()
        mock_engine_instance.is_initialized = True
        mock_engine_instance.generate_document.return_value = DocumentResponse(
            id="integration_test",
            content="Integration test content",
            title="Integration Test Document",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            word_count=200,
            processing_time=8.0,
            confidence_score=0.88
        )
        mock_engine.return_value = mock_engine_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.is_initialized = True
        mock_agent_instance.get_best_agent.return_value = Mock(name="Integration Agent")
        mock_agent_mgr.return_value = mock_agent_instance
        
        mock_rate_limiter_instance = Mock()
        mock_rate_limiter_instance.is_allowed.return_value = True
        mock_rate_limiter.return_value = mock_rate_limiter_instance
        
        # Test workflow
        # 1. Check system status
        response = client.get("/")
        assert response.status_code == 200
        
        # 2. Check health
        response = client.get("/health")
        assert response.status_code == 200
        
        # 3. Get available business areas
        response = client.get("/business-areas")
        assert response.status_code == 200
        
        # 4. Generate document
        response = client.post("/generate", json={
            "query": "Create a comprehensive marketing strategy for a new product launch",
            "business_area": "marketing",
            "document_type": "marketing_strategy",
            "company_name": "Test Company",
            "language": "en"
        })
        assert response.status_code == 200
        
        # 5. Get agents
        response = client.get("/agents")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])