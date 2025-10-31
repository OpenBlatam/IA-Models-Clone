"""
Comprehensive Test Suite for BUL Enhanced API
============================================

Modern testing suite with unit tests, integration tests, and performance tests.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

from fastapi.testclient import TestClient
from fastapi import status
import httpx

from api.enhanced_api import app, get_engine_dependency, get_agent_manager_dependency
from core.bul_engine import BULEngine, DocumentRequest, DocumentResponse, BusinessArea, DocumentType
from agents import SMEAgentManager
from database.enhanced_database import get_database_manager
from security.enhanced_security import get_security_manager
from utils import get_cache_manager

# Test client
client = TestClient(app)

# Mock data
MOCK_DOCUMENT_REQUEST = {
    "query": "Create a marketing strategy for a small restaurant",
    "business_area": "marketing",
    "document_type": "marketing_strategy",
    "company_name": "Restaurant ABC",
    "industry": "food_service",
    "company_size": "small",
    "target_audience": "local customers",
    "language": "es",
    "format": "markdown",
    "style": "professional",
    "priority": "normal"
}

MOCK_DOCUMENT_RESPONSE = {
    "id": "doc_123456",
    "request_id": "req_789012",
    "content": "# Marketing Strategy for Restaurant ABC\n\n## Executive Summary\n\nThis comprehensive marketing strategy...",
    "title": "Marketing Strategy for Restaurant ABC",
    "summary": "Comprehensive marketing strategy for Restaurant ABC",
    "business_area": "marketing",
    "document_type": "marketing_strategy",
    "word_count": 1500,
    "processing_time": 2.5,
    "confidence_score": 0.85,
    "created_at": datetime.now().isoformat(),
    "agent_used": "Marketing Expert Agent",
    "format": "markdown",
    "style": "professional",
    "metadata": {"generation_method": "ai", "model_used": "gpt-4"},
    "quality_score": 0.88,
    "readability_score": 0.82
}

class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "features" in data
        assert "capabilities" in data
        assert data["version"] == "3.0.0"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert "components" in data
        assert "metrics" in data
        assert "performance" in data
        assert "dependencies" in data
    
    def test_business_areas_endpoint(self):
        """Test business areas endpoint"""
        response = client.get("/business-areas")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert "marketing" in data
        assert "sales" in data
        assert "operations" in data
    
    def test_document_types_endpoint(self):
        """Test document types endpoint"""
        response = client.get("/document-types")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert "business_plan" in data
        assert "marketing_strategy" in data
        assert "sales_proposal" in data

class TestDocumentGeneration:
    """Test document generation endpoints"""
    
    @patch('api.enhanced_api.get_engine_dependency')
    @patch('api.enhanced_api.get_agent_manager_dependency')
    @patch('api.enhanced_api.get_cache_dependency')
    @patch('api.enhanced_api.get_rate_limiter_dependency')
    def test_generate_document_success(self, mock_rate_limiter, mock_cache, mock_agent_mgr, mock_engine):
        """Test successful document generation"""
        # Setup mocks
        mock_rate_limiter.return_value.is_allowed.return_value = True
        mock_cache.return_value = Mock()
        
        mock_agent = Mock()
        mock_agent.name = "Marketing Expert Agent"
        mock_agent_mgr.return_value.get_best_agent.return_value = mock_agent
        
        mock_response = Mock()
        mock_response.id = "doc_123456"
        mock_response.request_id = "req_789012"
        mock_response.content = "# Marketing Strategy\n\nContent here..."
        mock_response.title = "Marketing Strategy"
        mock_response.summary = "Strategy summary"
        mock_response.business_area = BusinessArea.MARKETING
        mock_response.document_type = DocumentType.MARKETING_STRATEGY
        mock_response.word_count = 1500
        mock_response.processing_time = 2.5
        mock_response.confidence_score = 0.85
        mock_response.created_at = datetime.now()
        mock_response.metadata = {}
        
        mock_engine.return_value.generate_document.return_value = mock_response
        
        # Make request
        response = client.post("/generate", json=MOCK_DOCUMENT_REQUEST)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "content" in data
        assert "title" in data
        assert "processing_time" in data
        assert "agent_used" in data
    
    def test_generate_document_validation_error(self):
        """Test document generation with validation error"""
        invalid_request = {
            "query": "Short",  # Too short
            "business_area": "invalid_area",
            "document_type": "invalid_type"
        }
        
        response = client.post("/generate", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_generate_document_rate_limit(self):
        """Test rate limiting"""
        with patch('api.enhanced_api.get_rate_limiter_dependency') as mock_rate_limiter:
            mock_rate_limiter.return_value.is_allowed.return_value = False
            
            response = client.post("/generate", json=MOCK_DOCUMENT_REQUEST)
            assert response.status_code == 429
    
    @patch('api.enhanced_api.get_engine_dependency')
    @patch('api.enhanced_api.get_agent_manager_dependency')
    def test_generate_document_service_unavailable(self, mock_agent_mgr, mock_engine):
        """Test service unavailable error"""
        mock_engine.return_value = None
        mock_agent_mgr.return_value = None
        
        response = client.post("/generate", json=MOCK_DOCUMENT_REQUEST)
        assert response.status_code == 503
    
    @patch('api.enhanced_api.get_engine_dependency')
    @patch('api.enhanced_api.get_agent_manager_dependency')
    def test_batch_generation_success(self, mock_agent_mgr, mock_engine):
        """Test successful batch generation"""
        # Setup mocks
        mock_agent = Mock()
        mock_agent.name = "Marketing Expert Agent"
        mock_agent_mgr.return_value.get_best_agent.return_value = mock_agent
        
        mock_response = Mock()
        mock_response.id = "doc_123456"
        mock_response.request_id = "req_789012"
        mock_response.content = "# Marketing Strategy\n\nContent here..."
        mock_response.title = "Marketing Strategy"
        mock_response.summary = "Strategy summary"
        mock_response.business_area = BusinessArea.MARKETING
        mock_response.document_type = DocumentType.MARKETING_STRATEGY
        mock_response.word_count = 1500
        mock_response.processing_time = 2.5
        mock_response.confidence_score = 0.85
        mock_response.created_at = datetime.now()
        mock_response.metadata = {}
        
        mock_engine.return_value.generate_document.return_value = mock_response
        
        # Make batch request
        batch_request = {
            "requests": [MOCK_DOCUMENT_REQUEST, MOCK_DOCUMENT_REQUEST],
            "parallel": True,
            "priority": "normal",
            "max_concurrent": 2
        }
        
        response = client.post("/generate/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

class TestSecurity:
    """Test security features"""
    
    def test_security_headers(self):
        """Test security headers are present"""
        response = client.get("/")
        headers = response.headers
        
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
    
    def test_cors_headers(self):
        """Test CORS headers"""
        response = client.options("/")
        headers = response.headers
        
        assert "Access-Control-Allow-Origin" in headers
        assert "Access-Control-Allow-Methods" in headers
        assert "Access-Control-Allow-Headers" in headers
    
    def test_rate_limiting_headers(self):
        """Test rate limiting headers"""
        response = client.get("/")
        headers = response.headers
        
        # These headers should be added by middleware
        assert "X-Request-ID" in headers
        assert "X-Processing-Time" in headers

class TestErrorHandling:
    """Test error handling"""
    
    def test_404_error(self):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert "detail" in data
        assert "timestamp" in data
        assert "available_endpoints" in data
    
    def test_500_error(self):
        """Test 500 error handling"""
        with patch('api.enhanced_api.get_engine_dependency') as mock_engine:
            mock_engine.side_effect = Exception("Database connection failed")
            
            response = client.post("/generate", json=MOCK_DOCUMENT_REQUEST)
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
            assert "detail" in data
            assert "timestamp" in data

class TestPerformance:
    """Test performance features"""
    
    def test_response_compression(self):
        """Test response compression"""
        response = client.get("/")
        headers = response.headers
        
        # Check if compression is applied for large responses
        if "Content-Encoding" in headers:
            assert headers["Content-Encoding"] == "gzip"
    
    def test_response_time(self):
        """Test response time"""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            response = client.get("/health")
            results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests succeeded
        while not results.empty():
            status_code = results.get()
            assert status_code == 200

class TestWebSocket:
    """Test WebSocket functionality"""
    
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws/test_user") as websocket:
            # Send ping message
            websocket.send_text(json.dumps({"type": "ping"}))
            
            # Receive pong response
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"
            assert "timestamp" in response
    
    def test_websocket_subscription(self):
        """Test WebSocket subscription"""
        with client.websocket_connect("/ws/test_user") as websocket:
            # Send subscription message
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "channel": "document_generation"
            }))
            
            # Receive subscription confirmation
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "subscribed"
            assert response["channel"] == "document_generation"

class TestDatabaseIntegration:
    """Test database integration"""
    
    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test database connection"""
        db_manager = await get_database_manager()
        health = await db_manager.health_check()
        
        assert "database_manager" in health
        assert health["database_manager"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_database_session(self):
        """Test database session"""
        db_manager = await get_database_manager()
        
        async with db_manager.get_session() as session:
            assert session is not None
            # Test basic query
            result = await session.execute("SELECT 1")
            assert result is not None

class TestCacheIntegration:
    """Test cache integration"""
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache operations"""
        cache_manager = get_cache_manager()
        
        # Test set and get
        await cache_manager.set("test_key", "test_value", 60)
        value = await cache_manager.get("test_key")
        assert value == "test_value"
        
        # Test delete
        await cache_manager.delete("test_key")
        value = await cache_manager.get("test_key")
        assert value is None

class TestSecurityIntegration:
    """Test security integration"""
    
    @pytest.mark.asyncio
    async def test_security_manager(self):
        """Test security manager"""
        security_manager = get_security_manager()
        
        # Test password validation
        password_result = security_manager.password_manager.validate_password_strength("StrongPassword123!")
        assert password_result["is_valid"] is True
        assert password_result["score"] >= 4
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiter"""
        from security.enhanced_security import get_rate_limiter
        
        rate_limiter = get_rate_limiter()
        
        # Test rate limiting
        client_id = "test_client"
        assert rate_limiter.is_allowed(client_id) is True
        
        # Test remaining requests
        remaining = rate_limiter.get_remaining_requests(client_id)
        assert remaining >= 0

class TestAgentIntegration:
    """Test agent integration"""
    
    @pytest.mark.asyncio
    async def test_agent_manager(self):
        """Test agent manager"""
        agent_manager = await get_global_agent_manager()
        
        # Test agent stats
        stats = await agent_manager.get_agent_stats()
        assert "total_agents" in stats
        assert "active_agents" in stats
        assert "is_initialized" in stats

class TestMonitoring:
    """Test monitoring and metrics"""
    
    def test_health_check_components(self):
        """Test health check components"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        components = data["components"]
        
        assert "engine" in components
        assert "agent_manager" in components
        assert "cache" in components
        assert "rate_limiter" in components
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        metrics = data["metrics"]
        
        assert "total_requests" in metrics
        assert "uptime_seconds" in metrics
        assert "requests_per_minute" in metrics
    
    def test_performance_metrics(self):
        """Test performance metrics"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        performance = data["performance"]
        
        assert "avg_response_time" in performance
        assert "cache_hit_rate" in performance
        assert "error_rate" in performance

class TestInputValidation:
    """Test input validation"""
    
    def test_document_request_validation(self):
        """Test document request validation"""
        # Test valid request
        valid_request = MOCK_DOCUMENT_REQUEST.copy()
        response = client.post("/generate", json=valid_request)
        # Should not return validation error (might return other errors due to mocking)
        assert response.status_code != 422
    
    def test_language_validation(self):
        """Test language validation"""
        invalid_request = MOCK_DOCUMENT_REQUEST.copy()
        invalid_request["language"] = "invalid_language"
        
        response = client.post("/generate", json=invalid_request)
        assert response.status_code == 422
    
    def test_business_area_validation(self):
        """Test business area validation"""
        invalid_request = MOCK_DOCUMENT_REQUEST.copy()
        invalid_request["business_area"] = "invalid_area"
        
        response = client.post("/generate", json=invalid_request)
        assert response.status_code == 422
    
    def test_document_type_validation(self):
        """Test document type validation"""
        invalid_request = MOCK_DOCUMENT_REQUEST.copy()
        invalid_request["document_type"] = "invalid_type"
        
        response = client.post("/generate", json=invalid_request)
        assert response.status_code == 422

class TestBatchProcessing:
    """Test batch processing"""
    
    def test_batch_validation(self):
        """Test batch request validation"""
        # Test empty requests
        empty_batch = {"requests": []}
        response = client.post("/generate/batch", json=empty_batch)
        assert response.status_code == 422
        
        # Test too many requests
        too_many_requests = {
            "requests": [MOCK_DOCUMENT_REQUEST] * 11  # Max is 10
        }
        response = client.post("/generate/batch", json=too_many_requests)
        assert response.status_code == 422
    
    def test_batch_parallel_processing(self):
        """Test parallel batch processing"""
        batch_request = {
            "requests": [MOCK_DOCUMENT_REQUEST, MOCK_DOCUMENT_REQUEST],
            "parallel": True,
            "max_concurrent": 2
        }
        
        with patch('api.enhanced_api.get_engine_dependency') as mock_engine:
            with patch('api.enhanced_api.get_agent_manager_dependency') as mock_agent_mgr:
                # Setup mocks
                mock_agent = Mock()
                mock_agent.name = "Test Agent"
                mock_agent_mgr.return_value.get_best_agent.return_value = mock_agent
                
                mock_response = Mock()
                mock_response.id = "doc_123456"
                mock_response.request_id = "req_789012"
                mock_response.content = "Test content"
                mock_response.title = "Test title"
                mock_response.summary = "Test summary"
                mock_response.business_area = BusinessArea.MARKETING
                mock_response.document_type = DocumentType.MARKETING_STRATEGY
                mock_response.word_count = 100
                mock_response.processing_time = 1.0
                mock_response.confidence_score = 0.8
                mock_response.created_at = datetime.now()
                mock_response.metadata = {}
                
                mock_engine.return_value.generate_document.return_value = mock_response
                
                response = client.post("/generate/batch", json=batch_request)
                assert response.status_code == 200
                
                data = response.json()
                assert isinstance(data, list)
                assert len(data) == 2

# Performance tests
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_response_time_benchmark(self):
        """Benchmark response times"""
        times = []
        
        for _ in range(10):
            start = time.time()
            response = client.get("/health")
            end = time.time()
            
            assert response.status_code == 200
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 0.5  # Average response time should be under 500ms
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 10MB)
        assert memory_growth < 10 * 1024 * 1024

# Integration tests
class TestFullIntegration:
    """Full integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from request to response"""
        # This would test the full integration
        # including database, cache, external APIs, etc.
        pass
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery mechanisms"""
        # Test how the system recovers from various errors
        pass

# Fixtures
@pytest.fixture
def mock_engine():
    """Mock BUL engine"""
    engine = Mock(spec=BULEngine)
    engine.is_initialized = True
    return engine

@pytest.fixture
def mock_agent_manager():
    """Mock agent manager"""
    agent_manager = Mock(spec=SMEAgentManager)
    agent_manager.is_initialized = True
    return agent_manager

@pytest.fixture
def mock_cache():
    """Mock cache manager"""
    cache = Mock()
    cache.get.return_value = None
    cache.set.return_value = None
    return cache

@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter"""
    rate_limiter = Mock()
    rate_limiter.is_allowed.return_value = True
    return rate_limiter

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

# Test markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.performance
]












