"""
System Tests
============

Comprehensive test suite for the Bulk TruthGPT system.
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

# Import the refactored main application
from ..main_refactored import app, components, initialize_components, cleanup_components
from ..config.settings import settings
from ..utils.exceptions import BulkTruthGPTException, ErrorCode
from ..utils.metrics import metrics_collector

class TestSystemInitialization:
    """Test system initialization and component management."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    async def initialized_system(self):
        """Initialize system for testing."""
        await initialize_components()
        yield
        await cleanup_components()
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
    
    def test_database_health(self, client):
        """Test database health check."""
        response = client.get("/health/database")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_redis_health(self, client):
        """Test Redis health check."""
        response = client.get("/health/redis")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_model_health(self, client):
        """Test model health check."""
        response = client.get("/health/model")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_summary(self, client):
        """Test metrics summary endpoint."""
        response = client.get("/api/metrics/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data
        assert "error_stats" in data

class TestBulkGeneration:
    """Test bulk document generation functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_generate_bulk_documents_success(self, client):
        """Test successful bulk document generation."""
        request_data = {
            "query": "Write about artificial intelligence",
            "max_documents": 5,
            "config": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        with patch('bulk_truthgpt.main_refactored.components', {'queue_manager': Mock()}):
            components['queue_manager'].start_bulk_generation = AsyncMock(return_value="test-task-id")
            
            response = client.post("/api/generate/bulk", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "task_id" in data
            assert "status" in data
            assert "message" in data
    
    def test_generate_bulk_documents_empty_query(self, client):
        """Test bulk generation with empty query."""
        request_data = {
            "query": "",
            "max_documents": 5
        }
        
        response = client.post("/api/generate/bulk", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Query cannot be empty" in data["detail"]
    
    def test_generate_bulk_documents_invalid_max_documents(self, client):
        """Test bulk generation with invalid max documents."""
        request_data = {
            "query": "Write about AI",
            "max_documents": 0
        }
        
        response = client.post("/api/generate/bulk", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Max documents must be positive" in data["detail"]
    
    def test_generate_bulk_documents_exceeds_limit(self, client):
        """Test bulk generation exceeding limit."""
        request_data = {
            "query": "Write about AI",
            "max_documents": 10000  # Exceeds default limit
        }
        
        response = client.post("/api/generate/bulk", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Max documents cannot exceed" in data["detail"]
    
    def test_get_task_status(self, client):
        """Test getting task status."""
        task_id = "test-task-id"
        
        with patch('bulk_truthgpt.main_refactored.components', {'queue_manager': Mock()}):
            components['queue_manager'].get_task_status = AsyncMock(return_value={
                "task_id": task_id,
                "status": "running",
                "progress": 50
            })
            
            response = client.get(f"/api/tasks/{task_id}/status")
            assert response.status_code == 200
            
            data = response.json()
            assert "task_id" in data
            assert "status" in data
            assert "progress" in data
    
    def test_get_task_status_not_found(self, client):
        """Test getting status for non-existent task."""
        task_id = "non-existent-task"
        
        with patch('bulk_truthgpt.main_refactored.components', {'queue_manager': Mock()}):
            components['queue_manager'].get_task_status = AsyncMock(return_value=None)
            
            response = client.get(f"/api/tasks/{task_id}/status")
            assert response.status_code == 404
            
            data = response.json()
            assert "detail" in data
            assert "Task not found" in data["detail"]
    
    def test_get_task_results(self, client):
        """Test getting task results."""
        task_id = "test-task-id"
        
        with patch('bulk_truthgpt.main_refactored.components', {'queue_manager': Mock()}):
            components['queue_manager'].get_task_results = AsyncMock(return_value={
                "task_id": task_id,
                "status": "completed",
                "documents": [
                    {"id": "doc1", "content": "Document 1"},
                    {"id": "doc2", "content": "Document 2"}
                ]
            })
            
            response = client.get(f"/api/tasks/{task_id}/results")
            assert response.status_code == 200
            
            data = response.json()
            assert "task_id" in data
            assert "status" in data
            assert "documents" in data
    
    def test_get_task_results_not_found(self, client):
        """Test getting results for non-existent task."""
        task_id = "non-existent-task"
        
        with patch('bulk_truthgpt.main_refactored.components', {'queue_manager': Mock()}):
            components['queue_manager'].get_task_results = AsyncMock(return_value=None)
            
            response = client.get(f"/api/tasks/{task_id}/results")
            assert response.status_code == 404
            
            data = response.json()
            assert "detail" in data
            assert "Task not found" in data["detail"]

class TestContentAnalysis:
    """Test content analysis functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_analyze_content_quality_success(self, client):
        """Test successful content quality analysis."""
        request_data = {
            "content": "This is a test document about artificial intelligence. It contains multiple sentences and should be analyzed for quality."
        }
        
        with patch('bulk_truthgpt.main_refactored.components', {'content_analyzer': Mock()}):
            mock_analysis = Mock()
            mock_analysis.quality_score = 0.85
            mock_analysis.readability_score = 0.80
            mock_analysis.coherence_score = 0.90
            mock_analysis.engagement_score = 0.75
            mock_analysis.metrics = {"word_count": 25, "sentence_count": 3}
            mock_analysis.suggestions = ["Consider adding more examples"]
            
            components['content_analyzer'].analyze_content = AsyncMock(return_value=mock_analysis)
            
            response = client.post("/api/analyze/quality", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "quality_score" in data
            assert "readability_score" in data
            assert "coherence_score" in data
            assert "engagement_score" in data
            assert "metrics" in data
            assert "suggestions" in data
    
    def test_analyze_content_quality_empty_content(self, client):
        """Test content analysis with empty content."""
        request_data = {
            "content": ""
        }
        
        response = client.post("/api/analyze/quality", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Content cannot be empty" in data["detail"]
    
    def test_analyze_content_quality_missing_content(self, client):
        """Test content analysis with missing content."""
        request_data = {}
        
        response = client.post("/api/analyze/quality", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Content cannot be empty" in data["detail"]

class TestSystemStatus:
    """Test system status functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_get_system_status(self, client):
        """Test getting system status."""
        with patch('bulk_truthgpt.main_refactored.components', {'system_monitor': Mock()}):
            components['system_monitor'].get_system_status = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "components": {
                    "truthgpt_engine": {"status": "running"},
                    "document_generator": {"status": "running"}
                }
            })
            
            response = client.get("/api/system/status")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "components" in data

class TestErrorHandling:
    """Test error handling functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_bulk_truthgpt_exception_handling(self, client):
        """Test BulkTruthGPT exception handling."""
        with patch('bulk_truthgpt.main_refactored.components', {}):
            # Simulate missing component
            response = client.post("/api/generate/bulk", json={
                "query": "Test query",
                "max_documents": 5
            })
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
            assert "code" in data["error"]
            assert "message" in data["error"]
    
    def test_general_exception_handling(self, client):
        """Test general exception handling."""
        with patch('bulk_truthgpt.main_refactored.components', {'queue_manager': Mock()}):
            components['queue_manager'].start_bulk_generation = AsyncMock(side_effect=Exception("Test error"))
            
            response = client.post("/api/generate/bulk", json={
                "query": "Test query",
                "max_documents": 5
            })
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
            assert "code" in data["error"]
            assert "message" in data["error"]

class TestMetrics:
    """Test metrics functionality."""
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        # Test custom metric recording
        metrics_collector.record_custom_metric("test_metric", 100.0, {"label": "test"})
        
        # Test metric summary
        summary = metrics_collector.get_metric_summary("test_metric")
        assert summary is not None
        assert summary["name"] == "test_metric"
        assert summary["count"] == 1
        assert summary["mean"] == 100.0
    
    def test_request_metrics(self):
        """Test request metrics recording."""
        # Test request recording
        metrics_collector.record_request("GET", "/health", 200, 0.1)
        metrics_collector.record_request("POST", "/api/generate/bulk", 200, 0.5)
        
        # Test generation metrics
        metrics_collector.record_generation("task1", "completed", 10.0, 0.85)
        metrics_collector.record_generation("task2", "failed", 5.0, 0.0)
        
        # Test error metrics
        metrics_collector.record_error("ValidationError", "api")
        metrics_collector.record_error("SystemError", "core")
    
    def test_metrics_export(self):
        """Test metrics export."""
        # Test JSON export
        json_metrics = metrics_collector.export_metrics("json")
        assert isinstance(json_metrics, str)
        
        # Test Prometheus export
        prometheus_metrics = metrics_collector.export_metrics("prometheus")
        assert isinstance(prometheus_metrics, str)
        assert "bulk_truthgpt_requests_total" in prometheus_metrics

class TestConfiguration:
    """Test configuration functionality."""
    
    def test_settings_validation(self):
        """Test settings validation."""
        # Test valid configuration
        assert settings.validate_config() is True
        
        # Test configuration access
        assert settings.api_host is not None
        assert settings.api_port is not None
        assert settings.database_url is not None
        assert settings.redis_url is not None
    
    def test_settings_export(self):
        """Test settings export."""
        # Test to_dict
        settings_dict = settings.to_dict()
        assert isinstance(settings_dict, dict)
        assert "api_host" in settings_dict
        assert "api_port" in settings_dict
    
    def test_database_config(self):
        """Test database configuration."""
        db_config = settings.get_database_config()
        assert db_config.url is not None
        assert db_config.pool_size > 0
        assert db_config.max_overflow > 0
    
    def test_redis_config(self):
        """Test Redis configuration."""
        redis_config = settings.get_redis_config()
        assert redis_config.url is not None
        assert redis_config.pool_size > 0
    
    def test_security_config(self):
        """Test security configuration."""
        security_config = settings.get_security_config()
        assert security_config.secret_key is not None
        assert security_config.access_token_expire_minutes > 0
        assert len(security_config.cors_origins) > 0

# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_complete_workflow(self, client):
        """Test complete document generation workflow."""
        # 1. Check system health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Start bulk generation
        generation_request = {
            "query": "Write about machine learning",
            "max_documents": 3,
            "config": {
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
        
        with patch('bulk_truthgpt.main_refactored.components', {'queue_manager': Mock()}):
            components['queue_manager'].start_bulk_generation = AsyncMock(return_value="workflow-task-123")
            
            generation_response = client.post("/api/generate/bulk", json=generation_request)
            assert generation_response.status_code == 200
            
            task_id = generation_response.json()["task_id"]
            
            # 3. Check task status
            components['queue_manager'].get_task_status = AsyncMock(return_value={
                "task_id": task_id,
                "status": "running",
                "progress": 50
            })
            
            status_response = client.get(f"/api/tasks/{task_id}/status")
            assert status_response.status_code == 200
            
            # 4. Get task results
            components['queue_manager'].get_task_results = AsyncMock(return_value={
                "task_id": task_id,
                "status": "completed",
                "documents": [
                    {"id": "doc1", "content": "Machine learning is a subset of AI..."},
                    {"id": "doc2", "content": "Deep learning uses neural networks..."},
                    {"id": "doc3", "content": "Supervised learning uses labeled data..."}
                ]
            })
            
            results_response = client.get(f"/api/tasks/{task_id}/results")
            assert results_response.status_code == 200
            
            # 5. Analyze content quality
            with patch('bulk_truthgpt.main_refactored.components', {'content_analyzer': Mock()}):
                mock_analysis = Mock()
                mock_analysis.quality_score = 0.88
                mock_analysis.readability_score = 0.82
                mock_analysis.coherence_score = 0.90
                mock_analysis.engagement_score = 0.85
                mock_analysis.metrics = {"word_count": 150, "sentence_count": 8}
                mock_analysis.suggestions = ["Consider adding more examples"]
                
                components['content_analyzer'].analyze_content = AsyncMock(return_value=mock_analysis)
                
                analysis_request = {
                    "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models."
                }
                
                analysis_response = client.post("/api/analyze/quality", json=analysis_request)
                assert analysis_response.status_code == 200
                
                analysis_data = analysis_response.json()
                assert analysis_data["quality_score"] == 0.88
                assert analysis_data["readability_score"] == 0.82
                assert analysis_data["coherence_score"] == 0.90
                assert analysis_data["engagement_score"] == 0.85
    
    def test_error_recovery(self, client):
        """Test error recovery and resilience."""
        # Test with invalid request
        invalid_request = {
            "query": "",
            "max_documents": -1
        }
        
        response = client.post("/api/generate/bulk", json=invalid_request)
        assert response.status_code == 400
        
        # Test with missing component
        with patch('bulk_truthgpt.main_refactored.components', {}):
            response = client.post("/api/generate/bulk", json={
                "query": "Valid query",
                "max_documents": 5
            })
            assert response.status_code == 500
        
        # Test system should still be responsive
        health_response = client.get("/health")
        assert health_response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])











