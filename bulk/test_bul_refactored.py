"""
BUL System Tests
================

Tests for the refactored BUL system.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from bul_main import BULSystem
from bul_config import BULConfig

class TestBULSystem:
    """Test cases for BUL system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.system = BULSystem()
        self.client = TestClient(self.system.app)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "3.0.0"
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "active_tasks" in data
    
    def test_document_generation(self):
        """Test document generation endpoint."""
        request_data = {
            "query": "Create a marketing strategy for a new restaurant",
            "business_area": "marketing",
            "document_type": "strategy",
            "priority": 1
        }
        
        response = self.client.post("/documents/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert "message" in data
        assert "estimated_time" in data
        
        # Store task_id for further tests
        self.task_id = data["task_id"]
    
    def test_task_status(self):
        """Test task status endpoint."""
        # First create a task
        request_data = {
            "query": "Test query",
            "business_area": "marketing"
        }
        
        response = self.client.post("/documents/generate", json=request_data)
        task_id = response.json()["task_id"]
        
        # Check task status
        response = self.client.get(f"/tasks/{task_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
        assert 0 <= data["progress"] <= 100
    
    def test_list_tasks(self):
        """Test list tasks endpoint."""
        response = self.client.get("/tasks")
        assert response.status_code == 200
        
        data = response.json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
    
    def test_delete_task(self):
        """Test delete task endpoint."""
        # First create a task
        request_data = {
            "query": "Test query for deletion",
            "business_area": "marketing"
        }
        
        response = self.client.post("/documents/generate", json=request_data)
        task_id = response.json()["task_id"]
        
        # Delete the task
        response = self.client.delete(f"/tasks/{task_id}")
        assert response.status_code == 200
        
        # Verify task is deleted
        response = self.client.get(f"/tasks/{task_id}/status")
        assert response.status_code == 404
    
    def test_invalid_task_id(self):
        """Test invalid task ID handling."""
        response = self.client.get("/tasks/invalid_task_id/status")
        assert response.status_code == 404
        
        response = self.client.delete("/tasks/invalid_task_id")
        assert response.status_code == 404
    
    def test_invalid_document_request(self):
        """Test invalid document request."""
        # Missing required field
        request_data = {
            "business_area": "marketing"
            # Missing "query" field
        }
        
        response = self.client.post("/documents/generate", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_priority_validation(self):
        """Test priority field validation."""
        # Invalid priority (too high)
        request_data = {
            "query": "Test query",
            "priority": 10  # Should be 1-5
        }
        
        response = self.client.post("/documents/generate", json=request_data)
        assert response.status_code == 422
        
        # Valid priority
        request_data = {
            "query": "Test query",
            "priority": 3
        }
        
        response = self.client.post("/documents/generate", json=request_data)
        assert response.status_code == 200

class TestBULConfig:
    """Test cases for BUL configuration."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = BULConfig()
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.debug_mode is False
    
    def test_business_areas(self):
        """Test business areas configuration."""
        config = BULConfig()
        assert "marketing" in config.enabled_business_areas
        assert "sales" in config.enabled_business_areas
        assert len(config.enabled_business_areas) == 10
    
    def test_document_types(self):
        """Test document types configuration."""
        config = BULConfig()
        assert "marketing" in config.document_types
        assert "strategy" in config.document_types["marketing"]
        assert "sales" in config.document_types
        assert "proposal" in config.document_types["sales"]
    
    def test_get_business_area_config(self):
        """Test getting business area configuration."""
        config = BULConfig()
        area_config = config.get_business_area_config("marketing")
        
        assert area_config["enabled"] is True
        assert "document_types" in area_config
        assert "priority" in area_config
    
    def test_get_document_type_config(self):
        """Test getting document type configuration."""
        config = BULConfig()
        doc_config = config.get_document_type_config("strategy")
        
        assert "supported_formats" in doc_config
        assert "max_size" in doc_config
        assert "timeout" in doc_config

@pytest.mark.asyncio
async def test_async_document_processing():
    """Test asynchronous document processing."""
    system = BULSystem()
    
    # Create a mock request
    from bul_main import DocumentRequest
    request = DocumentRequest(
        query="Test async processing",
        business_area="marketing",
        document_type="strategy"
    )
    
    # Test the processing function
    task_id = "test_task_123"
    system.tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "request": request.dict(),
        "created_at": None,
        "result": None,
        "error": None
    }
    
    # Run the processing
    await system.process_document(task_id, request)
    
    # Check results
    assert system.tasks[task_id]["status"] == "completed"
    assert system.tasks[task_id]["progress"] == 100
    assert system.tasks[task_id]["result"] is not None
    assert "document_id" in system.tasks[task_id]["result"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


