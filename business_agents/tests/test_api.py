"""
API Tests
=========

Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock

from ..main import app

class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_system_info(self, test_client):
        """Test system info endpoint."""
        response = test_client.get("/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "capabilities" in data
    
    def test_system_metrics(self, test_client):
        """Test system metrics endpoint."""
        response = test_client.get("/system/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "workflows" in data
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "features" in data
    
    def test_business_agents_overview(self, test_client):
        """Test business agents overview endpoint."""
        response = test_client.get("/business-agents/")
        assert response.status_code == 200
        data = response.json()
        assert "system_name" in data
        assert "total_agents" in data
        assert "business_areas" in data
    
    def test_list_agents(self, test_client):
        """Test list agents endpoint."""
        response = test_client.get("/business-agents/agents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_agent_not_found(self, test_client):
        """Test get agent with non-existent ID."""
        response = test_client.get("/business-agents/agents/nonexistent_agent")
        assert response.status_code == 404
    
    def test_list_workflows(self, test_client):
        """Test list workflows endpoint."""
        response = test_client.get("/business-agents/workflows")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_workflow_not_found(self, test_client):
        """Test get workflow with non-existent ID."""
        response = test_client.get("/business-agents/workflows/nonexistent_workflow")
        assert response.status_code == 404
    
    def test_list_documents(self, test_client):
        """Test list documents endpoint."""
        response = test_client.get("/business-agents/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_document_not_found(self, test_client):
        """Test get document with non-existent ID."""
        response = test_client.get("/business-agents/documents/nonexistent_document")
        assert response.status_code == 404
    
    def test_business_areas(self, test_client):
        """Test business areas endpoint."""
        response = test_client.get("/business-agents/business-areas")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_workflow_templates(self, test_client):
        """Test workflow templates endpoint."""
        response = test_client.get("/business-agents/workflow-templates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
