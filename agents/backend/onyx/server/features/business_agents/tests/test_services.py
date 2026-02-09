"""
Service Tests
=============

Tests for service layer functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from ..services.health_service import HealthService
from ..services.system_info_service import SystemInfoService
from ..services.metrics_service import MetricsService

class TestHealthService:
    """Test HealthService functionality."""
    
    def test_initialization(self, mock_agent_manager):
        """Test health service initialization."""
        service = HealthService(mock_agent_manager)
        assert service.agent_manager == mock_agent_manager
    
    @pytest.mark.asyncio
    async def test_get_health_status_healthy(self, mock_agent_manager):
        """Test healthy status."""
        mock_agent_manager.list_agents.return_value = [Mock()]
        mock_agent_manager.list_workflows.return_value = [Mock()]
        
        service = HealthService(mock_agent_manager)
        result = await service.get_health_status()
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "components" in result
        assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_get_health_status_unhealthy(self):
        """Test unhealthy status when agent manager is None."""
        service = HealthService(None)
        result = await service.get_health_status()
        
        assert result["status"] == "unhealthy"
        assert "reason" in result

class TestSystemInfoService:
    """Test SystemInfoService functionality."""
    
    def test_initialization(self, mock_agent_manager):
        """Test system info service initialization."""
        service = SystemInfoService(mock_agent_manager)
        assert service.agent_manager == mock_agent_manager
    
    @pytest.mark.asyncio
    async def test_get_system_info(self, mock_agent_manager):
        """Test getting system information."""
        mock_agent_manager.get_business_areas.return_value = []
        mock_agent_manager.get_workflow_templates.return_value = {}
        mock_agent_manager.get_agents_by_business_area.return_value = []
        
        service = SystemInfoService(mock_agent_manager)
        result = await service.get_system_info()
        
        assert "system" in result
        assert "capabilities" in result
        assert "business_areas" in result
        assert "configuration" in result

class TestMetricsService:
    """Test MetricsService functionality."""
    
    def test_initialization(self, mock_agent_manager):
        """Test metrics service initialization."""
        service = MetricsService(mock_agent_manager)
        assert service.agent_manager == mock_agent_manager
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, mock_agent_manager):
        """Test getting system metrics."""
        mock_agent_manager.list_agents.return_value = []
        mock_agent_manager.list_workflows.return_value = []
        mock_agent_manager.get_business_areas.return_value = []
        
        service = MetricsService(mock_agent_manager)
        result = await service.get_system_metrics()
        
        assert "agents" in result
        assert "workflows" in result
        assert "business_areas" in result
