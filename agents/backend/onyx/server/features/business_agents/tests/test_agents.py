"""
Agent Tests
===========

Tests for business agent functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException

from ..business_agents import BusinessAgentManager, BusinessArea, BusinessAgent, AgentCapability
from ..core.exceptions import AgentNotFoundError, AgentInactiveError, CapabilityNotFoundError
from ..services.agent_service import AgentService

class TestBusinessAgentManager:
    """Test BusinessAgentManager functionality."""
    
    def test_initialization(self):
        """Test agent manager initialization."""
        manager = BusinessAgentManager()
        assert manager is not None
        assert len(manager.agents) > 0
    
    def test_get_agent_success(self):
        """Test successful agent retrieval."""
        manager = BusinessAgentManager()
        agent = manager.get_agent("marketing_001")
        assert agent is not None
        assert agent.id == "marketing_001"
    
    def test_get_agent_not_found(self):
        """Test agent not found scenario."""
        manager = BusinessAgentManager()
        agent = manager.get_agent("nonexistent_agent")
        assert agent is None
    
    def test_list_agents_with_filters(self):
        """Test listing agents with filters."""
        manager = BusinessAgentManager()
        
        # Test with business area filter
        marketing_agents = manager.list_agents(business_area=BusinessArea.MARKETING)
        assert len(marketing_agents) > 0
        assert all(agent.business_area == BusinessArea.MARKETING for agent in marketing_agents)
        
        # Test with active filter
        active_agents = manager.list_agents(is_active=True)
        assert len(active_agents) > 0
        assert all(agent.is_active for agent in active_agents)
    
    def test_get_agent_capabilities(self):
        """Test getting agent capabilities."""
        manager = BusinessAgentManager()
        capabilities = manager.get_agent_capabilities("marketing_001")
        assert len(capabilities) > 0
        assert all(isinstance(cap, AgentCapability) for cap in capabilities)
    
    def test_get_agent_capabilities_not_found(self):
        """Test getting capabilities for non-existent agent."""
        manager = BusinessAgentManager()
        capabilities = manager.get_agent_capabilities("nonexistent_agent")
        assert capabilities == []
    
    @pytest.mark.asyncio
    async def test_execute_agent_capability_success(self):
        """Test successful agent capability execution."""
        manager = BusinessAgentManager()
        result = await manager.execute_agent_capability(
            agent_id="marketing_001",
            capability_name="campaign_planning",
            inputs={"target_audience": "test", "budget": 1000}
        )
        assert result["status"] == "completed"
        assert "result" in result
    
    @pytest.mark.asyncio
    async def test_execute_agent_capability_agent_not_found(self):
        """Test agent capability execution with non-existent agent."""
        manager = BusinessAgentManager()
        with pytest.raises(ValueError, match="Agent nonexistent_agent not found"):
            await manager.execute_agent_capability(
                agent_id="nonexistent_agent",
                capability_name="test_capability",
                inputs={}
            )
    
    @pytest.mark.asyncio
    async def test_execute_agent_capability_capability_not_found(self):
        """Test agent capability execution with non-existent capability."""
        manager = BusinessAgentManager()
        with pytest.raises(ValueError, match="Capability nonexistent_capability not found"):
            await manager.execute_agent_capability(
                agent_id="marketing_001",
                capability_name="nonexistent_capability",
                inputs={}
            )
    
    def test_get_business_areas(self):
        """Test getting all business areas."""
        manager = BusinessAgentManager()
        areas = manager.get_business_areas()
        assert len(areas) > 0
        assert BusinessArea.MARKETING in areas
        assert BusinessArea.SALES in areas
    
    def test_get_agents_by_business_area(self):
        """Test getting agents by business area."""
        manager = BusinessAgentManager()
        marketing_agents = manager.get_agents_by_business_area(BusinessArea.MARKETING)
        assert len(marketing_agents) > 0
        assert all(agent.business_area == BusinessArea.MARKETING for agent in marketing_agents)

class TestAgentService:
    """Test AgentService functionality."""
    
    def test_initialization(self, mock_agent_manager):
        """Test agent service initialization."""
        service = AgentService(mock_agent_manager)
        assert service.agent_manager == mock_agent_manager
    
    @pytest.mark.asyncio
    async def test_list_agents(self, mock_agent_manager):
        """Test listing agents through service."""
        # Setup mock
        mock_agent = Mock(spec=BusinessAgent)
        mock_agent.id = "test_agent"
        mock_agent.name = "Test Agent"
        mock_agent.business_area = BusinessArea.MARKETING
        mock_agent.description = "Test description"
        mock_agent.capabilities = []
        mock_agent.is_active = True
        mock_agent.created_at = "2024-01-01T00:00:00"
        mock_agent.updated_at = "2024-01-01T00:00:00"
        
        mock_agent_manager.list_agents.return_value = [mock_agent]
        
        service = AgentService(mock_agent_manager)
        result = await service.list_agents()
        
        assert len(result) == 1
        assert result[0]["id"] == "test_agent"
        assert result[0]["business_area"] == "marketing"
    
    @pytest.mark.asyncio
    async def test_get_agent_success(self, mock_agent_manager):
        """Test successful agent retrieval through service."""
        # Setup mock
        mock_agent = Mock(spec=BusinessAgent)
        mock_agent.id = "test_agent"
        mock_agent.name = "Test Agent"
        mock_agent.business_area = BusinessArea.MARKETING
        mock_agent.description = "Test description"
        mock_agent.capabilities = []
        mock_agent.is_active = True
        mock_agent.created_at = "2024-01-01T00:00:00"
        mock_agent.updated_at = "2024-01-01T00:00:00"
        mock_agent.metadata = {}
        
        mock_agent_manager.get_agent.return_value = mock_agent
        
        service = AgentService(mock_agent_manager)
        result = await service.get_agent("test_agent")
        
        assert result["id"] == "test_agent"
        assert result["name"] == "Test Agent"
    
    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, mock_agent_manager):
        """Test agent not found through service."""
        mock_agent_manager.get_agent.return_value = None
        
        service = AgentService(mock_agent_manager)
        with pytest.raises(ValueError, match="Agent test_agent not found"):
            await service.get_agent("test_agent")
    
    @pytest.mark.asyncio
    async def test_execute_agent_capability(self, mock_agent_manager):
        """Test agent capability execution through service."""
        mock_agent_manager.execute_agent_capability.return_value = {
            "status": "completed",
            "result": {"test": "value"}
        }
        
        service = AgentService(mock_agent_manager)
        result = await service.execute_agent_capability(
            agent_id="test_agent",
            capability_name="test_capability",
            inputs={"test_input": "test_value"}
        )
        
        assert result["status"] == "completed"
        assert "result" in result
        mock_agent_manager.execute_agent_capability.assert_called_once()
