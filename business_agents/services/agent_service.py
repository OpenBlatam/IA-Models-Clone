"""
Agent Service
=============

Service layer for business agent operations.
"""

from typing import Dict, List, Any, Optional
import logging

from ..business_agents import BusinessAgentManager, BusinessArea, BusinessAgent, AgentCapability

logger = logging.getLogger(__name__)

class AgentService:
    """Service for business agent operations."""
    
    def __init__(self, agent_manager: BusinessAgentManager):
        self.agent_manager = agent_manager
    
    async def list_agents(
        self,
        business_area: Optional[BusinessArea] = None,
        is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """List all business agents with optional filters."""
        
        agents = self.agent_manager.list_agents(business_area=business_area, is_active=is_active)
        
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "business_area": agent.business_area.value,
                "description": agent.description,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "input_types": cap.input_types,
                        "output_types": cap.output_types,
                        "estimated_duration": cap.estimated_duration
                    }
                    for cap in agent.capabilities
                ],
                "is_active": agent.is_active,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat()
            }
            for agent in agents
        ]
    
    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get specific agent details."""
        
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        return {
            "id": agent.id,
            "name": agent.name,
            "business_area": agent.business_area.value,
            "description": agent.description,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_types": cap.input_types,
                    "output_types": cap.output_types,
                    "parameters": cap.parameters,
                    "estimated_duration": cap.estimated_duration
                }
                for cap in agent.capabilities
            ],
            "is_active": agent.is_active,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat(),
            "metadata": agent.metadata
        }
    
    async def get_agent_capabilities(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get capabilities for a specific agent."""
        
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        capabilities = self.agent_manager.get_agent_capabilities(agent_id)
        
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "input_types": cap.input_types,
                "output_types": cap.output_types,
                "parameters": cap.parameters,
                "estimated_duration": cap.estimated_duration
            }
            for cap in capabilities
        ]
    
    async def execute_agent_capability(
        self,
        agent_id: str,
        capability_name: str,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a specific agent capability."""
        
        try:
            result = await self.agent_manager.execute_agent_capability(
                agent_id=agent_id,
                capability_name=capability_name,
                inputs=inputs,
                parameters=parameters or {}
            )
            
            return result
            
        except ValueError as e:
            logger.error(f"Agent capability execution failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in agent capability execution: {str(e)}")
            raise Exception("Internal server error")
    
    async def get_agents_by_business_area(self, business_area: BusinessArea) -> List[Dict[str, Any]]:
        """Get all agents for a specific business area."""
        
        agents = self.agent_manager.get_agents_by_business_area(business_area)
        
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "capabilities_count": len(agent.capabilities),
                "is_active": agent.is_active
            }
            for agent in agents
        ]
