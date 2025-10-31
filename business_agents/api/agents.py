"""
Agents API Router
=================

API endpoints for business agent operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional
import logging

from ..business_agents import BusinessArea
from ..schemas.agent_schemas import (
    AgentResponse, AgentListResponse, AgentCapabilityResponse,
    CapabilityExecutionRequest, CapabilityExecutionResponse,
    BusinessAreaResponse
)
from ..core.dependencies import get_agent_service
from ..core.exceptions import convert_to_http_exception
from ..services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agents"])

@router.get("/", response_model=AgentListResponse)
async def list_agents(
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    agent_service: AgentService = Depends(get_agent_service)
):
    """List all business agents with optional filters."""
    
    try:
        agents_data = await agent_service.list_agents(
            business_area=business_area,
            is_active=is_active
        )
        
        return AgentListResponse(
            agents=agents_data,
            total=len(agents_data),
            business_area=business_area.value if business_area else None,
            is_active=is_active
        )
        
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list agents")

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get specific agent details."""
    
    try:
        agent_data = await agent_service.get_agent(agent_id)
        return AgentResponse(**agent_data)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agent")

@router.get("/{agent_id}/capabilities", response_model=List[AgentCapabilityResponse])
async def get_agent_capabilities(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get capabilities for a specific agent."""
    
    try:
        capabilities_data = await agent_service.get_agent_capabilities(agent_id)
        return [AgentCapabilityResponse(**cap) for cap in capabilities_data]
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get agent capabilities for {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agent capabilities")

@router.post("/{agent_id}/execute", response_model=CapabilityExecutionResponse)
async def execute_agent_capability(
    agent_id: str,
    request: CapabilityExecutionRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Execute a specific agent capability."""
    
    try:
        result = await agent_service.execute_agent_capability(
            agent_id=request.agent_id,
            capability_name=request.capability_name,
            inputs=request.inputs,
            parameters=request.parameters
        )
        
        return CapabilityExecutionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute agent capability: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute agent capability")

@router.get("/by-business-area/{business_area}", response_model=List[BusinessAreaResponse])
async def get_agents_by_business_area(
    business_area: BusinessArea,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get all agents for a specific business area."""
    
    try:
        agents_data = await agent_service.get_agents_by_business_area(business_area)
        return [BusinessAreaResponse(**agent) for agent in agents_data]
        
    except Exception as e:
        logger.error(f"Failed to get agents by business area {business_area}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agents by business area")
