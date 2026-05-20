"""
Business Agents API Router
==========================

FastAPI router for business agents management and operations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from ..business_agents import BusinessAgentManager, BusinessAgent, BusinessArea, AgentCapability
from ..models import AgentResponse, CapabilityResponse, ExecutionRequest, ExecutionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Business Agents"])

# Dependency to get agent manager
def get_agent_manager() -> BusinessAgentManager:
    """Get the global agent manager instance."""
    # This would be injected from the main app
    from ..main import app
    return app.state.agent_manager

# Request/Response Models
class AgentListResponse(BaseModel):
    agents: List[AgentResponse]
    total: int
    business_areas: List[str]

class CapabilityExecutionRequest(BaseModel):
    inputs: Dict[str, Any] = Field(..., description="Input data for the capability")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class CapabilityExecutionResponse(BaseModel):
    status: str
    agent_id: str
    capability: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[int] = None

# Endpoints
@router.get("/", response_model=AgentListResponse)
async def list_agents(
    business_area: Optional[BusinessArea] = None,
    is_active: Optional[bool] = None,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """List all business agents with optional filtering."""
    
    try:
        agents = agent_manager.list_agents(business_area=business_area, is_active=is_active)
        
        agent_responses = [
            AgentResponse(
                id=agent.id,
                name=agent.name,
                business_area=agent.business_area.value,
                description=agent.description,
                capabilities=[
                    CapabilityResponse(
                        name=cap.name,
                        description=cap.description,
                        input_types=cap.input_types,
                        output_types=cap.output_types,
                        estimated_duration=cap.estimated_duration
                    ) for cap in agent.capabilities
                ],
                is_active=agent.is_active,
                created_at=agent.created_at.isoformat(),
                updated_at=agent.updated_at.isoformat()
            ) for agent in agents
        ]
        
        business_areas = [area.value for area in BusinessArea]
        
        return AgentListResponse(
            agents=agent_responses,
            total=len(agent_responses),
            business_areas=business_areas
        )
        
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get a specific business agent by ID."""
    
    try:
        agent = agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            business_area=agent.business_area.value,
            description=agent.description,
            capabilities=[
                CapabilityResponse(
                    name=cap.name,
                    description=cap.description,
                    input_types=cap.input_types,
                    output_types=cap.output_types,
                    estimated_duration=cap.estimated_duration
                ) for cap in agent.capabilities
            ],
            is_active=agent.is_active,
            created_at=agent.created_at.isoformat(),
            updated_at=agent.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")

@router.get("/{agent_id}/capabilities", response_model=List[CapabilityResponse])
async def get_agent_capabilities(
    agent_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get capabilities for a specific agent."""
    
    try:
        capabilities = agent_manager.get_agent_capabilities(agent_id)
        if not capabilities:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found or has no capabilities")
        
        return [
            CapabilityResponse(
                name=cap.name,
                description=cap.description,
                input_types=cap.input_types,
                output_types=cap.output_types,
                estimated_duration=cap.estimated_duration
            ) for cap in capabilities
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get capabilities for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@router.post("/{agent_id}/capabilities/{capability_name}/execute", response_model=CapabilityExecutionResponse)
async def execute_capability(
    agent_id: str,
    capability_name: str,
    request: CapabilityExecutionRequest,
    background_tasks: BackgroundTasks,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Execute a specific agent capability."""
    
    try:
        result = await agent_manager.execute_agent_capability(
            agent_id=agent_id,
            capability_name=capability_name,
            inputs=request.inputs,
            parameters=request.parameters
        )
        
        return CapabilityExecutionResponse(
            status=result["status"],
            agent_id=result["agent_id"],
            capability=result["capability"],
            result=result.get("result"),
            error=result.get("error"),
            execution_time=result.get("execution_time")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute capability {capability_name} for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute capability: {str(e)}")

@router.get("/business-areas/{business_area}/agents", response_model=List[AgentResponse])
async def get_agents_by_business_area(
    business_area: BusinessArea,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get all agents for a specific business area."""
    
    try:
        agents = agent_manager.get_agents_by_business_area(business_area)
        
        return [
            AgentResponse(
                id=agent.id,
                name=agent.name,
                business_area=agent.business_area.value,
                description=agent.description,
                capabilities=[
                    CapabilityResponse(
                        name=cap.name,
                        description=cap.description,
                        input_types=cap.input_types,
                        output_types=cap.output_types,
                        estimated_duration=cap.estimated_duration
                    ) for cap in agent.capabilities
                ],
                is_active=agent.is_active,
                created_at=agent.created_at.isoformat(),
                updated_at=agent.updated_at.isoformat()
            ) for agent in agents
        ]
        
    except Exception as e:
        logger.error(f"Failed to get agents for business area {business_area}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")

@router.get("/business-areas/", response_model=List[str])
async def get_business_areas(
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get all available business areas."""
    
    try:
        business_areas = agent_manager.get_business_areas()
        return [area.value for area in business_areas]
        
    except Exception as e:
        logger.error(f"Failed to get business areas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get business areas: {str(e)}")

@router.get("/templates/workflows", response_model=Dict[str, List[Dict[str, Any]]])
async def get_workflow_templates(
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get predefined workflow templates for each business area."""
    
    try:
        templates = agent_manager.get_workflow_templates()
        return templates
        
    except Exception as e:
        logger.error(f"Failed to get workflow templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow templates: {str(e)}")


