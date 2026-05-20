"""
Agent Controller
Handles agent lifecycle and operations
Refactored with service layer separation and middleware
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException

from ...core.agent_service import get_agent_service
from ...core.exceptions import (
    AgentNotFoundError,
    AgentAlreadyRunningError,
    AgentNotRunningError,
    TaskNotFoundError,
    RateLimitExceededError,
    InvalidAgentStateError
)
from ..middleware.rate_limit_middleware import rate_limit
from ..middleware.error_handler import handle_agent_exceptions
from ..schemas.requests import (
    StartAgentRequest,
    AddTaskRequest,
    UpdateAgentRequest,
    ParallelAgentRequest
)
from ..schemas.responses import (
    AgentStatusResponse,
    TaskResponse,
    AgentListResponse,
    MessageResponse,
    ParallelAgentsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# Agent service
_agent_service = get_agent_service()


@router.post("/start", response_model=MessageResponse)
@rate_limit("start_agent")
@handle_agent_exceptions
async def start_agent(request: StartAgentRequest) -> MessageResponse:
    """
    Start a new autonomous agent
    
    Creates and starts an autonomous agent that will run continuously
    until explicitly stopped.
    """
    agent = await _agent_service.create_and_start_agent(
        agent_id=request.agent_id,
        instruction=request.instruction,
        enhanced=request.enhanced
    )
    
    return MessageResponse(
        message=f"Agent {agent.agent_id} started successfully",
        success=True
    )


@router.post("/parallel", response_model=ParallelAgentsResponse)
@rate_limit("parallel_agents")
@handle_agent_exceptions
async def start_parallel_agents(request: ParallelAgentRequest) -> ParallelAgentsResponse:
    """
    Start multiple agents in parallel
    
    Creates and starts multiple agents simultaneously with the same instruction.
    Maximum number of parallel agents is controlled by settings.
    """
    agent_ids = await _agent_service.create_parallel_agents(
        count=request.count,
        instruction=request.instruction,
        enhanced=request.enhanced
    )
    
    return ParallelAgentsResponse(
        agent_ids=agent_ids,
        total=len(agent_ids),
        message=f"Started {len(agent_ids)} agents in parallel"
    )


@router.post("/{agent_id}/stop", response_model=MessageResponse)
@handle_agent_exceptions
async def stop_agent(agent_id: str) -> MessageResponse:
    """Stop an agent"""
    await _agent_service.stop_agent(agent_id)
    return MessageResponse(
        message=f"Agent {agent_id} stopped successfully",
        success=True
    )


@router.post("/{agent_id}/pause", response_model=MessageResponse)
@handle_agent_exceptions
async def pause_agent(agent_id: str) -> MessageResponse:
    """Pause an agent"""
    await _agent_service.pause_agent(agent_id)
    return MessageResponse(
        message=f"Agent {agent_id} paused",
        success=True
    )


@router.post("/{agent_id}/resume", response_model=MessageResponse)
@handle_agent_exceptions
async def resume_agent(agent_id: str) -> MessageResponse:
    """Resume a paused agent"""
    await _agent_service.resume_agent(agent_id)
    return MessageResponse(
        message=f"Agent {agent_id} resumed",
        success=True
    )


@router.get("/{agent_id}/status", response_model=AgentStatusResponse)
@handle_agent_exceptions
async def get_agent_status(agent_id: str) -> AgentStatusResponse:
    """Get agent status"""
    status = await _agent_service.get_agent_status(agent_id)
    return AgentStatusResponse(**status)


@router.get("/{agent_id}/health")
@handle_agent_exceptions
async def get_agent_health(agent_id: str) -> Dict[str, Any]:
    """Get agent health status"""
    return await _agent_service.get_agent_health(agent_id)


@router.post("/{agent_id}/tasks", response_model=TaskResponse)
@handle_agent_exceptions
async def add_task(agent_id: str, request: AddTaskRequest) -> TaskResponse:
    """Add a task to an agent"""
    task_id = await _agent_service.add_task(
        agent_id=agent_id,
        instruction=request.instruction,
        metadata=request.metadata
    )
    return await _agent_service.get_task(agent_id, task_id)


@router.get("/{agent_id}/tasks", response_model=List[TaskResponse])
@handle_agent_exceptions
async def list_tasks(agent_id: str, status: Optional[str] = None) -> List[TaskResponse]:
    """List tasks for an agent"""
    return await _agent_service.list_tasks(agent_id, status)


@router.get("", response_model=AgentListResponse)
async def list_agents() -> AgentListResponse:
    """List all active agents"""
    agents_list = await _agent_service.list_all_agents()
    return AgentListResponse(
        agents=agents_list,
        total=len(agents_list)
    )


@router.post("/stop-all", response_model=MessageResponse)
async def stop_all_agents() -> MessageResponse:
    """Stop all agents"""
    stopped_count = await _agent_service.stop_all_agents()
    return MessageResponse(
        message=f"Stopped {stopped_count} agents",
        success=True
    )

