"""
API endpoints for Continuous Agents
"""
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session

from unified_core.auth.users import current_user
from unified_core.db.engine.sql_engine import get_session
from unified_core.db.models import User
from unified_core.utils.logger import setup_logger

from .schemas import (
    CreateAgentRequest,
    UpdateAgentRequest,
    AgentResponse,
    ExecutionLogResponse
)
from . import service

logger = setup_logger()

router = APIRouter(prefix="/continuous-agent", tags=["continuous-agent"])

@router.get("")
def get_agents(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[AgentResponse]:
    """Get all continuous agents for the current user"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    agents = service.get_user_agents(user.id, db_session)
    credits = service.check_stripe_credits(user.id)
    return [AgentResponse.from_model(agent, credits) for agent in agents]


@router.post("")
def create_agent(
    request: CreateAgentRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> AgentResponse:
    """Create a new continuous agent"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    agent = service.create_new_agent(user.id, request, db_session)
    credits = service.check_stripe_credits(user.id)
    return AgentResponse.from_model(agent, credits)


@router.get("/{agent_id}")
def get_agent(
    agent_id: str,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> AgentResponse:
    """Get a specific continuous agent"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    agent = service.get_agent_by_id(UUID(agent_id), user.id, db_session)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    credits = service.check_stripe_credits(user.id)
    return AgentResponse.from_model(agent, credits)


@router.patch("/{agent_id}")
def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> AgentResponse:
    """Update a continuous agent"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    agent = service.get_agent_by_id(UUID(agent_id), user.id, db_session)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = service.update_existing_agent(agent, request, db_session)
    credits = service.check_stripe_credits(user.id)
    
    return AgentResponse.from_model(agent, credits)


@router.delete("/{agent_id}")
def delete_agent(
    agent_id: str,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> dict[str, str]:
    """Delete a continuous agent"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    agent = service.get_agent_by_id(UUID(agent_id), user.id, db_session)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    service.delete_existing_agent(agent, db_session)

    return {"message": "Agent deleted successfully"}


@router.get("/{agent_id}/logs")
def get_agent_logs(
    agent_id: str,
    limit: int = 50,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[ExecutionLogResponse]:
    """Get execution logs for an agent"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    agent = service.get_agent_by_id(UUID(agent_id), user.id, db_session)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    logs = service.get_agent_logs_by_id(UUID(agent_id), limit, db_session)
    return [ExecutionLogResponse.from_model(log) for log in logs]


@router.get("/stripe/credits")
def get_stripe_credits(
    user: User | None = Depends(current_user),
) -> dict[str, int | None]:
    """Get Stripe credits for the current user"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    credits = service.check_stripe_credits(user.id)
    return {"credits": credits}
