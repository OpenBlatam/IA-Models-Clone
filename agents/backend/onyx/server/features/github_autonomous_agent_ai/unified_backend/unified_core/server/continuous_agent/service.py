"""
Business logic and services for Continuous Agents
"""
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from unified_core.db.continuous_agent import (
    fetch_agent_execution_logs,
    fetch_continuous_agent,
    fetch_user_continuous_agents,
)
from unified_core.db.models import ContinuousAgent
from unified_core.utils.logger import setup_logger

from .schemas import CreateAgentRequest, UpdateAgentRequest

logger = setup_logger()

def check_stripe_credits(user_id: UUID) -> int | None:
    """Check Stripe credits for a user"""
    try:
        from unified_core.utils.variable_functionality import fetch_ee_implementation_or_noop

        check_credits = fetch_ee_implementation_or_noop(
            "ee.onyx.server.tenants.billing", "check_user_stripe_credits", None
        )
        if check_credits:
            return check_credits(user_id)
    except Exception as e:
        logger.warning(f"Failed to check stripe credits: {e}")
    return None

def get_user_agents(user_id: UUID, db_session: Session) -> list[ContinuousAgent]:
    """Get all continuous agents for a user"""
    return fetch_user_continuous_agents(user_id, db_session)

def get_agent_by_id(agent_id: UUID, user_id: UUID, db_session: Session) -> ContinuousAgent | None:
    """Fetch a specific continuous agent by ID for a user"""
    return fetch_continuous_agent(agent_id, user_id, db_session)

def create_new_agent(
    user_id: UUID, 
    request: CreateAgentRequest, 
    db_session: Session
) -> ContinuousAgent:
    """Create a new continuous agent"""
    agent = ContinuousAgent(
        user_id=user_id,
        name=request.name,
        description=request.description,
        is_active=True,
        config={
            "taskType": request.config.taskType,
            "frequency": request.config.frequency,
            "parameters": request.config.parameters,
        },
        stats={
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "last_execution_at": None,
            "next_execution_at": None,
            "credits_used": 0,
            "average_execution_time": 0,
        },
    )

    db_session.add(agent)
    db_session.commit()
    db_session.refresh(agent)
    
    return agent

def update_existing_agent(
    agent: ContinuousAgent,
    request: UpdateAgentRequest,
    db_session: Session
) -> ContinuousAgent:
    """Update an existing continuous agent"""
    if request.name is not None:
        agent.name = request.name
    if request.description is not None:
        agent.description = request.description
    if request.is_active is not None:
        agent.is_active = request.is_active
    if request.config is not None:
        agent.config = {
            "taskType": request.config.taskType,
            "frequency": request.config.frequency,
            "parameters": request.config.parameters,
        }

    agent.updated_at = datetime.utcnow()
    db_session.commit()
    db_session.refresh(agent)
    
    return agent

def delete_existing_agent(agent: ContinuousAgent, db_session: Session) -> None:
    """Delete a continuous agent"""
    db_session.delete(agent)
    db_session.commit()

def get_agent_logs_by_id(agent_id: UUID, limit: int, db_session: Session):
    """Fetch execution logs for a continuous agent"""
    return fetch_agent_execution_logs(agent_id, limit, db_session)
