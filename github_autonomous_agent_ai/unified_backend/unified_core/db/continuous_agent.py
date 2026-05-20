"""
Database operations for Continuous Agents
"""
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from unified_core.db.models import ContinuousAgent
from unified_core.db.models import ContinuousAgentExecutionLog
from unified_core.utils.logger import setup_logger

logger = setup_logger()


def fetch_user_continuous_agents(
    user_id: UUID, db_session: Session
) -> list[ContinuousAgent]:
    """Fetch all continuous agents for a user"""
    stmt = select(ContinuousAgent).where(ContinuousAgent.user_id == user_id)
    return list(db_session.scalars(stmt).all())


def fetch_continuous_agent(
    agent_id: UUID, user_id: UUID, db_session: Session
) -> ContinuousAgent | None:
    """Fetch a specific continuous agent by ID"""
    stmt = (
        select(ContinuousAgent)
        .where(ContinuousAgent.id == agent_id, ContinuousAgent.user_id == user_id)
    )
    return db_session.scalar(stmt)


def fetch_active_continuous_agents(db_session: Session) -> list[ContinuousAgent]:
    """Fetch all active continuous agents (for Celery tasks)"""
    stmt = select(ContinuousAgent).where(ContinuousAgent.is_active == True)  # noqa: E712
    return list(db_session.scalars(stmt).all())


def create_continuous_agent_execution_log(
    agent_id: UUID,
    status: str,
    started_at: datetime,
    credits_used: int = 0,
    execution_time_ms: int = 0,
    error: str | None = None,
    db_session: Session | None = None,
) -> ContinuousAgentExecutionLog:
    """Create a new execution log entry"""
    log = ContinuousAgentExecutionLog(
        agent_id=agent_id,
        status=status,
        started_at=started_at,
        completed_at=datetime.utcnow() if status != "running" else None,
        credits_used=credits_used,
        execution_time_ms=execution_time_ms,
        error=error,
    )
    if db_session:
        db_session.add(log)
        db_session.commit()
    return log


def fetch_agent_execution_logs(
    agent_id: UUID, limit: int = 50, db_session: Session | None = None
) -> list[ContinuousAgentExecutionLog]:
    """Fetch execution logs for an agent"""
    if not db_session:
        return []

    stmt = (
        select(ContinuousAgentExecutionLog)
        .where(ContinuousAgentExecutionLog.agent_id == agent_id)
        .order_by(ContinuousAgentExecutionLog.started_at.desc())
        .limit(limit)
    )
    return list(db_session.scalars(stmt).all())


def update_agent_stats(
    agent: ContinuousAgent,
    success: bool,
    credits_used: int,
    execution_time_ms: int,
    db_session: Session,
) -> None:
    """Update agent statistics after execution"""
    agent.stats["total_executions"] = agent.stats.get("total_executions", 0) + 1

    if success:
        agent.stats["successful_executions"] = agent.stats.get("successful_executions", 0) + 1
    else:
        agent.stats["failed_executions"] = agent.stats.get("failed_executions", 0) + 1

    agent.stats["credits_used"] = agent.stats.get("credits_used", 0) + credits_used

    # Update average execution time
    total_executions = agent.stats.get("total_executions", 1)
    current_avg = agent.stats.get("average_execution_time", 0)
    new_avg = ((current_avg * (total_executions - 1)) + execution_time_ms) / total_executions
    agent.stats["average_execution_time"] = int(new_avg)

    agent.stats["last_execution_at"] = datetime.utcnow().isoformat()

    # Calculate next execution time
    from datetime import timedelta

    next_execution = datetime.utcnow() + timedelta(seconds=agent.config.get("frequency", 3600))
    agent.stats["next_execution_at"] = next_execution.isoformat()

    db_session.commit()







