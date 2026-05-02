"""
Celery tasks for executing continuous agents 24/7
"""
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from celery import shared_task
from sqlalchemy.orm import Session

from unified_core.background.celery.apps.app_base import task_logger
from unified_core.configs.app_configs import JOB_TIMEOUT
from unified_core.configs.constants import OnyxCeleryTask
from unified_core.db.continuous_agent import (
    create_continuous_agent_execution_log,
    fetch_active_continuous_agents,
    update_agent_stats,
)
from unified_core.db.engine.sql_engine import get_session_with_current_tenant
from unified_core.db.models import ContinuousAgent
from unified_core.utils.logger import setup_logger

logger = setup_logger()


def check_stripe_credits(user_id: UUID) -> int | None:
    """
    Check Stripe credits for a user.
    Returns None if Stripe is not configured or user has no credits.
    Returns the credit amount if available.
    """
    try:
        # Try to import EE implementation
        from unified_core.utils.variable_functionality import fetch_ee_implementation_or_noop

        check_credits = fetch_ee_implementation_or_noop(
            "ee.onyx.server.tenants.billing", "check_user_stripe_credits", None
        )
        if check_credits:
            return check_credits(user_id)
    except Exception as e:
        logger.warning(f"Could not check Stripe credits: {e}")
    return None


def execute_agent_task(agent: ContinuousAgent, db_session: Session) -> tuple[bool, int, int]:
    """
    Execute the actual task for an agent.
    Returns: (success, credits_used, execution_time_ms)
    """
    start_time = time.time()
    credits_used = 0

    try:
        task_type = agent.config.get("taskType", "custom")
        parameters = agent.config.get("parameters", {})

        # Execute based on task type
        if task_type == "content_generation":
            # Example: Generate content using LLM
            credits_used = 10  # Example credit cost
            logger.info(f"Agent {agent.id} executing content generation task")
            # TODO: Implement actual content generation

        elif task_type == "data_processing":
            credits_used = 5
            logger.info(f"Agent {agent.id} executing data processing task")
            # TODO: Implement actual data processing

        elif task_type == "api_monitoring":
            credits_used = 3
            logger.info(f"Agent {agent.id} executing API monitoring task")
            # TODO: Implement actual API monitoring

        elif task_type == "automated_research":
            credits_used = 15
            logger.info(f"Agent {agent.id} executing automated research task")
            # TODO: Implement actual research

        else:
            credits_used = 1
            logger.info(f"Agent {agent.id} executing custom task")
            # TODO: Implement custom task execution

        execution_time_ms = int((time.time() - start_time) * 1000)
        return True, credits_used, execution_time_ms

    except Exception as e:
        logger.error(f"Error executing agent task {agent.id}: {e}")
        execution_time_ms = int((time.time() - start_time) * 1000)
        return False, credits_used, execution_time_ms


@shared_task(
    name=OnyxCeleryTask.CHECK_CONTINUOUS_AGENTS_TASK,
    soft_time_limit=JOB_TIMEOUT,
    bind=True,
)
def check_continuous_agents_task(self: Any, tenant_id: str) -> int:
    """
    Periodic task that checks for active continuous agents and executes them.
    This runs every minute via Celery Beat.
    """
    executed_count = 0

    with get_session_with_current_tenant() as db_session:
        active_agents = fetch_active_continuous_agents(db_session)

        if not active_agents:
            task_logger.info("No active continuous agents to execute")
            return 0

        task_logger.info(f"Found {len(active_agents)} active continuous agents")

        for agent in active_agents:
            try:
                # Check if it's time to execute this agent
                frequency = agent.config.get("frequency", 3600)  # Default 1 hour
                last_execution_str = agent.stats.get("last_execution_at")

                if last_execution_str:
                    last_execution = datetime.fromisoformat(last_execution_str.replace("Z", "+00:00"))
                    time_since_last = (datetime.utcnow() - last_execution.replace(tzinfo=None)).total_seconds()

                    if time_since_last < frequency:
                        # Not time to execute yet
                        continue

                # Check Stripe credits before executing
                credits_remaining = check_stripe_credits(agent.user_id)
                if credits_remaining is not None and credits_remaining < 10:
                    # Not enough credits, skip execution
                    task_logger.warning(
                        f"Agent {agent.id} skipped: insufficient Stripe credits ({credits_remaining})"
                    )
                    log = create_continuous_agent_execution_log(
                        agent_id=agent.id,
                        status="skipped",
                        started_at=datetime.utcnow(),
                        credits_used=0,
                        execution_time_ms=0,
                        error=f"Insufficient Stripe credits: {credits_remaining}",
                        db_session=db_session,
                    )
                    continue

                # Execute the agent task
                task_logger.info(f"Executing agent {agent.id} ({agent.name})")
                success, credits_used, execution_time_ms = execute_agent_task(agent, db_session)

                # Create execution log
                log = create_continuous_agent_execution_log(
                    agent_id=agent.id,
                    status="success" if success else "failed",
                    started_at=datetime.utcnow(),
                    credits_used=credits_used,
                    execution_time_ms=execution_time_ms,
                    error=None if success else "Task execution failed",
                    db_session=db_session,
                )

                # Update agent stats
                update_agent_stats(agent, success, credits_used, execution_time_ms, db_session)

                # If credits are running low, deactivate agent
                if credits_remaining is not None:
                    remaining_after = credits_remaining - credits_used
                    if remaining_after < 10:
                        agent.is_active = False
                        task_logger.warning(
                            f"Agent {agent.id} deactivated due to low Stripe credits"
                        )
                        db_session.commit()

                executed_count += 1

            except Exception as e:
                task_logger.error(f"Error processing agent {agent.id}: {e}", exc_info=True)
                continue

    task_logger.info(f"Executed {executed_count} continuous agents")
    return executed_count


@shared_task(
    name=OnyxCeleryTask.EXECUTE_CONTINUOUS_AGENT_TASK,
    soft_time_limit=JOB_TIMEOUT,
    bind=True,
)
def execute_continuous_agent_task(self: Any, agent_id: str, tenant_id: str) -> dict[str, Any]:
    """
    Execute a specific continuous agent immediately.
    Used for manual triggers or testing.
    """
    with get_session_with_current_tenant() as db_session:
        from unified_core.db.models import ContinuousAgent
        from sqlalchemy import select

        stmt = select(ContinuousAgent).where(ContinuousAgent.id == UUID(agent_id))
        agent = db_session.scalar(stmt)

        if not agent:
            return {"success": False, "error": "Agent not found"}

        if not agent.is_active:
            return {"success": False, "error": "Agent is not active"}

        # Check credits
        credits_remaining = check_stripe_credits(agent.user_id)
        if credits_remaining is not None and credits_remaining < 10:
            return {
                "success": False,
                "error": f"Insufficient Stripe credits: {credits_remaining}",
            }

        # Execute
        success, credits_used, execution_time_ms = execute_agent_task(agent, db_session)

        # Log and update stats
        create_continuous_agent_execution_log(
            agent_id=agent.id,
            status="success" if success else "failed",
            started_at=datetime.utcnow(),
            credits_used=credits_used,
            execution_time_ms=execution_time_ms,
            error=None if success else "Task execution failed",
            db_session=db_session,
        )

        update_agent_stats(agent, success, credits_used, execution_time_ms, db_session)

        return {
            "success": success,
            "credits_used": credits_used,
            "execution_time_ms": execution_time_ms,
        }







