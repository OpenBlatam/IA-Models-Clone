"""
Pydantic schemas for Continuous Agents API
"""
from typing import Any
from pydantic import BaseModel
from unified_core.db.models import ContinuousAgent, ContinuousAgentExecutionLog

class AgentConfigRequest(BaseModel):
    taskType: str
    frequency: int
    parameters: dict[str, Any] = {}

class CreateAgentRequest(BaseModel):
    name: str
    description: str
    config: AgentConfigRequest

class UpdateAgentRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    config: AgentConfigRequest | None = None
    is_active: bool | None = None

class AgentStatsResponse(BaseModel):
    total_executions: int
    successful_executions: int
    failed_executions: int
    last_execution_at: str | None
    next_execution_at: str | None
    credits_used: int
    average_execution_time: int

class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    is_active: bool
    created_at: str
    updated_at: str
    config: dict[str, Any]
    stats: AgentStatsResponse
    stripe_credits_remaining: int | None = None

    @classmethod
    def from_model(cls, agent: ContinuousAgent, credits: int | None = None) -> "AgentResponse":
        stats = agent.stats or {}
        return cls(
            id=str(agent.id),
            name=agent.name,
            description=agent.description,
            is_active=agent.is_active,
            created_at=agent.created_at.isoformat(),
            updated_at=agent.updated_at.isoformat(),
            config=agent.config,
            stats=AgentStatsResponse(
                total_executions=stats.get("total_executions", 0),
                successful_executions=stats.get("successful_executions", 0),
                failed_executions=stats.get("failed_executions", 0),
                last_execution_at=stats.get("last_execution_at"),
                next_execution_at=stats.get("next_execution_at"),
                credits_used=stats.get("credits_used", 0),
                average_execution_time=stats.get("average_execution_time", 0),
            ),
            stripe_credits_remaining=credits,
        )

class ExecutionLogResponse(BaseModel):
    id: str
    agent_id: str
    status: str
    started_at: str
    completed_at: str | None
    error: str | None
    credits_used: int
    execution_time: int

    @classmethod
    def from_model(cls, log: ContinuousAgentExecutionLog) -> "ExecutionLogResponse":
        return cls(
            id=str(log.id),
            agent_id=str(log.agent_id),
            status=log.status,
            started_at=log.started_at.isoformat(),
            completed_at=log.completed_at.isoformat() if log.completed_at else None,
            error=log.error,
            credits_used=log.credits_used,
            execution_time=log.execution_time_ms,
        )
