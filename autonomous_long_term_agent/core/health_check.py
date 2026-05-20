"""
Health Check System for Autonomous Agent
Improved with better organization and error handling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Protocol, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from .health_check_helpers import execute_health_check

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class HealthCheckProvider(Protocol):
    """Protocol for components that can provide health information"""
    
    async def get_health_info(self) -> Dict[str, Any]:
        """Get health information"""
        ...


class HealthChecker:
    """
    Comprehensive health checking system
    Improved with better organization and extensibility
    """
    
    def __init__(self, check_interval_seconds: int = 30):
        self.checks: Dict[str, HealthCheck] = {}
        self.last_check_time: Optional[datetime] = None
        self.check_history: List[Dict[str, Any]] = []
        self.check_interval = check_interval_seconds
        self._max_history_size = 100
    
    async def check_agent_health(
        self,
        agent: Any,
        openrouter_client: Any
    ) -> Dict[str, Any]:
        """
        Perform comprehensive health checks
        
        Args:
            agent: Agent instance to check
            openrouter_client: OpenRouter client to check
        
        Returns:
            Dictionary with health status and checks
        """
        checks = {}
        
        # Run all health checks
        checks["agent_status"] = await self._check_agent_status(agent)
        checks["openrouter"] = await self._check_openrouter(openrouter_client)
        checks["knowledge_base"] = await self._check_knowledge_base(agent.knowledge_base)
        checks["task_queue"] = await self._check_task_queue(agent.task_queue)
        checks["learning_engine"] = await self._check_learning_engine(agent.learning_engine)
        
        # Calculate overall health
        overall_status = self._calculate_overall_health(checks)
        
        # Update history
        self._update_history(overall_status, checks)
        
        return {
            "overall_status": overall_status.value,
            "timestamp": self.last_check_time.isoformat() if self.last_check_time else datetime.utcnow().isoformat(),
            "checks": {
                name: check.to_dict()
                for name, check in checks.items()
            }
        }
    
    async def _check_agent_status(self, agent: Any) -> HealthCheck:
        """Check agent status"""
        async def _perform_check() -> HealthCheck:
            status_value = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
            
            if status_value == "running":
                return HealthCheck(
                    name="agent_status",
                    status=HealthStatus.HEALTHY,
                    message="Agent is running",
                    details={"status": status_value}
                )
            elif status_value == "paused":
                return HealthCheck(
                    name="agent_status",
                    status=HealthStatus.DEGRADED,
                    message="Agent is paused",
                    details={"status": status_value}
                )
            else:
                return HealthCheck(
                    name="agent_status",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Agent status: {status_value}",
                    details={"status": status_value}
                )
        
        return await execute_health_check("agent_status", _perform_check)
    
    async def _check_openrouter(self, client: Any) -> HealthCheck:
        """Check OpenRouter connectivity"""
        async def _perform_check() -> HealthCheck:
            if hasattr(client, 'get_resilience_stats'):
                stats = client.get_resilience_stats()
                circuit_state = stats.get("circuit_breaker", {}).get("state", "unknown")
                
                if circuit_state == "open":
                    return HealthCheck(
                        name="openrouter",
                        status=HealthStatus.UNHEALTHY,
                        message="OpenRouter circuit breaker is OPEN",
                        details={"circuit_state": circuit_state, **stats}
                    )
                elif circuit_state == "half_open":
                    return HealthCheck(
                        name="openrouter",
                        status=HealthStatus.DEGRADED,
                        message="OpenRouter circuit breaker is HALF_OPEN",
                        details={"circuit_state": circuit_state, **stats}
                    )
            
            return HealthCheck(
                name="openrouter",
                status=HealthStatus.HEALTHY,
                message="OpenRouter connectivity OK",
                details={
                    "api_key_configured": bool(getattr(client, 'api_key', None))
                }
            )
        
        return await execute_health_check("openrouter", _perform_check)
    
    async def _check_knowledge_base(self, knowledge_base: Any) -> HealthCheck:
        """Check knowledge base health"""
        async def _perform_check() -> HealthCheck:
            stats = await knowledge_base.get_stats()
            total_entries = stats.get("total_entries", 0)
            
            if total_entries > 0:
                return HealthCheck(
                    name="knowledge_base",
                    status=HealthStatus.HEALTHY,
                    message=f"Knowledge base has {total_entries} entries",
                    details=stats
                )
            else:
                return HealthCheck(
                    name="knowledge_base",
                    status=HealthStatus.DEGRADED,
                    message="Knowledge base is empty",
                    details=stats
                )
        
        return await execute_health_check("knowledge_base", _perform_check)
    
    async def _check_task_queue(self, task_queue: Any) -> HealthCheck:
        """Check task queue health"""
        async def _perform_check() -> HealthCheck:
            queue_size = await task_queue.get_queue_size()
            tasks = await task_queue.list_tasks(limit=10)
            
            failed_tasks = sum(1 for t in tasks if hasattr(t, 'status') and t.status.value == "failed")
            
            if queue_size > 100:
                return HealthCheck(
                    name="task_queue",
                    status=HealthStatus.DEGRADED,
                    message=f"Task queue has {queue_size} pending tasks",
                    details={
                        "queue_size": queue_size,
                        "failed_tasks": failed_tasks,
                        "warning": "Queue size exceeds recommended limit"
                    }
                )
            else:
                return HealthCheck(
                    name="task_queue",
                    status=HealthStatus.HEALTHY,
                    message=f"Task queue healthy ({queue_size} tasks)",
                    details={
                        "queue_size": queue_size,
                        "failed_tasks": failed_tasks
                    }
                )
        
        return await execute_health_check("task_queue", _perform_check)
    
    async def _check_learning_engine(self, learning_engine: Any) -> HealthCheck:
        """Check learning engine health"""
        async def _perform_check() -> HealthCheck:
            stats = await learning_engine.get_learning_stats()
            total_events = stats.get("total_events", 0)
            learning_enabled = stats.get("learning_enabled", False)
            
            if learning_enabled and total_events > 0:
                return HealthCheck(
                    name="learning_engine",
                    status=HealthStatus.HEALTHY,
                    message=f"Learning engine active ({total_events} events)",
                    details=stats
                )
            elif learning_enabled:
                return HealthCheck(
                    name="learning_engine",
                    status=HealthStatus.DEGRADED,
                    message="Learning engine enabled but no events recorded",
                    details=stats
                )
            else:
                return HealthCheck(
                    name="learning_engine",
                    status=HealthStatus.DEGRADED,
                    message="Learning engine disabled",
                    details=stats
                )
        
        return await execute_health_check("learning_engine", _perform_check)
    
    def _calculate_overall_health(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Calculate overall health status"""
        if not checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in checks.values()]
        
        # Priority: UNHEALTHY > UNKNOWN > DEGRADED > HEALTHY
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _update_history(self, overall_status: HealthStatus, checks: Dict[str, HealthCheck]) -> None:
        """Update health check history"""
        self.last_check_time = datetime.utcnow()
        
        self.check_history.append({
            "timestamp": self.last_check_time.isoformat(),
            "overall_status": overall_status.value,
            "checks": {k: v.status.value for k, v in checks.items()}
        })
        
        # Keep only last N checks
        if len(self.check_history) > self._max_history_size:
            self.check_history = self.check_history[-self._max_history_size:]
    
    def get_check_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health check history"""
        return self.check_history[-limit:]
    
    def should_run_check(self) -> bool:
        """Check if health check should run based on interval"""
        if not self.last_check_time:
            return True
        
        elapsed = (datetime.utcnow() - self.last_check_time).total_seconds()
        return elapsed >= self.check_interval
