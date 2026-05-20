"""
Health Check Module
Comprehensive health checking for agent components.
Ported from autonomous_long_term_agent/core/health_check.py
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details or {}
        }


async def execute_health_check(
    check_name: str,
    check_function: Callable[[], Awaitable[HealthCheck]]
) -> HealthCheck:
    """Execute a health check with consistent error handling."""
    try:
        return await check_function()
    except Exception as e:
        logger.error(f"Error checking {check_name}: {e}", exc_info=True)
        return HealthCheck(
            name=check_name,
            status=HealthStatus.UNKNOWN,
            message=f"Error checking {check_name}: {e}",
            details={"error": str(e)}
        )


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, check_interval_seconds: int = 30):
        self.checks: Dict[str, HealthCheck] = {}
        self.last_check_time: Optional[datetime] = None
        self.check_history: List[Dict[str, Any]] = []
        self.check_interval = check_interval_seconds
        self._max_history_size = 100
    
    def should_run_check(self) -> bool:
        """Check if enough time has passed for next check."""
        if not self.last_check_time:
            return True
        elapsed = (datetime.now() - self.last_check_time).total_seconds()
        return elapsed >= self.check_interval
    
    async def check_component_health(
        self,
        component_name: str,
        component: Any,
        get_stats_method: str = "get_stats"
    ) -> HealthCheck:
        """Check health of a component with stats method."""
        async def _perform_check() -> HealthCheck:
            if component is None:
                return HealthCheck(
                    name=component_name,
                    status=HealthStatus.DEGRADED,
                    message=f"{component_name} not initialized"
                )
            
            try:
                if hasattr(component, get_stats_method):
                    method = getattr(component, get_stats_method)
                    if callable(method):
                        stats = await method() if asyncio.iscoroutinefunction(method) else method()
                        return HealthCheck(
                            name=component_name,
                            status=HealthStatus.HEALTHY,
                            message=f"{component_name} operational",
                            details=stats
                        )
                
                return HealthCheck(
                    name=component_name,
                    status=HealthStatus.HEALTHY,
                    message=f"{component_name} available"
                )
            except Exception as e:
                return HealthCheck(
                    name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"{component_name} error: {e}",
                    details={"error": str(e)}
                )
        
        return await execute_health_check(component_name, _perform_check)
    
    async def check_agent_health(self, agent: Any) -> Dict[str, Any]:
        """Perform comprehensive agent health checks."""
        checks = {}
        
        # Check agent status
        checks["agent_status"] = await self._check_agent_status(agent)
        
        # Check components if available
        if hasattr(agent, 'components'):
            components = agent.components
            if hasattr(components, 'knowledge_base'):
                checks["knowledge_base"] = await self.check_component_health(
                    "knowledge_base", components.knowledge_base
                )
            if hasattr(components, 'learning_engine'):
                checks["learning_engine"] = await self.check_component_health(
                    "learning_engine", components.learning_engine
                )
            if hasattr(components, 'analytics'):
                checks["analytics"] = await self.check_component_health(
                    "analytics", components.analytics, "get_task_analytics"
                )
        
        # Calculate overall health
        overall_status = self._calculate_overall_health(checks)
        
        # Update history
        self._update_history(overall_status, checks)
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {name: check.to_dict() for name, check in checks.items()}
        }
    
    async def _check_agent_status(self, agent: Any) -> HealthCheck:
        """Check agent status."""
        async def _perform_check() -> HealthCheck:
            if not hasattr(agent, 'status'):
                return HealthCheck(
                    name="agent_status",
                    status=HealthStatus.UNKNOWN,
                    message="No status attribute"
                )
            
            status_value = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
            
            if status_value == "running":
                return HealthCheck(
                    name="agent_status",
                    status=HealthStatus.HEALTHY,
                    message="Agent is running",
                    details={"status": status_value}
                )
            elif status_value in ("idle", "paused"):
                return HealthCheck(
                    name="agent_status",
                    status=HealthStatus.DEGRADED,
                    message=f"Agent is {status_value}",
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
    
    def _calculate_overall_health(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Calculate overall health status."""
        if not checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in checks.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _update_history(self, status: HealthStatus, checks: Dict[str, HealthCheck]) -> None:
        """Update check history."""
        self.last_check_time = datetime.now()
        self.checks = checks
        
        entry = {
            "timestamp": self.last_check_time.isoformat(),
            "status": status.value,
            "checks": {name: check.status.value for name, check in checks.items()}
        }
        
        self.check_history.append(entry)
        if len(self.check_history) > self._max_history_size:
            self.check_history = self.check_history[-self._max_history_size:]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get check history."""
        return self.check_history.copy()


# Need asyncio for iscoroutinefunction
import asyncio
