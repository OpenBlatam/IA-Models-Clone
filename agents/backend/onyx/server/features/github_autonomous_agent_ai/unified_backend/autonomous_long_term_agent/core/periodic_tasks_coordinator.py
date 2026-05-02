"""
Periodic Tasks Coordinator
Coordinates periodic tasks (health checks, self-reflection, metrics updates)
"""

import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Dict, Any

from .health_check import HealthChecker
from .metrics_manager import MetricsManager
from .self_reflection import SelfReflectionEngine
from .task_queue import TaskQueue
from .async_helpers import safe_async_call
from ..config import settings
from .task_utils import tasks_to_dict_list

if TYPE_CHECKING:
    from ..infrastructure.openrouter.client import OpenRouterClient
    from .agent import AutonomousLongTermAgent

logger = logging.getLogger(__name__)


class PeriodicTasksCoordinator:
    """
    Coordinates periodic tasks execution
    Separated from AutonomousLongTermAgent for better organization
    """
    
    def __init__(
        self,
        health_checker: HealthChecker,
        metrics_manager: MetricsManager,
        self_reflection_engine: Optional[SelfReflectionEngine],
        task_queue: TaskQueue,
        agent_id: str
    ):
        self.health_checker = health_checker
        self.metrics_manager = metrics_manager
        self.self_reflection_engine = self_reflection_engine
        self.task_queue = task_queue
        self.agent_id = agent_id
        
        self._last_health_check: Optional[datetime] = None
        self._last_reflection: Optional[datetime] = None
    
    async def execute_periodic_tasks(
        self,
        agent: "AutonomousLongTermAgent",
        openrouter_client: "OpenRouterClient"
    ) -> None:
        """
        Execute all periodic tasks
        
        Args:
            agent: Agent instance for health checks
            openrouter_client: OpenRouter client for health checks
        """
        # Update metrics
        await self._update_metrics()
        
        # Periodic health checks
        await self._perform_health_check(agent, openrouter_client)
        
        # Periodic self-reflection
        await self._perform_self_reflection()
    
    async def _update_metrics(self) -> None:
        """Update agent metrics"""
        self.metrics_manager.update_uptime()
    
    async def _perform_health_check(
        self,
        agent: "AutonomousLongTermAgent",
        openrouter_client: "OpenRouterClient"
    ) -> None:
        """Perform periodic health checks"""
        if not self.health_checker.should_run_check():
            return
        
        result = await safe_async_call(
            self.health_checker.check_agent_health,
            agent,
            openrouter_client,
            error_message=f"Error in health check for agent {self.agent_id}"
        )
        
        if result is not None:
            self._last_health_check = datetime.utcnow()
    
    async def _perform_self_reflection(self) -> None:
        """
        Perform periodic self-reflection (EvoAgent paper).
        
        Coordinates all types of self-reflection:
        - Performance reflection
        - Capabilities reflection
        - Periodic reflection
        
        Only runs if reflection interval has elapsed.
        """
        if not self.self_reflection_engine:
            return
        
        # Check if enough time has passed since last reflection
        if not self._should_run_reflection():
            return
        
        # Prepare reflection data
        metrics = self.metrics_manager.get_metrics_dict()
        recent_tasks_dict = await self._get_recent_tasks_for_reflection()
        
        # Execute all enabled reflection types
        await self._reflect_on_performance(metrics, recent_tasks_dict)
        await self._reflect_on_capabilities(recent_tasks_dict)
        await self._perform_periodic_reflection()
        
        # Update last reflection time
        self._last_reflection = datetime.utcnow()
        logger.debug(f"Self-reflection completed for agent {self.agent_id}")
    
    def _should_run_reflection(self) -> bool:
        """
        Check if reflection should run based on interval.
        
        Returns:
            True if reflection should run, False otherwise
        """
        now = datetime.utcnow()
        if not self._last_reflection:
            return True
        
        elapsed = (now - self._last_reflection).total_seconds()
        return elapsed >= settings.self_reflection_interval
    
    async def _get_recent_tasks_for_reflection(self) -> list:
        """
        Get recent tasks formatted for reflection.
        
        Returns:
            List of task dictionaries
        """
        recent_tasks = await safe_async_call(
            self.task_queue.get_recent_tasks,
            limit=10,
            default=[],
            error_message=f"Error getting recent tasks for reflection (agent {self.agent_id})"
        )
        return tasks_to_dict_list(recent_tasks) if recent_tasks else []
    
    async def _reflect_on_performance(
        self,
        metrics: Dict[str, Any],
        recent_tasks: list
    ) -> None:
        """
        Reflect on agent performance.
        
        Args:
            metrics: Current agent metrics
            recent_tasks: Recent tasks for context
        """
        if not settings.self_reflection_on_performance:
            return
        
        await safe_async_call(
            self.self_reflection_engine.reflect_on_performance,
            metrics=metrics,
            recent_tasks=recent_tasks,
            error_message=f"Error in performance reflection (agent {self.agent_id})"
        )
    
    async def _reflect_on_capabilities(self, recent_tasks: list) -> None:
        """
        Reflect on agent capabilities.
        
        Args:
            recent_tasks: Recent tasks for context
        """
        if not settings.self_reflection_on_capabilities:
            return
        
        capabilities = {
            "reasoning": True,
            "learning": settings.learning_enabled,
            "knowledge_retrieval": True
        }
        task_requirements = [
            {"type": "reasoning", "complexity": "medium"}
            for _ in recent_tasks[:5]
        ]
        
        await safe_async_call(
            self.self_reflection_engine.reflect_on_capabilities,
            capabilities=capabilities,
            task_requirements=task_requirements,
            error_message=f"Error in capabilities reflection (agent {self.agent_id})"
        )
    
    async def _perform_periodic_reflection(self) -> None:
        """Perform general periodic reflection."""
        await safe_async_call(
            self.self_reflection_engine.periodic_reflection,
            error_message=f"Error in periodic reflection (agent {self.agent_id})"
        )

