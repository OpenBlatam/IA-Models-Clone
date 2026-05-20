"""
Loop Coordinator
Coordinates the main agent execution loop
Extracted for better testability and organization
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING, Callable

from ..config import settings
from .agent import AgentStatus
from .task_queue import TaskQueue, Task
from .task_processor import TaskProcessor
from .autonomous_operation_handler import AutonomousOperationHandler
from .periodic_tasks_coordinator import PeriodicTasksCoordinator
from .async_helpers import safe_async_call

if TYPE_CHECKING:
    from .agent import AutonomousLongTermAgent
    from ..infrastructure.openrouter.client import OpenRouterClient

logger = logging.getLogger(__name__)


class LoopCoordinator:
    """
    Coordinates the main agent execution loop
    
    Separated from AutonomousLongTermAgent for better organization.
    Handles the main loop logic, task processing, and periodic operations.
    """
    
    def __init__(
        self,
        task_queue: TaskQueue,
        task_processor: TaskProcessor,
        autonomous_handler: AutonomousOperationHandler,
        periodic_coordinator: PeriodicTasksCoordinator,
        agent_id: str,
        status_getter: Callable[[], AgentStatus],
        stop_event: asyncio.Event
    ):
        """
        Initialize LoopCoordinator
        
        Args:
            task_queue: Task queue to get tasks from
            task_processor: Processor for tasks
            autonomous_handler: Handler for autonomous operations
            periodic_coordinator: Coordinator for periodic tasks
            agent_id: Agent identifier for logging
            status_getter: Callable to get current agent status
            stop_event: Event to signal loop stop
        """
        self.task_queue = task_queue
        self.task_processor = task_processor
        self.autonomous_handler = autonomous_handler
        self.periodic_coordinator = periodic_coordinator
        self.agent_id = agent_id
        self.status_getter = status_getter
        self.stop_event = stop_event
    
    async def run_loop_iteration(
        self,
        agent: "AutonomousLongTermAgent",
        openrouter_client: "OpenRouterClient"
    ) -> None:
        """
        Execute one iteration of the main agent loop.
        
        Loop iteration flow:
        1. Check if agent is paused (skip if paused)
        2. Process next task if available, otherwise execute autonomous operations
        3. Execute periodic tasks (health checks, reflection, metrics)
        4. Sleep before next iteration
        
        Args:
            agent: Agent instance for periodic tasks
            openrouter_client: OpenRouter client for periodic tasks
        """
        # Check if paused - skip processing if paused
        status = self.status_getter()
        if status == AgentStatus.PAUSED:
            await asyncio.sleep(settings.agent_poll_interval)
            return
        
        # Process tasks or execute autonomous operation
        task = await self.task_queue.get_next_task()
        if task:
            await self._process_task_safely(task)
        else:
            # No tasks available - execute autonomous operations
            # This allows the agent to continue learning and planning
            await self.autonomous_handler.execute()
        
        # Execute periodic tasks (health checks, reflection, metrics)
        # These run regardless of whether a task was processed
        await self.periodic_coordinator.execute_periodic_tasks(
            agent,
            openrouter_client
        )
        
        # Sleep before next iteration to avoid busy-waiting
        await asyncio.sleep(settings.agent_poll_interval)
    
    async def _process_task_safely(self, task: Task) -> None:
        """
        Process a task with comprehensive error handling.
        
        This method ensures that:
        - Task processing errors are caught and handled
        - Task queue is updated with completion or failure status
        - Error information is properly logged and recorded
        
        Args:
            task: Task to process
        """
        try:
            result = await self.task_processor.process_task(task)
            await self.task_queue.complete_task(task.id, result)
        except Exception as e:
            await self.task_processor.handle_task_error(task, e)
            await self.task_queue.fail_task(task.id, str(e))

