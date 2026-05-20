"""
Task Processor
Handles task processing logic, extracted from AutonomousLongTermAgent
"""

import logging
from typing import Dict, Any, Optional

from .task_queue import Task
from .reasoning_engine import ReasoningEngine, ReasoningResult
from .learning_engine import LearningEngine
from .metrics_manager import MetricsManager
from .agent_observers import AgentObserverManager
from ..infrastructure.storage.knowledge_base import KnowledgeBase
from .agent_utils import reasoning_result_to_dict
from .async_helpers import safe_async_call

logger = logging.getLogger(__name__)


class TaskProcessor:
    """
    Handles task processing logic
    
    Separated from AutonomousLongTermAgent for better organization and testability.
    Responsible for:
    - Processing tasks using reasoning engine
    - Storing knowledge from completed tasks
    - Recording learning events
    - Notifying observers of task outcomes
    - Updating metrics
    
    This separation follows Single Responsibility Principle and makes the code
    more maintainable and testable.
    """
    
    def __init__(
        self,
        reasoning_engine: ReasoningEngine,
        learning_engine: LearningEngine,
        knowledge_base: KnowledgeBase,
        metrics_manager: MetricsManager,
        observer_manager: AgentObserverManager
    ):
        """
        Initialize TaskProcessor
        
        Args:
            reasoning_engine: Engine for long-horizon reasoning
            learning_engine: Engine for self-initiated learning
            knowledge_base: Storage for accumulated knowledge
            metrics_manager: Manager for agent metrics
            observer_manager: Manager for task event observers
        """
        self.reasoning_engine = reasoning_engine
        self.learning_engine = learning_engine
        self.knowledge_base = knowledge_base
        self.metrics_manager = metrics_manager
        self.observer_manager = observer_manager
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a single task
        
        Args:
            task: Task to process
        
        Returns:
            Result dictionary with response and metadata
        
        Raises:
            Exception: If task processing fails
        """
        logger.info(f"Processing task {task.id}: {task.instruction[:50]}...")
        
        # Use long-horizon reasoning
        reasoning_result = await self.reasoning_engine.reason(
            instruction=task.instruction,
            metadata=task.metadata
        )
        
        # Convert to dict format
        result = reasoning_result_to_dict(reasoning_result)
        
        # Store knowledge
        await self._store_task_knowledge(task, reasoning_result)
        
        # Record learning event
        await self._record_task_completion(task.id, "success")
        
        # Notify observers of task success
        await self.observer_manager.notify_task_success(task, result)
        
        # Update metrics
        self.metrics_manager.record_task_completed(
            tokens_used=reasoning_result.tokens_used
        )
        self.metrics_manager.record_reasoning_call()
        
        return result
    
    async def handle_task_error(
        self,
        task: Task,
        error: Exception
    ) -> None:
        """
        Handle task processing error
        
        Args:
            task: Task that failed
            error: Exception that occurred
        """
        error_str = str(error)
        logger.error(f"Error processing task {task.id}: {error_str}", exc_info=True)
        
        # Notify observers of task failure
        await self.observer_manager.notify_task_failure(task, error_str)
        
        # Record learning event
        await self.learning_engine.record_event(
            "task_failed",
            {"task_id": task.id, "error": error_str},
            "failure"
        )
        
        # Update metrics
        self.metrics_manager.record_task_failed()
    
    async def _store_task_knowledge(
        self,
        task: Task,
        reasoning_result: ReasoningResult
    ) -> None:
        """
        Store knowledge from completed task in knowledge base.
        
        Extracts the reasoning result and task context to build knowledge
        entries that can be retrieved for future tasks. This enables the
        agent to learn from past experiences.
        
        Args:
            task: Completed task
            reasoning_result: Result from reasoning engine
        """
        result = await safe_async_call(
            self.knowledge_base.add_knowledge,
            content=reasoning_result.response,
            context={
                "task_id": task.id,
                "instruction": task.instruction,
                "metadata": task.metadata,
                "confidence": reasoning_result.confidence
            },
            error_message=f"Error storing knowledge for task {task.id}"
        )
        
        if result is not None:
            self.metrics_manager.record_knowledge_retrieval()
    
    async def _record_task_completion(
        self,
        task_id: str,
        outcome: str
    ) -> None:
        """
        Record task completion in learning engine.
        
        Tracks task completion events to enable the learning engine to
        adapt and improve based on task outcomes.
        
        Args:
            task_id: ID of completed task
            outcome: Outcome of task ("success" or "failure")
        """
        await safe_async_call(
            self.learning_engine.record_event,
            "task_completed",
            {"task_id": task_id},
            outcome,
            error_message=f"Error recording task completion for {task_id}"
        )

