"""
Agent Observers
Handles observation recording for optional learning components
Follows Observer pattern and Single Responsibility Principle
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, List
from abc import ABC, abstractmethod

from .async_helpers import safe_async_call

logger = logging.getLogger(__name__)


class TaskObserver(ABC):
    """Abstract observer for task events"""
    
    @abstractmethod
    async def on_task_success(self, task: Any, result: Any) -> None:
        """Called when a task succeeds"""
        pass
    
    @abstractmethod
    async def on_task_failure(self, task: Any, error: str) -> None:
        """Called when a task fails"""
        pass


class ExperienceLearningObserver(TaskObserver):
    """Observer for experience-driven learning"""
    
    def __init__(self, experience_learning, knowledge_base=None):
        self.experience_learning = experience_learning
        self.knowledge_base = knowledge_base
    
    async def on_task_success(self, task: Any, result: Any) -> None:
        """Record successful task experience"""
        if not self.experience_learning:
            return
        
        # Record experience safely
        experience = await safe_async_call(
            self.experience_learning.record_experience,
            interaction_type="task_processing",
            context={
                "task_id": task.id,
                "instruction": task.instruction,
                "metadata": getattr(task, 'metadata', None)
            },
            outcome="success",
            error_message=f"Error recording experience for task {task.id}"
        )
        
        # Internalize knowledge from experience if available
        if experience and self.knowledge_base:
            await safe_async_call(
                self.experience_learning.internalize_knowledge,
                experience,
                self.knowledge_base,
                error_message=f"Error internalizing knowledge for task {task.id}"
            )
    
    async def on_task_failure(self, task: Any, error: str) -> None:
        """Record failed task experience"""
        if not self.experience_learning:
            return
        
        await safe_async_call(
            self.experience_learning.record_experience,
            interaction_type="task_processing",
            context={
                "task_id": task.id,
                "instruction": task.instruction,
                "error": error
            },
            outcome="failure",
            error_message=f"Error recording failed experience for task {task.id}"
        )


class WorldModelObserver(TaskObserver):
    """Observer for world model updates"""
    
    def __init__(self, world_model):
        self.world_model = world_model
    
    async def on_task_success(self, task: Any, result: Any) -> None:
        """Record successful task in world model"""
        if not self.world_model:
            return
        
        await safe_async_call(
            self.world_model.observe,
            observation_type="task_success",
            data={
                "task_id": task.id,
                "success": True,
                "result": result if isinstance(result, dict) else {}
            },
            confidence=0.7,
            error_message=f"Error recording world observation for task {task.id}"
        )
    
    async def on_task_failure(self, task: Any, error: str) -> None:
        """Record failed task in world model"""
        if not self.world_model:
            return
        
        await safe_async_call(
            self.world_model.observe,
            observation_type="task_failure",
            data={
                "task_id": task.id,
                "error": error,
                "success": False
            },
            confidence=0.3,
            error_message=f"Error recording world observation for task {task.id}"
        )


class AgentObserverManager:
    """
    Manages all observers for agent events.
    Centralizes observer pattern implementation.
    """
    
    def __init__(
        self,
        experience_learning=None,
        world_model=None,
        knowledge_base=None
    ):
        self._observers: List[TaskObserver] = []
        
        if experience_learning:
            self._observers.append(ExperienceLearningObserver(
                experience_learning,
                knowledge_base=knowledge_base
            ))
        
        if world_model:
            self._observers.append(WorldModelObserver(world_model))
    
    async def notify_task_success(self, task: Any, result: Any) -> None:
        """Notify all observers of task success"""
        for observer in self._observers:
            try:
                await observer.on_task_success(task, result)
            except Exception as e:
                logger.warning(f"Error in observer {observer.__class__.__name__}: {e}")
    
    async def notify_task_failure(self, task: Any, error: str) -> None:
        """Notify all observers of task failure"""
        for observer in self._observers:
            try:
                await observer.on_task_failure(task, error)
            except Exception as e:
                logger.warning(f"Error in observer {observer.__class__.__name__}: {e}")

