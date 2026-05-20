"""
Autonomous Long-Term Agent
Implements concepts from papers on long-horizon agents and continual learning
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..config import settings
from ..infrastructure.openrouter.client import get_openrouter_client
from ..infrastructure.storage.knowledge_base import KnowledgeBase
from .task_queue import TaskQueue, Task, TaskStatus
from .learning_engine import LearningEngine
from .health_check import HealthChecker
from .reasoning_engine import ReasoningEngine, ReasoningResult
from .metrics_manager import MetricsManager
from .self_reflection import SelfReflectionEngine
from .experience_driven_learning import ExperienceDrivenLearning
from .world_model import ContinualWorldModel
from .agent_observers import AgentObserverManager
from .agent_status_collector import StatusCollector
from .task_utils import tasks_to_dict_list
from .task_processor import TaskProcessor
from .autonomous_operation_handler import AutonomousOperationHandler
from .periodic_tasks_coordinator import PeriodicTasksCoordinator
from .loop_coordinator import LoopCoordinator
from .component_initializer import ComponentInitializer

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# AgentMetrics moved to metrics_manager.py


class AutonomousLongTermAgent:
    """
    Autonomous agent that runs continuously until explicitly stopped
    Implements:
    - Long-horizon reasoning (WebResearcher paper)
    - Continual learning (SOLA paper)
    - Self-initiated adaptation
    
    For enhanced version with paper-based optimizations, use:
    - EnhancedAutonomousAgent (if available)
    - Or set enable_papers=True in config
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        instruction: Optional[str] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.instruction = instruction or "Operate autonomously and continuously learn"
        self.status = AgentStatus.IDLE
        self.task_queue = TaskQueue()
        self.knowledge_base = KnowledgeBase(
            storage_path=f"{settings.storage_path}/knowledge/{self.agent_id}",
            max_entries=settings.max_knowledge_entries,
            retention_days=settings.knowledge_base_retention_days
        )
        self.learning_engine = LearningEngine(
            adaptation_rate=settings.learning_adaptation_rate,
            learning_enabled=settings.learning_enabled
        )
        self.openrouter_client = get_openrouter_client()
        self.reasoning_engine = ReasoningEngine(
            knowledge_base=self.knowledge_base,
            openrouter_client=self.openrouter_client
        )
        self._stop_event = asyncio.Event()
        self._running_task: Optional[asyncio.Task] = None
        self._metrics_manager = MetricsManager()
        self._lock = asyncio.Lock()
        self.health_checker = HealthChecker()
        
        # Initialize optional components based on settings
        optional_components = ComponentInitializer.initialize_all_optional_components()
        self.self_reflection_engine = optional_components["self_reflection_engine"]
        self.experience_learning = optional_components["experience_learning"]
        self.world_model = optional_components["world_model"]
        
        # Observer manager for task events (consolidates observer pattern)
        self._observer_manager = AgentObserverManager(
            experience_learning=self.experience_learning,
            world_model=self.world_model,
            knowledge_base=self.knowledge_base
        )
        
        # Task processor (extracted responsibility)
        self._task_processor = TaskProcessor(
            reasoning_engine=self.reasoning_engine,
            learning_engine=self.learning_engine,
            knowledge_base=self.knowledge_base,
            metrics_manager=self._metrics_manager,
            observer_manager=self._observer_manager
        )
        
        # Autonomous operation handler (extracted responsibility)
        self._autonomous_handler = AutonomousOperationHandler(
            learning_engine=self.learning_engine,
            world_model=self.world_model,
            agent_id=self.agent_id,
            instruction=self.instruction
        )
        
        # Periodic tasks coordinator (extracted responsibility)
        self._periodic_coordinator = PeriodicTasksCoordinator(
            health_checker=self.health_checker,
            metrics_manager=self._metrics_manager,
            self_reflection_engine=self.self_reflection_engine,
            task_queue=self.task_queue,
            agent_id=self.agent_id
        )
        
        # Loop coordinator (extracted responsibility)
        self._loop_coordinator = LoopCoordinator(
            task_queue=self.task_queue,
            task_processor=self._task_processor,
            autonomous_handler=self._autonomous_handler,
            periodic_coordinator=self._periodic_coordinator,
            agent_id=self.agent_id,
            status_getter=lambda: self.status,
            stop_event=self._stop_event
        )
    
    async def start(self) -> None:
        """Start the autonomous agent"""
        if self.status == AgentStatus.RUNNING:
            logger.warning(f"Agent {self.agent_id} already running")
            return
        
        logger.info(f"🚀 Starting autonomous agent {self.agent_id}")
        logger.info("⚠️  Agent will run continuously until explicitly stopped")
        
        self.status = AgentStatus.RUNNING
        self._stop_event.clear()
        self._metrics_manager.start_tracking()
        self._running_task = asyncio.create_task(self._run_loop())
    
    async def stop(self) -> None:
        """Stop the autonomous agent"""
        logger.info(f"⏹️  Stopping autonomous agent {self.agent_id}")
        self.status = AgentStatus.STOPPED
        self._stop_event.set()
        
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            try:
                await self._running_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"✅ Agent {self.agent_id} stopped")
    
    async def pause(self) -> None:
        """Pause the agent"""
        logger.info(f"⏸️  Pausing agent {self.agent_id}")
        self.status = AgentStatus.PAUSED
    
    async def resume(self) -> None:
        """Resume the agent"""
        logger.info(f"▶️  Resuming agent {self.agent_id}")
        self.status = AgentStatus.RUNNING
    
    async def add_task(
        self,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a task to the agent's queue"""
        task_id = await self.task_queue.add_task(instruction, metadata)
        return task_id
    
    async def _run_loop(self) -> None:
        """Main execution loop - runs continuously until stopped"""
        logger.info(f"🔄 Agent {self.agent_id} main loop started")
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Execute one loop iteration (delegated to coordinator)
                    await self._loop_coordinator.run_loop_iteration(
                        self,
                        self.openrouter_client
                    )
                except Exception as e:
                    logger.error(f"Error in agent loop iteration: {e}", exc_info=True)
                    await self._handle_loop_error(e)
                    await asyncio.sleep(settings.agent_poll_interval)
                    
        except asyncio.CancelledError:
            logger.info(f"Agent {self.agent_id} loop cancelled")
        finally:
            self.status = AgentStatus.STOPPED
            logger.info(f"Agent {self.agent_id} loop ended")
    
    
    
    async def get_health(self) -> Dict[str, Any]:
        """Get current health status"""
        return await self.health_checker.check_agent_health(
            self,
            self.openrouter_client
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics"""
        queue_size = await self.task_queue.get_queue_size()
        learning_stats = await self.learning_engine.get_learning_stats()
        knowledge_stats = await self.knowledge_base.get_stats()
        metrics = self._metrics_manager.get_metrics_dict()
        
        status_dict = {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "instruction": self.instruction,
            "metrics": {
                **metrics,
                "success_rate": self._metrics_manager.get_success_rate(),
                "avg_tokens_per_task": self._metrics_manager.get_average_tokens_per_task()
            },
            "queue_size": queue_size,
            "learning_stats": learning_stats,
            "knowledge_stats": knowledge_stats,
            "resilience_stats": self.openrouter_client.get_resilience_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Collect optional component stats using centralized collector
        await self._collect_optional_component_stats(status_dict)
        
        return status_dict
    
    async def _handle_loop_error(self, error: Exception) -> None:
        """Handle errors in the main execution loop"""
        try:
            await self.learning_engine.record_event(
                "error",
                {"error": str(error)},
                "failure"
            )
            self._metrics_manager.record_error()
        except Exception as e:
            logger.warning(f"Error recording loop error: {e}")
    
    async def _collect_optional_component_stats(self, status_dict: Dict[str, Any]) -> None:
        """
        Collect stats from all optional components.
        Centralizes the repetitive pattern of checking and collecting stats.
        
        Args:
            status_dict: Dictionary to update with component stats
        """
        # Define all optional components to collect stats from
        components = [
            ("self_reflection_stats", self.self_reflection_engine, 
             lambda: self.self_reflection_engine.get_reflection_stats() if self.self_reflection_engine else None),
            ("experience_learning_stats", self.experience_learning,
             lambda: self.experience_learning.get_lifecycle_learning_stats() if self.experience_learning else None),
            ("world_model_stats", self.world_model,
             lambda: self.world_model.get_world_summary() if self.world_model else None),
        ]
        
        # Collect all stats in one call
        collected_stats = await StatusCollector.collect_multiple_status([
            (key, component, get_method) 
            for key, component, get_method in components 
            if component is not None
        ])
        
        # Update status_dict with collected stats
        status_dict.update(collected_stats)

