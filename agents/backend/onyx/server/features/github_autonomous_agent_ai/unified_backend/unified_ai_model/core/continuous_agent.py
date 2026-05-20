"""
Continuous Agent for Unified AI Model
Runs indefinitely until the user explicitly stops it
Based on patterns from implementations_autonomous_agents and bulk processing
"""

import asyncio
import logging
import signal
import uuid
import heapq
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ..config import get_config
from .llm_service import LLMService, get_llm_service
from .chat_service import ChatService, get_chat_service, MessageRole
from .knowledge_base import KnowledgeBase
from .learning_engine import LearningEngine
from .reasoning_engine import ReasoningEngine
from .self_reflection import SelfReflectionEngine
from .world_model import ContinualWorldModel
from .autonomous_operation_handler import AutonomousOperationHandler
from .experience_driven_learning import ExperienceDrivenLearning
from .workflow_engine import WorkflowEngine
from .analytics import AnalyticsSystem
from .component_manager import ComponentManager
from .batch_processor import BatchProcessor
from .priority_queue import PriorityTaskQueue, PriorityLevel
from .worker_pool import WorkerPool

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """A task for the continuous agent."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: int = 5  # 1-10, higher = more priority
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class AgentMetrics:
    """Metrics for the continuous agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    total_processing_time_ms: float = 0.0
    start_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    @property
    def uptime_seconds(self) -> float:
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def average_processing_time_ms(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return self.total_processing_time_ms / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_tokens_used": self.total_tokens_used,
            "average_processing_time_ms": self.average_processing_time_ms,
            "uptime_seconds": self.uptime_seconds,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


class ContinuousAgent:
    """
    Continuous Agent that runs indefinitely until explicitly stopped.
    
    Features:
    - Runs 24/7 until user stops it
    - Task queue with priority processing (heap-based)
    - Batch processing for high throughput
    - Worker pool for parallel execution
    - Automatic idle mode when no tasks
    - Pause/Resume functionality
    - Callbacks for task completion and errors
    - Graceful shutdown on signals
    - Performance metrics and monitoring
    """
    
    def __init__(
        self,
        name: str = "ContinuousAgent",
        llm_service: Optional[LLMService] = None,
        chat_service: Optional[ChatService] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        loop_interval: float = 1.0,
        idle_interval: float = 5.0,
        max_concurrent_tasks: int = 3,
        batch_size: int = 8,
        num_workers: int = 3,
        enable_parallel: bool = True
    ):
        """
        Initialize continuous agent.
        
        Args:
            name: Agent name
            llm_service: LLM service instance
            chat_service: Chat service instance
            system_prompt: System prompt for the agent
            model: Model to use
            loop_interval: Seconds between loop iterations
            idle_interval: Seconds to wait when idle
            max_concurrent_tasks: Max concurrent task processing
            batch_size: Size of batches for parallel processing
            num_workers: Number of worker threads for parallel execution
            enable_parallel: Enable parallel processing mode
        """
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.llm_service = llm_service or get_llm_service()
        self.chat_service = chat_service or get_chat_service()
        self.config = get_config()
        
        # Configuration
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.model = model or self.config.default_model
        self.loop_interval = loop_interval
        self.idle_interval = idle_interval
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.enable_parallel = enable_parallel
        
        # State
        self.status = AgentStatus.IDLE
        self.is_running = False
        self.should_stop = False
        self.is_paused = False
        
        # Task management with priority queue
        self.priority_queue = PriorityTaskQueue(max_size=10000)
        self.task_registry: Dict[str, AgentTask] = {}  # All tasks by ID
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: deque = deque(maxlen=100)
        self.task_lock = asyncio.Lock()
        
        # Parallel processing components
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            max_concurrent=max_concurrent_tasks
        )
        self.worker_pool: Optional[WorkerPool] = None
        if enable_parallel:
            self.worker_pool = WorkerPool(
                num_workers=num_workers,
                task_queue=self.priority_queue
            )
        
        # Metrics
        self.metrics = AgentMetrics()
        
        # Callbacks
        self.task_callbacks: List[Callable[[AgentTask], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        self.status_callbacks: List[Callable[[AgentStatus], None]] = []
        
        # Conversation for context
        self.conversation = self.chat_service.create_conversation(
            system_prompt=self.system_prompt,
            model=self.model
        )
        
        # Initialize Component Manager
        self.components = ComponentManager(
            config=self.config,
            llm_service=self.llm_service,
            agent_id=self.agent_id,
            system_prompt=self.system_prompt
        )
        
        # Shortcuts for compatibility (optional, but helpful for refactoring)
        self.knowledge_base = self.components.knowledge_base
        self.learning_engine = self.components.learning_engine
        self.reasoning_engine = self.components.reasoning_engine
        self.world_model = self.components.world_model
        self.self_reflection_engine = self.components.self_reflection_engine
        self.experience_learning = self.components.experience_learning
        self.workflow_engine = self.components.workflow_engine
        self.analytics = self.components.analytics
        self.data_processor = self.components.data_processor
        self.autonomous_handler = self.components.autonomous_handler
        
        # Signal handling
        self._setup_signal_handlers()
        
        logger.info(f"Continuous Agent '{name}' initialized (ID: {self.agent_id})")
        logger.info(f"   Parallel mode: {enable_parallel}, Workers: {num_workers}, Batch size: {batch_size}")
        logger.info(f"   Autonomous features: Learning={self.config.autonomous.learning_enabled}, WorldModel={self.config.autonomous.enable_world_model}")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent."""
        return """You are a continuous autonomous AI agent. You process tasks submitted by users 
and work on them until completion. You maintain context across tasks and can learn from 
previous interactions. Be thorough, helpful, and proactive in your responses."""
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            # Signal handling not available (e.g., not main thread)
            pass
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.should_stop = True
    
    def _set_status(self, status: AgentStatus) -> None:
        """Set agent status and notify callbacks."""
        if self.status != status:
            self.status = status
            for callback in self.status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")
    
    async def start(self) -> None:
        """
        Start the agent in continuous mode.
        
        The agent runs indefinitely until stop() is called or a signal is received.
        """
        if self.is_running:
            logger.warning(f"Agent '{self.name}' is already running")
            return
        
        self.is_running = True
        self.should_stop = False
        self.metrics.start_time = datetime.now()
        self._set_status(AgentStatus.RUNNING)
        
        logger.info(f"🚀 Starting Continuous Agent '{self.name}'")
        logger.info(f"   Agent ID: {self.agent_id}")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   Loop interval: {self.loop_interval}s")
        logger.info(f"   Parallel mode: {self.enable_parallel}")
        if self.enable_parallel:
            logger.info(f"   Workers: {self.num_workers}, Batch size: {self.batch_size}")
        logger.info(f"   Agent will run until explicitly stopped (Ctrl+C or stop())")
        
        try:
            # Start worker pool if parallel mode enabled
            if self.enable_parallel and self.worker_pool:
                await self.worker_pool.start(self._process_task_internal)
            
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Agent main loop cancelled")
        except Exception as e:
            logger.error(f"Error in agent main loop: {e}", exc_info=True)
            self._set_status(AgentStatus.ERROR)
        finally:
            await self._cleanup()
    
    async def _main_loop(self) -> None:
        """
        Main execution loop - runs until should_stop is True.
        """
        iteration = 0
        
        while not self.should_stop:
            iteration += 1
            
            try:
                # Check if paused
                if self.is_paused:
                    self._set_status(AgentStatus.PAUSED)
                    await asyncio.sleep(self.idle_interval)
                    continue
                
                # Check for tasks
                has_tasks = len(self.priority_queue) > 0 or len(self.active_tasks) > 0
                
                if has_tasks:
                    self._set_status(AgentStatus.PROCESSING)
                    await self._process_task_queue()
                    await asyncio.sleep(self.loop_interval)
                else:
                    # Idle mode - wait longer
                    self._set_status(AgentStatus.IDLE)
                    
                    # Run autonomous operations (self-learning, planning)
                    if self.autonomous_handler:
                        await self.autonomous_handler.execute()
                        
                    await asyncio.sleep(self.idle_interval)
                
                # Update last activity
                self.metrics.last_activity = datetime.now()
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in loop iteration {iteration}: {e}")
                for callback in self.error_callbacks:
                    try:
                        callback(e)
                    except Exception:
                        pass
                await asyncio.sleep(self.loop_interval)
        
        logger.info(f"Agent '{self.name}' main loop ended after {iteration} iterations")
    
    async def _process_task_queue(self) -> None:
        """Process pending tasks from the queue."""
        if self.enable_parallel and self.worker_pool:
            # Worker pool handles processing automatically
            # Just check if we need to do anything
            return
        
        # Non-parallel mode: use batch processor
        async with self.task_lock:
            # Check how many we can start
            available_slots = self.max_concurrent_tasks - len(self.active_tasks)
            
            if available_slots <= 0 or self.priority_queue.is_empty:
                return
            
            # Get batch of tasks from priority queue
            tasks_to_start = await self.priority_queue.pop_batch(
                min(available_slots, self.batch_size)
            )
        
        if not tasks_to_start:
            return
        
        # Process tasks using batch processor
        await self.batch_processor.process_batch(
            items=tasks_to_start,
            processor_fn=self._process_task_internal,
            on_complete=self._on_task_batch_complete,
            on_error=self._on_task_batch_error
        )
    
    async def _on_task_batch_complete(self, task: AgentTask, result: Any) -> None:
        """Callback for batch processor task completion."""
        for callback in self.task_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")
    
    async def _on_task_batch_error(self, task: AgentTask, error: Exception) -> None:
        """Callback for batch processor task errors."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    async def _process_task_internal(self, task: AgentTask) -> AgentTask:
        """
        Internal task processor used by workers and batch processor.
        
        Args:
            task: Task to process
            
        Returns:
            Processed task
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task
        
        logger.info(f"Processing task {task.task_id}: {task.description[:50]}...")
        
        try:
            start_time = datetime.now()
            
            # Retrieve similar experiences if enabled
            context_data = ""
            if self.experience_learning:
                similar_experiences = await self.experience_learning.retrieve_similar(task.description)
                if similar_experiences:
                    context_data = await self.experience_learning.format_experiences_for_prompt(similar_experiences)
                    logger.info(f"Retrieved {len(similar_experiences)} similar experiences for task {task.task_id}")
            
            # Use ReasoningEngine if available for enhanced processing
            # This adds knowledge retrieval and better context
            reasoning_result = await self.reasoning_engine.reason(
                instruction=task.description,
                metadata=task.metadata,
                context_data=context_data
            )
            
            # Also use chat service to maintain conversation history if needed
            # But for now, we trust the reasoning result
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            task.status = TaskStatus.COMPLETED
            task.result = reasoning_result.response
            self.metrics.tasks_completed += 1
            self.metrics.total_tokens_used += reasoning_result.tokens_used
            
            # Record success in learning engine
            await self.learning_engine.record_event(
                event_type="task_completion",
                context={"task_id": task.task_id, "priority": task.priority},
                outcome="success"
            )
            
            # Record experience for future learning
            if self.experience_learning:
                await self.experience_learning.add_experience(
                    task_id=task.task_id,
                    instruction=task.description,
                    result=task.result,
                    outcome="success",
                    metadata={"processing_time": processing_time}
                )
            
            # Add to knowledge base if significant
            if len(task.result) > 100:
                await self.knowledge_base.add_knowledge(
                    content=f"Task: {task.description}\nResult: {task.result}",
                    context={"task_id": task.task_id, "timestamp": datetime.now().isoformat()}
                )
            
            task.completed_at = datetime.now()
            self.metrics.total_processing_time_ms += processing_time
            
            # Log to analytics system
            self.analytics.log_task(
                task_id=task.task_id,
                description=task.description,
                priority=task.priority,
                status="completed",
                processing_time=processing_time,
                tokens_used=reasoning_result.tokens_used,
                result_length=len(task.result)
            )
            
            logger.info(f"Task {task.task_id} completed with status: {task.status.value}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            self.metrics.tasks_failed += 1
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Record failure in learning engine
            await self.learning_engine.record_event(
                event_type="task_failure",
                context={"task_id": task.task_id, "error": str(e)},
                outcome="failure"
            )
        
        finally:
            # Move to completed and remove from active
            self.completed_tasks.append(task)
            self.active_tasks.pop(task.task_id, None)
        
        return task
    
    async def _process_task(self, task: AgentTask) -> None:
        """Process a single task (wrapper for compatibility)."""
        result = await self._process_task_internal(task)
        
        # Execute callbacks
        if result.status == TaskStatus.COMPLETED:
            for callback in self.task_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
        else:
            for callback in self.error_callbacks:
                try:
                    callback(Exception(result.error or "Unknown error"))
                except Exception:
                    pass
    
    def submit_task(
        self,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a task for processing.
        
        Args:
            description: Task description
            priority: Priority 1-10 (higher = more urgent)
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        if not description.strip():
            raise ValueError("Task description cannot be empty")
        
        priority = max(1, min(10, priority))
        
        task = AgentTask(
            description=description.strip(),
            priority=priority,
            metadata=metadata or {}
        )
        
        # Register task and add to priority queue
        self.task_registry[task.task_id] = task
        
        # Use asyncio to add to priority queue
        asyncio.create_task(self._submit_to_queue(task, priority))
        
        logger.info(f"Task submitted: {task.task_id} (priority: {priority})")
        return task.task_id
    
    async def _submit_to_queue(self, task: AgentTask, priority: int) -> None:
        """Submit task to priority queue asynchronously."""
        success = await self.priority_queue.push(task, priority)
        if not success:
            logger.warning(f"Task queue full, task {task.task_id} not added")
            task.status = TaskStatus.FAILED
            task.error = "Task queue is full"
    
    async def submit_task_async(
        self,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a task asynchronously (preferred method).
        
        Args:
            description: Task description
            priority: Priority 1-10 (higher = more urgent)
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        if not description.strip():
            raise ValueError("Task description cannot be empty")
        
        priority = max(1, min(10, priority))
        
        task = AgentTask(
            description=description.strip(),
            priority=priority,
            metadata=metadata or {}
        )
        
        self.task_registry[task.task_id] = task
        
        success = await self.priority_queue.push(task, priority)
        if not success:
            task.status = TaskStatus.FAILED
            task.error = "Task queue is full"
            raise RuntimeError("Task queue is full")
        
        logger.info(f"Task submitted: {task.task_id} (priority: {priority})")
        return task.task_id
    
    async def submit_batch(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Submit multiple tasks at once for batch processing.
        
        Args:
            tasks: List of task dicts with 'description', 'priority', 'metadata'
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for task_data in tasks:
            try:
                task_id = await self.submit_task_async(
                    description=task_data.get("description", ""),
                    priority=task_data.get("priority", 5),
                    metadata=task_data.get("metadata")
                )
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Error submitting task: {e}")
                task_ids.append(None)
        
        logger.info(f"Batch submitted: {len([t for t in task_ids if t])} tasks")
        return task_ids
    
    def get_task(self, task_id: str) -> Optional[AgentTask]:
        """Get a task by ID."""
        # Check task registry first (contains all tasks)
        if task_id in self.task_registry:
            return self.task_registry[task_id]
        
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check completed
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task
        
        return None
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        task = self.get_task(task_id)
        return task.to_dict() if task else None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        for i, task in enumerate(self.task_queue):
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                self.task_queue.pop(i)
                self.completed_tasks.append(task)
                logger.info(f"Task {task_id} cancelled")
                return True
        return False
    
    def pause(self) -> None:
        """Pause the agent (stops processing new tasks)."""
        if not self.is_paused:
            self.is_paused = True
            self._set_status(AgentStatus.PAUSED)
            logger.info(f"Agent '{self.name}' paused")
    
    def resume(self) -> None:
        """Resume the agent."""
        if self.is_paused:
            self.is_paused = False
            self._set_status(AgentStatus.RUNNING)
            logger.info(f"Agent '{self.name}' resumed")
    
    def stop(self) -> None:
        """Stop the agent."""
        logger.info(f"Stopping agent '{self.name}'...")
        self.should_stop = True
        self.is_running = False
        self._set_status(AgentStatus.STOPPED)
    
    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        self._set_status(AgentStatus.STOPPED)
        self.is_running = False
        
        # Stop worker pool
        if self.worker_pool:
            await self.worker_pool.stop()
            worker_stats = self.worker_pool.get_stats()
            logger.info(f"   Worker pool stats: {worker_stats}")
        
        # Shutdown batch processor
        if self.batch_processor:
            self.batch_processor.shutdown()
        
        # Log final stats
        logger.info(f"Agent '{self.name}' shutdown complete")
        logger.info(f"   Tasks completed: {self.metrics.tasks_completed}")
        logger.info(f"   Tasks failed: {self.metrics.tasks_failed}")
        logger.info(f"   Uptime: {self.metrics.uptime_seconds:.2f}s")
        logger.info(f"   Total tokens: {self.metrics.total_tokens_used}")
        logger.info(f"   Avg processing time: {self.metrics.average_processing_time_ms:.2f}ms")
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete agent status."""
        status = {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "model": self.model,
            "queue_size": len(self.priority_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_tasks_registered": len(self.task_registry),
            "metrics": self.metrics.to_dict(),
            "conversation_id": self.conversation.conversation_id,
            "parallel_config": {
                "enabled": self.enable_parallel,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "max_concurrent": self.max_concurrent_tasks
            }
        }
        
        # Add worker pool stats if available
        if self.worker_pool:
            status["worker_pool"] = self.worker_pool.get_stats()
        
        return status
    
    def get_queue(self) -> List[Dict[str, Any]]:
        """Get current task queue (pending tasks)."""
        # Get pending tasks from registry
        pending = [
            t.to_dict() for t in self.task_registry.values()
            if t.status == TaskStatus.PENDING
        ]
        # Sort by priority (higher first)
        pending.sort(key=lambda t: t.get("priority", 5), reverse=True)
        return pending
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get active tasks."""
        return [t.to_dict() for t in self.active_tasks.values()]
    
    def get_completed_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently completed tasks."""
        tasks = list(self.completed_tasks)[-limit:]
        return [t.to_dict() for t in tasks]
    
    def set_task_callback(self, callback: Callable[[AgentTask], None]) -> None:
        """Add callback for task completion."""
        self.task_callbacks.append(callback)
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add callback for errors."""
        self.error_callbacks.append(callback)
    
    def set_status_callback(self, callback: Callable[[AgentStatus], None]) -> None:
        """Add callback for status changes."""
        self.status_callbacks.append(callback)


# Agent registry for managing multiple agents
_agents: Dict[str, ContinuousAgent] = {}


def create_agent(
    name: str = "ContinuousAgent",
    **kwargs
) -> ContinuousAgent:
    """Create and register a new continuous agent."""
    agent = ContinuousAgent(name=name, **kwargs)
    _agents[agent.agent_id] = agent
    return agent


def get_agent(agent_id: str) -> Optional[ContinuousAgent]:
    """Get an agent by ID."""
    return _agents.get(agent_id)


def list_agents() -> List[Dict[str, Any]]:
    """List all registered agents."""
    return [agent.get_status() for agent in _agents.values()]


def stop_agent(agent_id: str) -> bool:
    """Stop an agent by ID."""
    agent = _agents.get(agent_id)
    if agent:
        agent.stop()
        return True
    return False


def stop_all_agents() -> int:
    """Stop all agents."""
    count = 0
    for agent in _agents.values():
        agent.stop()
        count += 1
    return count



