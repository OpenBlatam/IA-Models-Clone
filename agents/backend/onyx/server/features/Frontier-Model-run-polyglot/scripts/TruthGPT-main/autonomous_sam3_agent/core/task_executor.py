"""
Task Execution Utilities
========================

Refactored with:
- ExecutionContext dataclass for state management
- ExecutionResult dataclass for typed results
- ExecutionStrategy pattern for different execution modes
- ExecutionHooks for extensibility
"""

import asyncio
import logging
import time
from typing import Callable, Any, Dict, Optional, List, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """
    Context for task execution.
    
    Encapsulates all state needed during execution.
    """
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    task_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_async(self) -> bool:
        """Check if function is async."""
        return asyncio.iscoroutinefunction(self.func)


@dataclass
class ExecutionResult:
    """
    Result from task execution.
    
    Provides typed access to execution outcomes.
    """
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    context: Optional[ExecutionContext] = None
    
    @classmethod
    def from_success(cls, value: Any, context: ExecutionContext) -> "ExecutionResult":
        """Create success result."""
        return cls(success=True, value=value, context=context)
    
    @classmethod
    def from_error(cls, error: Exception, context: ExecutionContext) -> "ExecutionResult":
        """Create error result."""
        return cls(success=False, error=error, context=context)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "value": self.value,
            "error": str(self.error) if self.error else None,
            "duration": self.context.duration if self.context else None,
        }


class ExecutionHook(Protocol):
    """Protocol for execution hooks."""
    
    async def on_start(self, context: ExecutionContext) -> None:
        """Called before execution starts."""
        ...
    
    async def on_complete(self, result: ExecutionResult) -> None:
        """Called after execution completes."""
        ...
    
    async def on_error(self, context: ExecutionContext, error: Exception) -> None:
        """Called when execution fails."""
        ...


class LoggingExecutionHook:
    """Hook that logs execution events."""
    
    async def on_start(self, context: ExecutionContext) -> None:
        logger.debug(f"Starting execution: {context.func.__name__}")
    
    async def on_complete(self, result: ExecutionResult) -> None:
        logger.debug(f"Completed execution: success={result.success}")
    
    async def on_error(self, context: ExecutionContext, error: Exception) -> None:
        logger.error(f"Execution failed: {error}")


class ExecutionHookRegistry:
    """Registry for execution hooks."""
    
    def __init__(self):
        self._hooks: List[ExecutionHook] = []
    
    def register(self, hook: ExecutionHook):
        """Register a hook."""
        self._hooks.append(hook)
    
    async def notify_start(self, context: ExecutionContext):
        """Notify all hooks of start."""
        for hook in self._hooks:
            try:
                await hook.on_start(context)
            except Exception as e:
                logger.warning(f"Hook error on start: {e}")
    
    async def notify_complete(self, result: ExecutionResult):
        """Notify all hooks of completion."""
        for hook in self._hooks:
            try:
                await hook.on_complete(result)
            except Exception as e:
                logger.warning(f"Hook error on complete: {e}")
    
    async def notify_error(self, context: ExecutionContext, error: Exception):
        """Notify all hooks of error."""
        for hook in self._hooks:
            try:
                await hook.on_error(context, error)
            except Exception as e:
                logger.warning(f"Hook error on error: {e}")


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""
    
    @abstractmethod
    async def execute(self, context: ExecutionContext) -> Any:
        """Execute the task."""
        pass


class DefaultExecutionStrategy(ExecutionStrategy):
    """Default execution strategy - handles both sync and async."""
    
    async def execute(self, context: ExecutionContext) -> Any:
        """Execute the task."""
        if context.is_async:
            return await context.func(*context.args, **context.kwargs)
        else:
            return context.func(*context.args, **context.kwargs)


class RetryExecutionStrategy(ExecutionStrategy):
    """Execution strategy with retries."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
        self._default = DefaultExecutionStrategy()
    
    async def execute(self, context: ExecutionContext) -> Any:
        """Execute with retries."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._default.execute(context)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.delay * (attempt + 1))
        
        raise last_error


class TaskExecutor:
    """
    Executes tasks with consistent error handling and future management.
    """
    
    _strategy: ExecutionStrategy = DefaultExecutionStrategy()
    _hooks = ExecutionHookRegistry()
    
    @classmethod
    def set_strategy(cls, strategy: ExecutionStrategy):
        """Set the execution strategy."""
        cls._strategy = strategy
    
    @classmethod
    def add_hook(cls, hook: ExecutionHook):
        """Add an execution hook."""
        cls._hooks.register(hook)
    
    @staticmethod
    async def execute_task(
        func: Callable,
        args: tuple,
        kwargs: dict,
        task_queue: asyncio.Queue,
        stats: Dict[str, Any],
        lock: asyncio.Lock,
        future: Optional[asyncio.Future] = None
    ) -> Any:
        """
        Execute a task with consistent error handling.
        """
        # Create execution context
        context = ExecutionContext(
            func=func,
            args=args,
            kwargs=kwargs,
            start_time=time.time(),
            status=ExecutionStatus.RUNNING,
        )
        
        # Notify hooks of start
        await TaskExecutor._hooks.notify_start(context)
        
        try:
            # Execute using strategy
            result = await TaskExecutor._strategy.execute(context)
            
            # Update context
            context.end_time = time.time()
            context.status = ExecutionStatus.COMPLETED
            
            # Mark task as done
            task_queue.task_done()
            
            # Resolve future if present
            if future and not future.done():
                future.set_result(result)
            
            # Update stats
            async with lock:
                stats["completed_tasks"] += 1
            
            # Create execution result and notify hooks
            exec_result = ExecutionResult.from_success(result, context)
            await TaskExecutor._hooks.notify_complete(exec_result)
            
            return result
            
        except Exception as e:
            # Update context
            context.end_time = time.time()
            context.status = ExecutionStatus.FAILED
            
            # Mark task as done even on error
            task_queue.task_done()
            
            # Reject future if present
            if future and not future.done():
                future.set_exception(e)
            
            # Update stats
            async with lock:
                stats["failed_tasks"] += 1
            
            # Notify hooks of error
            await TaskExecutor._hooks.notify_error(context, e)
            
            raise
