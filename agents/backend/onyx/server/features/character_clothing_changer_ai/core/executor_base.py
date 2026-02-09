"""
Executor Base
=============

Base executor pattern for all executor types.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ExecutionStatus(Enum):
    """Execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Execution result."""
    status: ExecutionStatus
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseExecutor(ABC):
    """Base executor interface."""
    
    def __init__(self, name: str = "Executor"):
        """
        Initialize executor.
        
        Args:
            name: Executor name
        """
        self.name = name
        self.running: Dict[str, asyncio.Task] = {}
        self.completed: List[ExecutionResult] = []
        self.max_completed = 1000
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> ExecutionResult:
        """
        Execute operation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Execution result
        """
        pass
    
    def _save_result(self, result: ExecutionResult):
        """Save execution result."""
        self.completed.append(result)
        
        # Limit completed results
        if len(self.completed) > self.max_completed:
            self.completed = self.completed[-self.max_completed:]
    
    def get_recent_results(self, limit: int = 100) -> List[ExecutionResult]:
        """Get recent execution results."""
        return self.completed[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        total = len(self.completed)
        completed = len([r for r in self.completed if r.status == ExecutionStatus.COMPLETED])
        failed = len([r for r in self.completed if r.status == ExecutionStatus.FAILED])
        
        return {
            "name": self.name,
            "running": len(self.running),
            "total_completed": total,
            "successful": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0.0
        }


class AsyncExecutor(BaseExecutor):
    """Async executor implementation."""
    
    def __init__(self, name: str = "AsyncExecutor", max_concurrent: int = 10):
        """
        Initialize async executor.
        
        Args:
            name: Executor name
            max_concurrent: Maximum concurrent executions
        """
        super().__init__(name)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute(
        self,
        func: Callable[..., Awaitable[R]],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute async function.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            timeout: Optional timeout
            **kwargs: Keyword arguments
            
        Returns:
            Execution result
        """
        execution_id = f"{self.name}_{datetime.now().timestamp()}"
        start = datetime.now()
        
        async def run():
            async with self.semaphore:
                try:
                    if timeout:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        result = await func(*args, **kwargs)
                    
                    duration = (datetime.now() - start).total_seconds()
                    exec_result = ExecutionResult(
                        status=ExecutionStatus.COMPLETED,
                        result=result,
                        duration=duration
                    )
                    self._save_result(exec_result)
                    return exec_result
                except asyncio.TimeoutError:
                    duration = (datetime.now() - start).total_seconds()
                    exec_result = ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=TimeoutError(f"Execution timed out after {timeout}s"),
                        duration=duration
                    )
                    self._save_result(exec_result)
                    return exec_result
                except Exception as e:
                    duration = (datetime.now() - start).total_seconds()
                    exec_result = ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=e,
                        duration=duration
                    )
                    self._save_result(exec_result)
                    return exec_result
        
        task = asyncio.create_task(run())
        self.running[execution_id] = task
        
        try:
            result = await task
            return result
        finally:
            self.running.pop(execution_id, None)
    
    async def execute_batch(
        self,
        func: Callable[..., Awaitable[R]],
        items: List[Any],
        *args,
        **kwargs
    ) -> List[ExecutionResult]:
        """
        Execute function on batch of items.
        
        Args:
            func: Async function to execute
            items: List of items
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            List of execution results
        """
        tasks = [
            self.execute(func, item, *args, **kwargs)
            for item in items
        ]
        return await asyncio.gather(*tasks)

