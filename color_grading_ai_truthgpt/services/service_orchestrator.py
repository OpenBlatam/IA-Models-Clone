"""
Service Orchestrator for Color Grading AI
==========================================

Orchestrates multiple services to execute complex workflows.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Orchestration strategies."""
    SEQUENTIAL = "sequential"  # Execute in order
    PARALLEL = "parallel"  # Execute in parallel
    CONDITIONAL = "conditional"  # Execute based on conditions
    PIPELINE = "pipeline"  # Pipeline execution


@dataclass
class ServiceTask:
    """Service task definition."""
    service_name: str
    method_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None
    timeout: Optional[float] = None
    retry_count: int = 0


@dataclass
class OrchestrationResult:
    """Orchestration result."""
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ServiceOrchestrator:
    """
    Service orchestrator.
    
    Features:
    - Multi-service orchestration
    - Dependency management
    - Parallel and sequential execution
    - Conditional execution
    - Error handling
    - Result aggregation
    """
    
    def __init__(self, services: Dict[str, Any]):
        """
        Initialize service orchestrator.
        
        Args:
            services: Dictionary of available services
        """
        self.services = services
        self._execution_history: List[OrchestrationResult] = []
    
    async def orchestrate(
        self,
        tasks: List[ServiceTask],
        strategy: OrchestrationStrategy = OrchestrationStrategy.SEQUENTIAL
    ) -> OrchestrationResult:
        """
        Orchestrate service tasks.
        
        Args:
            tasks: List of service tasks
            strategy: Orchestration strategy
            
        Returns:
            Orchestration result
        """
        start_time = datetime.now()
        results = {}
        errors = {}
        
        try:
            if strategy == OrchestrationStrategy.SEQUENTIAL:
                results, errors = await self._execute_sequential(tasks)
            elif strategy == OrchestrationStrategy.PARALLEL:
                results, errors = await self._execute_parallel(tasks)
            elif strategy == OrchestrationStrategy.CONDITIONAL:
                results, errors = await self._execute_conditional(tasks)
            elif strategy == OrchestrationStrategy.PIPELINE:
                results, errors = await self._execute_pipeline(tasks)
            
            success = len(errors) == 0
            
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            success = False
            errors["orchestration"] = str(e)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = OrchestrationResult(
            success=success,
            results=results,
            errors=errors,
            execution_time=execution_time
        )
        
        self._execution_history.append(result)
        return result
    
    async def _execute_sequential(self, tasks: List[ServiceTask]) -> tuple:
        """Execute tasks sequentially."""
        results = {}
        errors = {}
        
        for task in tasks:
            # Check dependencies
            if not all(dep in results for dep in task.dependencies):
                errors[task.service_name] = f"Dependencies not met: {task.dependencies}"
                continue
            
            # Check condition
            if task.condition and not task.condition(results):
                continue
            
            # Execute task
            try:
                result = await self._execute_task(task, results)
                results[task.service_name] = result
            except Exception as e:
                errors[task.service_name] = str(e)
                if task.retry_count > 0:
                    # Retry logic
                    for _ in range(task.retry_count):
                        try:
                            result = await self._execute_task(task, results)
                            results[task.service_name] = result
                            break
                        except Exception as retry_e:
                            errors[task.service_name] = str(retry_e)
        
        return results, errors
    
    async def _execute_parallel(self, tasks: List[ServiceTask]) -> tuple:
        """Execute tasks in parallel."""
        # Group tasks by dependencies
        ready_tasks = [t for t in tasks if not t.dependencies]
        pending_tasks = [t for t in tasks if t.dependencies]
        
        results = {}
        errors = {}
        
        # Execute ready tasks in parallel
        if ready_tasks:
            tasks_to_run = [
                self._execute_task(task, results)
                for task in ready_tasks
            ]
            task_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    errors[task.service_name] = str(result)
                else:
                    results[task.service_name] = result
        
        # Execute pending tasks sequentially (they depend on results)
        for task in pending_tasks:
            if all(dep in results for dep in task.dependencies):
                try:
                    result = await self._execute_task(task, results)
                    results[task.service_name] = result
                except Exception as e:
                    errors[task.service_name] = str(e)
        
        return results, errors
    
    async def _execute_conditional(self, tasks: List[ServiceTask]) -> tuple:
        """Execute tasks conditionally."""
        results = {}
        errors = {}
        
        for task in tasks:
            # Check condition
            if task.condition and not task.condition(results):
                continue
            
            # Check dependencies
            if not all(dep in results for dep in task.dependencies):
                continue
            
            try:
                result = await self._execute_task(task, results)
                results[task.service_name] = result
            except Exception as e:
                errors[task.service_name] = str(e)
        
        return results, errors
    
    async def _execute_pipeline(self, tasks: List[ServiceTask]) -> tuple:
        """Execute tasks as pipeline."""
        results = {}
        errors = {}
        
        for task in tasks:
            # Pipeline: each task receives previous results
            try:
                result = await self._execute_task(task, results)
                results[task.service_name] = result
            except Exception as e:
                errors[task.service_name] = str(e)
                break  # Pipeline stops on error
        
        return results, errors
    
    async def _execute_task(
        self,
        task: ServiceTask,
        previous_results: Dict[str, Any]
    ) -> Any:
        """Execute a single task."""
        if task.service_name not in self.services:
            raise ValueError(f"Service not found: {task.service_name}")
        
        service = self.services[task.service_name]
        
        if not hasattr(service, task.method_name):
            raise ValueError(
                f"Method '{task.method_name}' not found in service '{task.service_name}'"
            )
        
        method = getattr(service, task.method_name)
        
        # Merge parameters with previous results
        params = task.parameters.copy()
        params.update({k: v for k, v in previous_results.items() if k in task.dependencies})
        
        # Execute with timeout if specified
        if task.timeout:
            return await asyncio.wait_for(
                method(**params) if asyncio.iscoroutinefunction(method) else method(**params),
                timeout=task.timeout
            )
        else:
            if asyncio.iscoroutinefunction(method):
                return await method(**params)
            else:
                return method(**params)
    
    def get_execution_history(self, limit: int = 100) -> List[OrchestrationResult]:
        """Get execution history."""
        return self._execution_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
            }
        
        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_execution_time": sum(r.execution_time for r in self._execution_history) / total,
        }


