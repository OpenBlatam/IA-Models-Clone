"""
Startup Manager for Piel Mejorador AI SAM3
===========================================

Manages application startup and initialization.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StartupPhase(Enum):
    """Startup phases."""
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    SERVICES = "services"
    HEALTH_CHECKS = "health_checks"
    READY = "ready"


@dataclass
class StartupTask:
    """Startup task definition."""
    name: str
    phase: StartupPhase
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    critical: bool = True


class StartupManager:
    """
    Manages application startup.
    
    Features:
    - Phased startup
    - Dependency management
    - Timeout handling
    - Progress tracking
    - Error recovery
    """
    
    def __init__(self):
        """Initialize startup manager."""
        self._tasks: Dict[str, StartupTask] = {}
        self._executed: set[str] = set()
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}
        self._start_time: Optional[datetime] = None
    
    def register_task(
        self,
        name: str,
        phase: StartupPhase,
        func: Callable,
        dependencies: Optional[List[str]] = None,
        timeout: float = 30.0,
        critical: bool = True
    ):
        """
        Register a startup task.
        
        Args:
            name: Task name
            phase: Startup phase
            func: Task function (can be async or sync)
            dependencies: Optional task dependencies
            timeout: Task timeout in seconds
            critical: Whether task failure should stop startup
        """
        self._tasks[name] = StartupTask(
            name=name,
            phase=phase,
            func=func,
            dependencies=dependencies or [],
            timeout=timeout,
            critical=critical
        )
        logger.debug(f"Registered startup task: {name} (phase: {phase.value})")
    
    async def execute_phase(self, phase: StartupPhase) -> Dict[str, Any]:
        """
        Execute all tasks in a phase.
        
        Args:
            phase: Phase to execute
            
        Returns:
            Execution results
        """
        phase_tasks = [
            task for task in self._tasks.values()
            if task.phase == phase
        ]
        
        if not phase_tasks:
            return {"status": "skipped", "tasks": []}
        
        logger.info(f"Starting phase: {phase.value} ({len(phase_tasks)} tasks)")
        
        results = {}
        errors = {}
        
        # Execute tasks in dependency order
        remaining = {task.name: task for task in phase_tasks}
        
        while remaining:
            # Find tasks with all dependencies satisfied
            ready_tasks = [
                task for task in remaining.values()
                if all(dep in self._executed for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                unresolved = [name for name in remaining.keys()]
                raise RuntimeError(f"Cannot resolve dependencies for: {unresolved}")
            
            # Execute ready tasks in parallel
            task_results = await asyncio.gather(
                *[self._execute_task(task) for task in ready_tasks],
                return_exceptions=True
            )
            
            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    errors[task.name] = result
                    if task.critical:
                        raise RuntimeError(f"Critical task {task.name} failed: {result}") from result
                    logger.error(f"Task {task.name} failed (non-critical): {result}")
                else:
                    results[task.name] = result
                    self._executed.add(task.name)
                
                remaining.pop(task.name)
        
        return {
            "status": "completed",
            "tasks": list(results.keys()),
            "errors": list(errors.keys()),
        }
    
    async def _execute_task(self, task: StartupTask) -> Any:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        logger.debug(f"Executing task: {task.name}")
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await asyncio.wait_for(task.func(), timeout=task.timeout)
            else:
                result = task.func()
            
            self._results[task.name] = result
            logger.debug(f"Task {task.name} completed")
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task.name} timed out after {task.timeout}s")
        except Exception as e:
            self._errors[task.name] = e
            raise
    
    async def startup(self) -> Dict[str, Any]:
        """
        Execute full startup sequence.
        
        Returns:
            Startup results
        """
        self._start_time = datetime.now()
        logger.info("Starting application startup sequence")
        
        results = {}
        
        for phase in StartupPhase:
            try:
                phase_result = await self.execute_phase(phase)
                results[phase.value] = phase_result
            except Exception as e:
                logger.error(f"Startup failed at phase {phase.value}: {e}")
                raise
        
        duration = (datetime.now() - self._start_time).total_seconds()
        logger.info(f"Startup completed in {duration:.2f}s")
        
        return {
            "status": "success",
            "duration_seconds": duration,
            "phases": results,
            "executed_tasks": list(self._executed),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get startup status."""
        return {
            "started": self._start_time is not None,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "executed_tasks": list(self._executed),
            "total_tasks": len(self._tasks),
            "errors": {name: str(e) for name, e in self._errors.items()},
        }




