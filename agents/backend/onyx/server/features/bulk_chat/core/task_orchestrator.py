"""
Task Orchestrator - Orquestador de Tareas
==========================================

Sistema avanzado de orquestación de tareas con dependencias, scheduling y ejecución paralela.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estado de tarea."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Prioridad de tarea."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Task:
    """Tarea."""
    task_id: str
    name: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)  # task_ids
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    max_retries: int = 0
    retry_count: int = 0
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGroup:
    """Grupo de tareas."""
    group_id: str
    name: str
    tasks: List[str] = field(default_factory=list)  # task_ids
    parallel: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING


class TaskOrchestrator:
    """Orquestador de tareas."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.task_groups: Dict[str, TaskGroup] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: deque = deque()
        self._lock = asyncio.Lock()
        self._workers: List[asyncio.Task] = []
    
    def create_task(
        self,
        task_id: str,
        name: str,
        handler: Callable,
        dependencies: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear tarea."""
        task = Task(
            task_id=task_id,
            name=name,
            handler=handler,
            dependencies=dependencies or [],
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata or {},
        )
        
        async def save_task():
            async with self._lock:
                self.tasks[task_id] = task
                self.task_queue.append(task_id)
                self.task_queue = deque(sorted(self.task_queue, key=lambda tid: self.tasks[tid].priority.value, reverse=True))
        
        asyncio.create_task(save_task())
        
        logger.info(f"Created task: {task_id} - {name}")
        return task_id
    
    async def execute_task(self, task_id: str) -> Any:
        """Ejecutar tarea."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        # Verificar dependencias
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task:
                raise ValueError(f"Dependency task not found: {dep_id}")
            
            if dep_task.status != TaskStatus.COMPLETED:
                raise ValueError(f"Dependency task {dep_id} not completed")
        
        # Actualizar estado
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # Ejecutar handler con timeout si está configurado
            if task.timeout:
                result = await asyncio.wait_for(
                    task.handler() if asyncio.iscoroutinefunction(task.handler) else asyncio.to_thread(task.handler),
                    timeout=task.timeout,
                )
            else:
                if asyncio.iscoroutinefunction(task.handler):
                    result = await task.handler()
                else:
                    result = await asyncio.to_thread(task.handler)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            return result
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timeout after {task.timeout} seconds"
            raise
            
        except Exception as e:
            task.error = str(e)
            
            # Reintentar si es posible
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                task.retry_count += 1
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                return await self.execute_task(task_id)
            else:
                task.status = TaskStatus.FAILED
                raise
    
    async def start_workers(self):
        """Iniciar workers."""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self._workers.append(worker)
    
    async def _worker(self, worker_id: str):
        """Worker para procesar tareas."""
        while True:
            try:
                async with self._lock:
                    if not self.task_queue:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Obtener siguiente tarea
                    task_id = None
                    for tid in self.task_queue:
                        task = self.tasks[tid]
                        if task.status == TaskStatus.PENDING:
                            # Verificar dependencias
                            deps_ready = all(
                                self.tasks[dep_id].status == TaskStatus.COMPLETED
                                for dep_id in task.dependencies
                                if dep_id in self.tasks
                            )
                            
                            if deps_ready:
                                task_id = tid
                                self.task_queue.remove(tid)
                                break
                    
                    if not task_id:
                        await asyncio.sleep(0.1)
                        continue
                
                # Ejecutar tarea fuera del lock
                try:
                    await self.execute_task(task_id)
                except Exception as e:
                    logger.error(f"Error executing task {task_id}: {e}")
                
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    def create_task_group(
        self,
        group_id: str,
        name: str,
        task_ids: List[str],
        parallel: bool = False,
    ) -> str:
        """Crear grupo de tareas."""
        group = TaskGroup(
            group_id=group_id,
            name=name,
            tasks=task_ids,
            parallel=parallel,
        )
        
        async def save_group():
            async with self._lock:
                self.task_groups[group_id] = group
        
        asyncio.create_task(save_group())
        
        return group_id
    
    async def execute_group(self, group_id: str):
        """Ejecutar grupo de tareas."""
        group = self.task_groups.get(group_id)
        if not group:
            raise ValueError(f"Task group not found: {group_id}")
        
        group.status = TaskStatus.RUNNING
        
        try:
            if group.parallel:
                # Ejecutar en paralelo
                await asyncio.gather(*[self.execute_task(tid) for tid in group.tasks])
            else:
                # Ejecutar secuencialmente
                for tid in group.tasks:
                    await self.execute_task(tid)
            
            group.status = TaskStatus.COMPLETED
        except Exception as e:
            group.status = TaskStatus.FAILED
            raise
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener tarea."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "priority": task.priority.value,
            "dependencies": task.dependencies,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error": task.error,
        }
    
    def get_task_orchestrator_summary(self) -> Dict[str, Any]:
        """Obtener resumen del orquestador."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for task in self.tasks.values():
            by_status[task.status.value] += 1
        
        return {
            "total_tasks": len(self.tasks),
            "tasks_by_status": dict(by_status),
            "running_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue),
            "total_groups": len(self.task_groups),
            "active_workers": len(self._workers),
        }



