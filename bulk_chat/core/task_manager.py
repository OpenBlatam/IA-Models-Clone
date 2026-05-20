"""
Task Manager - Gestor de Tareas
================================

Sistema de gestión de tareas con prioridades, dependencias y seguimiento de progreso.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estado de tarea."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Prioridad de tarea."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Task:
    """Tarea."""
    task_id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assignee: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0 a 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskList:
    """Lista de tareas."""
    list_id: str
    name: str
    description: str = ""
    tasks: List[str] = field(default_factory=list)  # task_ids
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskManager:
    """Gestor de tareas."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_lists: Dict[str, TaskList] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def create_task(
        self,
        task_id: str,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        assignee: Optional[str] = None,
        due_date: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear tarea."""
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            assignee=assignee,
            due_date=due_date,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        self.tasks[task_id] = task
        
        # Actualizar grafo de dependencias
        for dep_id in task.dependencies:
            self.dependency_graph[dep_id].append(task_id)
        
        logger.info(f"Created task: {task_id} - {title}")
        return task_id
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
    ) -> bool:
        """Actualizar estado de tarea."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        old_status = task.status
        task.status = status
        
        if status == TaskStatus.IN_PROGRESS and not task.started_at:
            task.started_at = datetime.now()
        
        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now()
            task.progress = 100.0
        
        if status == TaskStatus.FAILED:
            task.progress = 0.0
        
        logger.info(f"Task {task_id} status: {old_status.value} -> {status.value}")
        return True
    
    def update_task_progress(self, task_id: str, progress: float) -> bool:
        """Actualizar progreso de tarea."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.progress = max(0.0, min(100.0, progress))
        
        if task.progress >= 100.0 and task.status != TaskStatus.COMPLETED:
            self.update_task_status(task_id, TaskStatus.COMPLETED)
        
        return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener tarea."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        # Verificar dependencias
        blocked_by = []
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                blocked_by.append(dep_id)
        
        if blocked_by and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.BLOCKED
        
        return {
            "task_id": task.task_id,
            "title": task.title,
            "description": task.description,
            "status": task.status.value,
            "priority": task.priority.value,
            "assignee": task.assignee,
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "dependencies": task.dependencies,
            "blocked_by": blocked_by,
            "metadata": task.metadata,
        }
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Dict[str, Any]]:
        """Obtener tareas por estado."""
        tasks = [self.get_task(tid) for tid, t in self.tasks.items() if t.status == status]
        return [t for t in tasks if t is not None]
    
    def get_tasks_by_assignee(self, assignee: str) -> List[Dict[str, Any]]:
        """Obtener tareas por asignado."""
        tasks = [self.get_task(tid) for tid, t in self.tasks.items() if t.assignee == assignee]
        return [t for t in tasks if t is not None]
    
    def get_overdue_tasks(self) -> List[Dict[str, Any]]:
        """Obtener tareas vencidas."""
        now = datetime.now()
        overdue = []
        
        for task in self.tasks.values():
            if task.due_date and task.due_date < now and task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
                task_dict = self.get_task(task.task_id)
                if task_dict:
                    overdue.append(task_dict)
        
        return overdue
    
    def create_task_list(
        self,
        list_id: str,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear lista de tareas."""
        task_list = TaskList(
            list_id=list_id,
            name=name,
            description=description,
            metadata=metadata or {},
        )
        
        self.task_lists[list_id] = task_list
        logger.info(f"Created task list: {list_id} - {name}")
        return list_id
    
    def add_task_to_list(self, list_id: str, task_id: str) -> bool:
        """Agregar tarea a lista."""
        task_list = self.task_lists.get(list_id)
        if not task_list:
            return False
        
        if task_id not in task_list.tasks:
            task_list.tasks.append(task_id)
        
        return True
    
    def get_task_list(self, list_id: str) -> Optional[Dict[str, Any]]:
        """Obtener lista de tareas."""
        task_list = self.task_lists.get(list_id)
        if not task_list:
            return None
        
        tasks = [self.get_task(tid) for tid in task_list.tasks]
        
        return {
            "list_id": task_list.list_id,
            "name": task_list.name,
            "description": task_list.description,
            "tasks": tasks,
            "task_count": len(tasks),
            "created_at": task_list.created_at.isoformat(),
            "metadata": task_list.metadata,
        }
    
    def get_task_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor de tareas."""
        by_status: Dict[str, int] = defaultdict(int)
        by_priority: Dict[str, int] = defaultdict(int)
        
        for task in self.tasks.values():
            by_status[task.status.value] += 1
            by_priority[task.priority.value] += 1
        
        return {
            "total_tasks": len(self.tasks),
            "tasks_by_status": dict(by_status),
            "tasks_by_priority": dict(by_priority),
            "total_task_lists": len(self.task_lists),
            "overdue_tasks": len(self.get_overdue_tasks()),
        }
















