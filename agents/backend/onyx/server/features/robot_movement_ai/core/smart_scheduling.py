"""
Smart Scheduling System
=======================

Sistema de programación inteligente para optimizar asignación de recursos y tareas.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SchedulePriority(Enum):
    """Prioridad de programación."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ScheduleStatus(Enum):
    """Estado de programación."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Tarea programada."""
    task_id: str
    name: str
    task_type: str
    priority: SchedulePriority
    estimated_duration: float  # segundos
    required_resources: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    scheduled_start: Optional[str] = None
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None
    status: ScheduleStatus = ScheduleStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    """Recurso."""
    resource_id: str
    name: str
    resource_type: str
    capacity: float
    available_capacity: float
    cost_per_unit: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Schedule:
    """Programación."""
    schedule_id: str
    tasks: List[str]  # Lista de task_ids
    start_time: str
    end_time: str
    total_cost: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartScheduler:
    """
    Programador inteligente.
    
    Programa tareas optimizando recursos y tiempo.
    """
    
    def __init__(self):
        """Inicializar programador."""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.resources: Dict[str, Resource] = {}
        self.schedules: List[Schedule] = []
        self.max_schedules = 1000
    
    def add_task(
        self,
        name: str,
        task_type: str,
        priority: SchedulePriority,
        estimated_duration: float,
        required_resources: Dict[str, float],
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Agregar tarea.
        
        Args:
            name: Nombre de la tarea
            task_type: Tipo de tarea
            priority: Prioridad
            estimated_duration: Duración estimada en segundos
            required_resources: Recursos requeridos
            dependencies: Lista de IDs de tareas dependientes
            metadata: Metadata adicional
            
        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            task_type=task_type,
            priority=priority,
            estimated_duration=estimated_duration,
            required_resources=required_resources,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        logger.debug(f"Added task: {name} ({task_id})")
        
        return task_id
    
    def add_resource(
        self,
        name: str,
        resource_type: str,
        capacity: float,
        cost_per_unit: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Agregar recurso.
        
        Args:
            name: Nombre del recurso
            resource_type: Tipo de recurso
            capacity: Capacidad total
            cost_per_unit: Costo por unidad
            metadata: Metadata adicional
            
        Returns:
            ID del recurso
        """
        resource_id = str(uuid.uuid4())
        
        resource = Resource(
            resource_id=resource_id,
            name=name,
            resource_type=resource_type,
            capacity=capacity,
            available_capacity=capacity,
            cost_per_unit=cost_per_unit,
            metadata=metadata or {}
        )
        
        self.resources[resource_id] = resource
        logger.debug(f"Added resource: {name} ({resource_id})")
        
        return resource_id
    
    def create_schedule(
        self,
        start_time: Optional[str] = None,
        optimization_goal: str = "minimize_time"
    ) -> Schedule:
        """
        Crear programación.
        
        Args:
            start_time: Tiempo de inicio (opcional)
            optimization_goal: Objetivo de optimización (minimize_time, minimize_cost, balance)
            
        Returns:
            Programación creada
        """
        if not start_time:
            start_time = datetime.now().isoformat()
        
        # Ordenar tareas por prioridad y dependencias
        sorted_tasks = self._topological_sort()
        
        # Asignar recursos y tiempos
        schedule_tasks = []
        current_time = datetime.fromisoformat(start_time)
        resource_usage = {r_id: [] for r_id in self.resources}  # (start, end, task_id)
        
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            
            # Verificar dependencias
            if task.dependencies:
                max_dependency_end = current_time
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if dep_task and dep_task.actual_end:
                        dep_end = datetime.fromisoformat(dep_task.actual_end)
                        if dep_end > max_dependency_end:
                            max_dependency_end = dep_end
                current_time = max_dependency_end
            
            # Encontrar recursos disponibles
            task_start = self._find_resource_slot(
                task, current_time, resource_usage
            )
            
            task.scheduled_start = task_start.isoformat()
            task.actual_start = task_start.isoformat()
            task.status = ScheduleStatus.SCHEDULED
            
            task_end = task_start + timedelta(seconds=task.estimated_duration)
            task.actual_end = task_end.isoformat()
            
            # Actualizar uso de recursos
            for resource_id, amount in task.required_resources.items():
                if resource_id in resource_usage:
                    resource_usage[resource_id].append((
                        task_start, task_end, task_id
                    ))
            
            schedule_tasks.append(task_id)
            current_time = task_end
        
        # Calcular costo total
        total_cost = self._calculate_total_cost(schedule_tasks)
        
        end_time = current_time.isoformat()
        
        schedule_id = str(uuid.uuid4())
        
        schedule = Schedule(
            schedule_id=schedule_id,
            tasks=schedule_tasks,
            start_time=start_time,
            end_time=end_time,
            total_cost=total_cost
        )
        
        self.schedules.append(schedule)
        if len(self.schedules) > self.max_schedules:
            self.schedules = self.schedules[-self.max_schedules:]
        
        return schedule
    
    def _topological_sort(self) -> List[str]:
        """Ordenar tareas topológicamente."""
        # Ordenar por prioridad primero
        tasks_by_priority = sorted(
            self.tasks.items(),
            key=lambda x: (x[1].priority.value, x[0]),
            reverse=True
        )
        
        sorted_tasks = []
        visited = set()
        
        def visit(task_id: str):
            if task_id in visited:
                return
            
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    visit(dep_id)
            
            visited.add(task_id)
            sorted_tasks.append(task_id)
        
        for task_id, _ in tasks_by_priority:
            if task_id not in visited:
                visit(task_id)
        
        return sorted_tasks
    
    def _find_resource_slot(
        self,
        task: ScheduledTask,
        earliest_start: datetime,
        resource_usage: Dict[str, List[tuple]]
    ) -> datetime:
        """Encontrar slot disponible para recursos."""
        start_time = earliest_start
        
        # Verificar disponibilidad de recursos
        while True:
            all_available = True
            
            for resource_id, amount in task.required_resources.items():
                if resource_id not in resource_usage:
                    continue
                
                resource = self.resources[resource_id]
                
                # Verificar si hay suficiente capacidad
                used_at_time = sum(
                    amount for start, end, _ in resource_usage[resource_id]
                    if start <= start_time < end
                )
                
                if used_at_time + amount > resource.capacity:
                    all_available = False
                    # Encontrar próximo slot disponible
                    for start, end, _ in sorted(resource_usage[resource_id], key=lambda x: x[1]):
                        if end > start_time:
                            start_time = end
                            break
                    break
            
            if all_available:
                break
        
        return start_time
    
    def _calculate_total_cost(self, task_ids: List[str]) -> float:
        """Calcular costo total de programación."""
        total_cost = 0.0
        
        for task_id in task_ids:
            task = self.tasks[task_id]
            task_duration = task.estimated_duration
            
            for resource_id, amount in task.required_resources.items():
                if resource_id in self.resources:
                    resource = self.resources[resource_id]
                    total_cost += resource.cost_per_unit * amount * task_duration
        
        return total_cost
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del programador."""
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "total_resources": len(self.resources),
            "total_schedules": len(self.schedules),
            "task_status_counts": status_counts
        }


# Instancia global
_smart_scheduler: Optional[SmartScheduler] = None


def get_smart_scheduler() -> SmartScheduler:
    """Obtener instancia global del programador."""
    global _smart_scheduler
    if _smart_scheduler is None:
        _smart_scheduler = SmartScheduler()
    return _smart_scheduler

