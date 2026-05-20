"""
Data Models for TurtleGPT Continuous Agent

Enums and dataclasses for continuous agent operations.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """Estado de una tarea."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Tarea para el agente."""
    task_id: str
    description: str
    priority: int = 5  # 1-10, mayor = más prioridad
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Métricas del agente."""
    start_time: datetime = field(default_factory=datetime.now)
    total_tasks_processed: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    average_response_time: float = 0.0
    last_activity: Optional[datetime] = None
    errors_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir métricas a diccionario."""
        runtime = datetime.now() - self.start_time
        return {
            "runtime_seconds": runtime.total_seconds(),
            "total_tasks_processed": self.total_tasks_processed,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "average_response_time": self.average_response_time,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "errors_count": self.errors_count,
            "success_rate": (
                (self.total_tasks_completed / self.total_tasks_processed * 100)
                if self.total_tasks_processed > 0 else 0.0
            )
        }


@dataclass
class ContinuousAgentConfig:
    """Configuración del agente continuo."""
    loop_sleep_seconds: float = 1.0  # Tiempo entre iteraciones del loop
    task_monitor_sleep_seconds: float = 0.5  # Tiempo para monitorear tareas
    idle_sleep_seconds: float = 5.0  # Tiempo en modo idle
    max_concurrent_tasks: int = 3  # Máximo de tareas concurrentes
    retry_sleep_seconds: float = 2.0  # Tiempo antes de reintentar
    max_retries: int = 3  # Máximo de reintentos por tarea
    enable_idle_mode: bool = True  # Habilitar modo idle cuando no hay tareas
    maintenance_interval_seconds: float = 300.0  # Intervalo para mantenimiento (5 min)



