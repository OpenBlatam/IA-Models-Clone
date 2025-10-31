"""
Base Agent - Clase base para todos los agentes de negocio
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Estados de un agente."""
    IDLE = "idle"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


class AgentType(Enum):
    """Tipos de agentes."""
    MARKETING = "marketing"
    SALES = "sales"
    OPERATIONS = "operations"
    HR = "hr"
    FINANCE = "finance"
    LEGAL = "legal"
    TECHNICAL = "technical"
    CONTENT = "content"
    ANALYTICS = "analytics"
    CUSTOMER_SERVICE = "customer_service"


@dataclass
class AgentTask:
    """Tarea de un agente."""
    task_id: str
    agent_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentMetrics:
    """Métricas de un agente."""
    agent_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    last_activity: Optional[datetime] = None
    uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


class BaseAgent(ABC):
    """
    Clase base para todos los agentes de negocio.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        name: str,
        description: str = "",
        capabilities: List[str] = None,
        configuration: Dict[str, Any] = None
    ):
        """Inicializar agente base."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.configuration = configuration or {}
        
        # Estado del agente
        self.status = AgentStatus.IDLE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Tareas y métricas
        self.current_task: Optional[AgentTask] = None
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        self.metrics = AgentMetrics(agent_id=agent_id)
        
        # Configuración de rendimiento
        self.max_concurrent_tasks = self.configuration.get("max_concurrent_tasks", 1)
        self.task_timeout = self.configuration.get("task_timeout", 300)  # 5 minutos
        self.retry_attempts = self.configuration.get("retry_attempts", 3)
        
        # Eventos y comunicación
        self.event_handlers: Dict[str, List[callable]] = {}
        self.communication_channels: List[str] = []
        
        logger.info(f"Agente {self.name} ({self.agent_id}) inicializado")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Inicializar el agente.
        
        Returns:
            True si la inicialización fue exitosa
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Cerrar el agente.
        
        Returns:
            True si el cierre fue exitoso
        """
        pass
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Ejecutar una tarea específica.
        
        Args:
            task: Tarea a ejecutar
            
        Returns:
            Resultado de la tarea
        """
        pass
    
    async def start(self) -> bool:
        """Iniciar el agente."""
        try:
            if self.status != AgentStatus.IDLE:
                logger.warning(f"Agente {self.agent_id} ya está en estado {self.status.value}")
                return False
            
            self.status = AgentStatus.RUNNING
            success = await self.initialize()
            
            if success:
                # Iniciar loop de procesamiento de tareas
                asyncio.create_task(self._task_processing_loop())
                logger.info(f"Agente {self.agent_id} iniciado exitosamente")
            else:
                self.status = AgentStatus.ERROR
                logger.error(f"Error al inicializar agente {self.agent_id}")
            
            return success
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Error al iniciar agente {self.agent_id}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Detener el agente."""
        try:
            if self.status == AgentStatus.STOPPED:
                return True
            
            self.status = AgentStatus.STOPPED
            success = await self.shutdown()
            
            # Cancelar tarea actual si existe
            if self.current_task:
                self.current_task.status = "cancelled"
                self.current_task = None
            
            logger.info(f"Agente {self.agent_id} detenido")
            return success
            
        except Exception as e:
            logger.error(f"Error al detener agente {self.agent_id}: {e}")
            return False
    
    async def submit_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """
        Enviar una tarea al agente.
        
        Args:
            task_type: Tipo de tarea
            parameters: Parámetros de la tarea
            priority: Prioridad de la tarea (1-10)
            
        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        
        task = AgentTask(
            task_id=task_id,
            agent_id=self.agent_id,
            task_type=task_type,
            parameters=parameters,
            priority=priority
        )
        
        # Agregar a la cola de tareas
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        self.metrics.total_tasks += 1
        self.last_activity = datetime.now()
        
        logger.info(f"Tarea {task_id} enviada al agente {self.agent_id}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de una tarea."""
        # Buscar en tareas actuales
        if self.current_task and self.current_task.task_id == task_id:
            return self._task_to_dict(self.current_task)
        
        # Buscar en cola de tareas
        for task in self.task_queue:
            if task.task_id == task_id:
                return self._task_to_dict(task)
        
        # Buscar en tareas completadas
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return self._task_to_dict(task)
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancelar una tarea."""
        # Cancelar tarea actual
        if self.current_task and self.current_task.task_id == task_id:
            self.current_task.status = "cancelled"
            self.current_task = None
            return True
        
        # Remover de cola de tareas
        for i, task in enumerate(self.task_queue):
            if task.task_id == task_id:
                task.status = "cancelled"
                self.task_queue.pop(i)
                return True
        
        return False
    
    async def _task_processing_loop(self):
        """Loop de procesamiento de tareas."""
        while self.status == AgentStatus.RUNNING:
            try:
                if self.task_queue and not self.current_task:
                    # Obtener siguiente tarea
                    task = self.task_queue.pop(0)
                    self.current_task = task
                    self.status = AgentStatus.BUSY
                    
                    # Ejecutar tarea
                    await self._execute_task_with_retry(task)
                    
                    # Limpiar tarea actual
                    self.current_task = None
                    self.status = AgentStatus.RUNNING
                
                # Esperar antes de la siguiente iteración
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error en loop de procesamiento de agente {self.agent_id}: {e}")
                self.status = AgentStatus.ERROR
                break
    
    async def _execute_task_with_retry(self, task: AgentTask):
        """Ejecutar tarea con reintentos."""
        task.started_at = datetime.now()
        task.status = "running"
        
        for attempt in range(self.retry_attempts):
            try:
                # Ejecutar tarea con timeout
                result = await asyncio.wait_for(
                    self.execute_task(task),
                    timeout=self.task_timeout
                )
                
                # Tarea completada exitosamente
                task.completed_at = datetime.now()
                task.status = "completed"
                task.result = result
                
                self.completed_tasks.append(task)
                self.metrics.completed_tasks += 1
                
                # Actualizar métricas
                execution_time = (task.completed_at - task.started_at).total_seconds()
                self._update_execution_time_metric(execution_time)
                
                logger.info(f"Tarea {task.task_id} completada exitosamente")
                break
                
            except asyncio.TimeoutError:
                error_msg = f"Tarea {task.task_id} excedió el tiempo límite"
                logger.warning(error_msg)
                
                if attempt == self.retry_attempts - 1:
                    task.status = "failed"
                    task.error = error_msg
                    self.metrics.failed_tasks += 1
                    break
                    
            except Exception as e:
                error_msg = f"Error en tarea {task.task_id}: {e}"
                logger.error(error_msg)
                
                if attempt == self.retry_attempts - 1:
                    task.status = "failed"
                    task.error = error_msg
                    self.metrics.failed_tasks += 1
                    break
        
        self.last_activity = datetime.now()
    
    def _update_execution_time_metric(self, execution_time: float):
        """Actualizar métrica de tiempo de ejecución."""
        if self.metrics.completed_tasks == 1:
            self.metrics.average_execution_time = execution_time
        else:
            # Promedio móvil
            self.metrics.average_execution_time = (
                (self.metrics.average_execution_time * (self.metrics.completed_tasks - 1) + execution_time) /
                self.metrics.completed_tasks
            )
    
    def _task_to_dict(self, task: AgentTask) -> Dict[str, Any]:
        """Convertir tarea a diccionario."""
        return {
            "task_id": task.task_id,
            "agent_id": task.agent_id,
            "task_type": task.task_type,
            "parameters": task.parameters,
            "priority": task.priority,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Obtener estado del agente."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "current_task": self._task_to_dict(self.current_task) if self.current_task else None,
            "queue_size": len(self.task_queue),
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "average_execution_time": self.metrics.average_execution_time,
                "success_rate": (
                    self.metrics.completed_tasks / self.metrics.total_tasks * 100
                    if self.metrics.total_tasks > 0 else 0
                )
            }
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del agente."""
        uptime = (datetime.now() - self.created_at).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "status": self.status.value,
            "total_tasks": self.metrics.total_tasks,
            "completed_tasks": self.metrics.completed_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "success_rate": (
                self.metrics.completed_tasks / self.metrics.total_tasks * 100
                if self.metrics.total_tasks > 0 else 0
            ),
            "average_execution_time": self.metrics.average_execution_time,
            "queue_size": len(self.task_queue),
            "last_activity": self.last_activity.isoformat(),
            "memory_usage": self.metrics.memory_usage,
            "cpu_usage": self.metrics.cpu_usage
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del agente."""
        try:
            # Verificar estado básico
            is_healthy = self.status in [AgentStatus.IDLE, AgentStatus.RUNNING, AgentStatus.BUSY]
            
            # Verificar si hay tareas atascadas
            if self.current_task:
                task_age = (datetime.now() - self.current_task.started_at).total_seconds()
                if task_age > self.task_timeout * 2:  # Tarea muy antigua
                    is_healthy = False
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "agent_id": self.agent_id,
                "status_value": self.status.value,
                "current_task": self.current_task.task_id if self.current_task else None,
                "queue_size": len(self.task_queue),
                "last_activity": self.last_activity.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check del agente {self.agent_id}: {e}")
            return {
                "status": "error",
                "agent_id": self.agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def add_event_handler(self, event_type: str, handler: callable):
        """Agregar manejador de eventos."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emitir evento."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error en manejador de evento {event_type}: {e}")
    
    def can_handle_task(self, task_type: str) -> bool:
        """Verificar si el agente puede manejar un tipo de tarea."""
        return task_type in self.capabilities
    
    def get_capabilities(self) -> List[str]:
        """Obtener capacidades del agente."""
        return self.capabilities.copy()
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Actualizar configuración del agente."""
        self.configuration.update(new_config)
        
        # Actualizar configuraciones específicas
        if "max_concurrent_tasks" in new_config:
            self.max_concurrent_tasks = new_config["max_concurrent_tasks"]
        if "task_timeout" in new_config:
            self.task_timeout = new_config["task_timeout"]
        if "retry_attempts" in new_config:
            self.retry_attempts = new_config["retry_attempts"]
        
        logger.info(f"Configuración actualizada para agente {self.agent_id}")




