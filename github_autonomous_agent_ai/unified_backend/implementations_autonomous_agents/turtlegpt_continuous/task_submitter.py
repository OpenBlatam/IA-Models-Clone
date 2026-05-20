"""
Task Submitter Module
=====================

Gestión centralizada del envío de tareas.
Proporciona una interfaz estructurada para validar, normalizar y enviar tareas.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .models import AgentTask
from .task_validator import TaskValidator
from .task_manager import TaskManager
from .event_publisher import EventPublisher
from .metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class TaskSubmitter:
    """
    Gestor centralizado del envío de tareas.
    
    Proporciona una interfaz estructurada para:
    - Validar tareas
    - Normalizar tareas
    - Enviar tareas al TaskManager
    - Trackear métricas
    - Publicar eventos
    """
    
    def __init__(
        self,
        task_validator: TaskValidator,
        task_manager: TaskManager,
        event_publisher: Optional[EventPublisher] = None,
        metrics_tracker: Optional[MetricsTracker] = None
    ):
        """
        Inicializar gestor de envío de tareas.
        
        Args:
            task_validator: Validador de tareas
            task_manager: Manager de tareas
            event_publisher: Publicador de eventos (opcional)
            metrics_tracker: Tracker de métricas (opcional)
        """
        self.task_validator = task_validator
        self.task_manager = task_manager
        self.event_publisher = event_publisher
        self.metrics_tracker = metrics_tracker
    
    def submit_task(
        self,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enviar una tarea para procesamiento.
        
        Args:
            description: Descripción de la tarea
            priority: Prioridad (1-10, mayor = más prioridad)
            metadata: Metadatos adicionales
        
        Returns:
            ID de la tarea
            
        Raises:
            ValueError: Si la tarea no es válida
        """
        # Validar y normalizar tarea
        normalized_task, error_message = self.task_validator.validate_and_normalize(
            description=description,
            priority=priority,
            metadata=metadata
        )
        
        if error_message:
            raise ValueError(f"Invalid task: {error_message}")
        
        # Enviar tarea normalizada al TaskManager
        task_id = self.task_manager.submit_task(
            normalized_task["description"],
            normalized_task["priority"],
            normalized_task["metadata"]
        )
        
        # Trackear métricas
        if self.metrics_tracker:
            self.metrics_tracker.track_task_processed(task_id=task_id)
        
        # Publicar evento de tarea enviada
        self._publish_task_submitted_event(
            task_id=task_id,
            description=normalized_task["description"],
            priority=normalized_task["priority"],
            metadata=normalized_task["metadata"]
        )
        
        logger.debug(f"Task submitted: {task_id} (priority: {priority})")
        
        return task_id
    
    def submit_task_async(
        self,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> asyncio.Task:
        """
        Enviar una tarea de forma asíncrona.
        
        Args:
            description: Descripción de la tarea
            priority: Prioridad (1-10, mayor = más prioridad)
            metadata: Metadatos adicionales
        
        Returns:
            Task asyncio con el ID de la tarea
        """
        async def submit():
            return self.submit_task(description, priority, metadata)
        
        try:
            loop = asyncio.get_event_loop()
            return loop.create_task(submit())
        except RuntimeError:
            # Si no hay loop, crear uno nuevo
            return asyncio.create_task(submit())
    
    def submit_multiple_tasks(
        self,
        tasks: list[Dict[str, Any]]
    ) -> list[str]:
        """
        Enviar múltiples tareas.
        
        Args:
            tasks: Lista de dicts con 'description', 'priority' (opcional), 'metadata' (opcional)
        
        Returns:
            Lista de IDs de tareas
        """
        task_ids = []
        
        for task_data in tasks:
            try:
                description = task_data.get("description", "")
                priority = task_data.get("priority", 5)
                metadata = task_data.get("metadata")
                
                task_id = self.submit_task(
                    description=description,
                    priority=priority,
                    metadata=metadata
                )
                task_ids.append(task_id)
            except ValueError as e:
                logger.warning(f"Failed to submit task: {e}")
                # Continuar con otras tareas
        
        return task_ids
    
    def _publish_task_submitted_event(
        self,
        task_id: str,
        description: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Publicar evento de tarea enviada.
        
        Args:
            task_id: ID de la tarea
            description: Descripción de la tarea
            priority: Prioridad de la tarea
            metadata: Metadatos adicionales
        """
        if not self.event_publisher:
            return
        
        try:
            # Intentar publicar de forma asíncrona
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self.event_publisher.publish_task_submitted(
                        task_id=task_id,
                        description=description,
                        priority=priority,
                        metadata=metadata
                    )
                )
            else:
                loop.run_until_complete(
                    self.event_publisher.publish_task_submitted(
                        task_id=task_id,
                        description=description,
                        priority=priority,
                        metadata=metadata
                    )
                )
        except RuntimeError:
            # Si no hay loop, crear uno nuevo
            asyncio.run(
                self.event_publisher.publish_task_submitted(
                    task_id=task_id,
                    description=description,
                    priority=priority,
                    metadata=metadata
                )
            )
        except Exception as e:
            logger.warning(f"Error publishing task submitted event: {e}")


def create_task_submitter(
    task_validator: TaskValidator,
    task_manager: TaskManager,
    event_publisher: Optional[EventPublisher] = None,
    metrics_tracker: Optional[MetricsTracker] = None
) -> TaskSubmitter:
    """
    Factory function para crear TaskSubmitter.
    
    Args:
        task_validator: Validador de tareas
        task_manager: Manager de tareas
        event_publisher: Publicador de eventos (opcional)
        metrics_tracker: Tracker de métricas (opcional)
        
    Returns:
        Instancia de TaskSubmitter
    """
    return TaskSubmitter(
        task_validator=task_validator,
        task_manager=task_manager,
        event_publisher=event_publisher,
        metrics_tracker=metrics_tracker
    )


