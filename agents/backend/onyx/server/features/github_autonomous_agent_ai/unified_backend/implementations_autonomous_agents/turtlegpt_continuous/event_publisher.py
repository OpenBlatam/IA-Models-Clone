"""
Event Publisher Module
=====================

Helper para publicar eventos de forma consistente y centralizada.
Facilita el uso del EventBus con métodos helper y decoradores.
"""

import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

from .event_system import EventBus, EventType, Event

logger = logging.getLogger(__name__)


class EventPublisher:
    """
    Helper para publicar eventos de forma consistente.
    
    Proporciona métodos convenientes para publicar eventos comunes
    y decoradores para publicar eventos automáticamente.
    """
    
    def __init__(self, event_bus: EventBus, source: Optional[str] = None):
        """
        Inicializar publisher.
        
        Args:
            event_bus: Instancia del EventBus
            source: Fuente por defecto para eventos
        """
        self.event_bus = event_bus
        self.default_source = source or "agent"
    
    async def publish_task_submitted(
        self,
        task_id: str,
        description: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publicar evento de tarea enviada."""
        await self.event_bus.publish(
            EventType.TASK_SUBMITTED,
            {
                "task_id": task_id,
                "description": description,
                "priority": priority,
                "metadata": metadata or {}
            },
            source=self.default_source
        )
    
    async def publish_task_started(
        self,
        task_id: str,
        strategy: Optional[str] = None
    ) -> None:
        """Publicar evento de tarea iniciada."""
        await self.event_bus.publish(
            EventType.TASK_STARTED,
            {
                "task_id": task_id,
                "strategy": strategy
            },
            source=self.default_source
        )
    
    async def publish_task_completed(
        self,
        task_id: str,
        result: Any,
        execution_time: Optional[float] = None
    ) -> None:
        """Publicar evento de tarea completada."""
        await self.event_bus.publish(
            EventType.TASK_COMPLETED,
            {
                "task_id": task_id,
                "result": str(result) if not isinstance(result, (dict, str)) else result,
                "execution_time": execution_time
            },
            source=self.default_source
        )
    
    async def publish_task_failed(
        self,
        task_id: str,
        error: Exception,
        error_message: Optional[str] = None
    ) -> None:
        """Publicar evento de tarea fallida."""
        await self.event_bus.publish(
            EventType.TASK_FAILED,
            {
                "task_id": task_id,
                "error_type": type(error).__name__,
                "error_message": error_message or str(error)
            },
            source=self.default_source
        )
    
    async def publish_reflection_triggered(
        self,
        reason: str,
        insights_count: int = 0
    ) -> None:
        """Publicar evento de reflexión activada."""
        await self.event_bus.publish(
            EventType.REFLECTION_TRIGGERED,
            {
                "reason": reason,
                "insights_count": insights_count
            },
            source=self.default_source
        )
    
    async def publish_learning_opportunity(
        self,
        opportunity_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publicar evento de oportunidad de aprendizaje."""
        await self.event_bus.publish(
            EventType.LEARNING_OPPORTUNITY,
            {
                "opportunity_type": opportunity_type,
                "description": description,
                "metadata": metadata or {}
            },
            source=self.default_source
        )
    
    async def publish_strategy_selected(
        self,
        strategy_name: str,
        task_id: str,
        reason: Optional[str] = None
    ) -> None:
        """Publicar evento de estrategia seleccionada."""
        await self.event_bus.publish(
            EventType.STRATEGY_SELECTED,
            {
                "strategy_name": strategy_name,
                "task_id": task_id,
                "reason": reason
            },
            source=self.default_source
        )
    
    async def publish_memory_updated(
        self,
        memory_type: str,
        operation: str,
        count: int = 0
    ) -> None:
        """Publicar evento de memoria actualizada."""
        await self.event_bus.publish(
            EventType.MEMORY_UPDATED,
            {
                "memory_type": memory_type,  # "episodic" or "semantic"
                "operation": operation,  # "add", "update", "remove"
                "count": count
            },
            source=self.default_source
        )
    
    async def publish_metrics_updated(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Publicar evento de métricas actualizadas."""
        await self.event_bus.publish(
            EventType.METRICS_UPDATED,
            {
                "metrics": metrics
            },
            source=self.default_source
        )
    
    async def publish_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publicar evento de error."""
        await self.event_bus.publish(
            EventType.ERROR_OCCURRED,
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {}
            },
            source=self.default_source
        )
    
    async def publish_agent_started(self) -> None:
        """Publicar evento de agente iniciado."""
        await self.event_bus.publish(
            EventType.AGENT_STARTED,
            {},
            source=self.default_source
        )
    
    async def publish_agent_stopped(self) -> None:
        """Publicar evento de agente detenido."""
        await self.event_bus.publish(
            EventType.AGENT_STOPPED,
            {},
            source=self.default_source
        )
    
    async def publish_custom(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> None:
        """Publicar evento personalizado."""
        await self.event_bus.publish(
            event_type,
            data,
            source=source or self.default_source
        )


def publish_on_success(event_type: EventType, publisher: EventPublisher):
    """
    Decorador para publicar evento cuando una función se ejecuta exitosamente.
    
    Args:
        event_type: Tipo de evento a publicar
        publisher: Instancia de EventPublisher
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                await publisher.publish_custom(
                    event_type,
                    {
                        "function": func.__name__,
                        "success": True,
                        "result": str(result) if result else None
                    }
                )
                return result
            except Exception as e:
                await publisher.publish_error(e, {"function": func.__name__})
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Para funciones síncronas, publicar de forma asíncrona en background
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(publisher.publish_custom(
                        event_type,
                        {
                            "function": func.__name__,
                            "success": True,
                            "result": str(result) if result else None
                        }
                    ))
                except RuntimeError:
                    # No hay loop, crear uno nuevo
                    asyncio.run(publisher.publish_custom(
                        event_type,
                        {
                            "function": func.__name__,
                            "success": True,
                            "result": str(result) if result else None
                        }
                    ))
                return result
            except Exception as e:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(publisher.publish_error(e, {"function": func.__name__}))
                except RuntimeError:
                    asyncio.run(publisher.publish_error(e, {"function": func.__name__}))
                raise
        
        # Determinar si es async o sync
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def create_event_publisher(
    event_bus: EventBus,
    source: Optional[str] = None
) -> EventPublisher:
    """
    Factory function para crear EventPublisher.
    
    Args:
        event_bus: Instancia del EventBus
        source: Fuente por defecto
        
    Returns:
        Instancia de EventPublisher
    """
    return EventPublisher(event_bus, source)


