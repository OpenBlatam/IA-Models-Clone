"""
Middleware Module
================

Sistema de middleware para interceptar y procesar tareas.
Permite agregar funcionalidad cross-cutting de forma desacoplada.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Interfaz base para middleware."""
    
    @abstractmethod
    async def before_execution(
        self,
        task: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Ejecutar antes de procesar la tarea.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            Contexto modificado o None
        """
        pass
    
    @abstractmethod
    async def after_execution(
        self,
        task: Any,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Ejecutar después de procesar la tarea.
        
        Args:
            task: Tarea procesada
            result: Resultado de la ejecución
            context: Contexto adicional
            
        Returns:
            Resultado modificado
        """
        pass
    
    async def on_error(
        self,
        task: Any,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ejecutar cuando ocurre un error.
        
        Args:
            task: Tarea que falló
            error: Excepción ocurrida
            context: Contexto adicional
        """
        pass


class LoggingMiddleware(Middleware):
    """Middleware para logging de tareas."""
    
    async def before_execution(
        self,
        task: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        task_id = getattr(task, 'task_id', str(task))
        logger.info(f"Starting task: {task_id}")
        return context
    
    async def after_execution(
        self,
        task: Any,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        task_id = getattr(task, 'task_id', str(task))
        logger.info(f"Completed task: {task_id}")
        return result
    
    async def on_error(
        self,
        task: Any,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        task_id = getattr(task, 'task_id', str(task))
        logger.error(f"Error in task {task_id}: {error}", exc_info=True)


class MetricsMiddleware(Middleware):
    """Middleware para tracking de métricas."""
    
    def __init__(self, metrics_manager):
        self.metrics_manager = metrics_manager
    
    async def before_execution(
        self,
        task: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        if context is None:
            context = {}
        context['start_time'] = asyncio.get_event_loop().time()
        return context
    
    async def after_execution(
        self,
        task: Any,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        if context and 'start_time' in context:
            elapsed = asyncio.get_event_loop().time() - context['start_time']
            self.metrics_manager.record_task_completed()
            # Podría agregar más métricas aquí
        return result
    
    async def on_error(
        self,
        task: Any,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.metrics_manager.record_task_failed()


class ValidationMiddleware(Middleware):
    """Middleware para validación de tareas."""
    
    def __init__(self, validator: Callable[[Any], bool]):
        self.validator = validator
    
    async def before_execution(
        self,
        task: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        if not self.validator(task):
            raise ValueError(f"Task validation failed: {task}")
        return context
    
    async def after_execution(
        self,
        task: Any,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return result
    
    async def on_error(
        self,
        task: Any,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        pass


class MiddlewarePipeline:
    """Pipeline de middleware para procesar tareas."""
    
    def __init__(self, middlewares: Optional[List[Middleware]] = None):
        self.middlewares = middlewares or []
    
    def add_middleware(self, middleware: Middleware) -> None:
        """Agregar middleware al pipeline."""
        self.middlewares.append(middleware)
    
    async def execute(
        self,
        task: Any,
        executor: Callable[[Any], Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Ejecutar tarea a través del pipeline de middleware.
        
        Args:
            task: Tarea a ejecutar
            executor: Función que ejecuta la tarea
            context: Contexto inicial
            
        Returns:
            Resultado de la ejecución
        """
        if context is None:
            context = {}
        
        # Ejecutar before hooks
        for middleware in self.middlewares:
            context_update = await middleware.before_execution(task, context)
            if context_update is not None:
                context.update(context_update)
        
        # Ejecutar tarea
        try:
            result = await executor(task)
            
            # Ejecutar after hooks
            for middleware in reversed(self.middlewares):
                result = await middleware.after_execution(task, result, context)
            
            return result
        
        except Exception as e:
            # Ejecutar error hooks
            for middleware in reversed(self.middlewares):
                await middleware.on_error(task, e, context)
            raise
