"""
Pipeline Hooks and Events
=========================

Sistema de hooks y eventos para pipelines.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, TypeVar
from enum import Enum
from dataclasses import dataclass
from functools import wraps

from .stages import PipelineStage

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PipelineEvent(str, Enum):
    """Eventos del pipeline."""
    BEFORE_PIPELINE = "before_pipeline"
    AFTER_PIPELINE = "after_pipeline"
    BEFORE_STAGE = "before_stage"
    AFTER_STAGE = "after_stage"
    ON_ERROR = "on_error"
    ON_SUCCESS = "on_success"


@dataclass(frozen=True)
class PipelineHook:
    """
    Hook para eventos de pipeline.
    Inmutable para mejor seguridad.
    """
    event: PipelineEvent
    callback: Callable[..., None]
    stage_filter: Optional[str] = None
    
    def should_execute(self, stage_name: Optional[str] = None) -> bool:
        """
        Verificar si el hook debe ejecutarse.
        
        Args:
            stage_name: Nombre de la etapa
            
        Returns:
            True si debe ejecutarse
        """
        if self.stage_filter is None:
            return True
        return stage_name == self.stage_filter
    
    def execute(self, *args: Any, **kwargs: Any) -> None:
        """
        Ejecutar callback con manejo de errores.
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
        """
        if not callable(self.callback):
            logger.warning(f"Callback for event {self.event.value} is not callable")
            return
        
        try:
            self.callback(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error executing hook {self.event.value}: {e}",
                exc_info=True
            )


def _validate_callback(callback: Callable) -> None:
    """
    Validar que el callback sea callable.
    
    Args:
        callback: Callback a validar
        
    Raises:
        TypeError: Si el callback no es callable
    """
    if not callable(callback):
        raise TypeError(f"Callback must be callable, got {type(callback)}")


class PipelineEventEmitter:
    """
    Emisor de eventos para pipelines.
    Optimizado con mejor validación y rendimiento.
    """
    
    def __init__(self) -> None:
        """Inicializar emisor."""
        self._hooks: Dict[PipelineEvent, List[PipelineHook]] = {
            event: [] for event in PipelineEvent
        }
    
    def on(
        self,
        event: PipelineEvent,
        callback: Callable[..., None],
        stage_filter: Optional[str] = None
    ) -> 'PipelineEventEmitter':
        """
        Registrar hook para evento.
        
        Args:
            event: Tipo de evento
            callback: Función callback
            stage_filter: Filtrar por etapa (opcional)
            
        Returns:
            Self para chaining
            
        Raises:
            TypeError: Si el callback no es callable
        """
        _validate_callback(callback)
        
        hook = PipelineHook(event, callback, stage_filter)
        self._hooks[event].append(hook)
        
        return self
    
    def emit(
        self,
        event: PipelineEvent,
        *args: Any,
        stage_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Emitir evento.
        
        Args:
            event: Tipo de evento
            *args: Argumentos para callbacks
            stage_name: Nombre de etapa (opcional)
            **kwargs: Argumentos adicionales
        """
        hooks = self._hooks.get(event, [])
        
        for hook in hooks:
            if hook.should_execute(stage_name):
                hook.execute(*args, **kwargs)
    
    def remove_hook(
        self,
        event: PipelineEvent,
        callback: Callable[..., None]
    ) -> bool:
        """
        Remover hook.
        
        Args:
            event: Tipo de evento
            callback: Callback a remover
            
        Returns:
            True si se removió, False si no se encontró
        """
        if event not in self._hooks:
            return False
        
        original_count = len(self._hooks[event])
        self._hooks[event] = [
            hook for hook in self._hooks[event]
            if hook.callback != callback
        ]
        
        return len(self._hooks[event]) < original_count
    
    def clear_hooks(self, event: Optional[PipelineEvent] = None) -> None:
        """
        Limpiar hooks.
        
        Args:
            event: Tipo de evento (None para todos)
        """
        if event is None:
            for event_type in PipelineEvent:
                self._hooks[event_type].clear()
        else:
            self._hooks[event].clear()
    
    def get_hook_count(self, event: Optional[PipelineEvent] = None) -> int:
        """
        Obtener cantidad de hooks registrados.
        
        Args:
            event: Tipo de evento (None para todos)
            
        Returns:
            Cantidad de hooks
        """
        if event is None:
            return sum(len(hooks) for hooks in self._hooks.values())
        return len(self._hooks.get(event, []))
