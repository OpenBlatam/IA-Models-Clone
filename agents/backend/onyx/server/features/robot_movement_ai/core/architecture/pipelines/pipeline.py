"""
Main Pipeline Class
===================

Clase principal de pipeline con soporte para middleware y ejecutores.
"""

import logging
from typing import Dict, Any, List, Optional, TypeVar, Callable

from .stages import PipelineStage
from .context import PipelineContext
from .executors import PipelineExecutor, SequentialExecutor
from .middleware import PipelineMiddleware
from .hooks import PipelineEventEmitter, PipelineEvent

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Pipeline:
    """
    Pipeline modular para procesamiento de datos.
    
    Soporta:
    - Múltiples ejecutores (secuencial, paralelo, condicional, async)
    - Middleware (logging, métricas, caché, retry, validación)
    - Contexto compartido
    - Validación de entrada/salida
    """
    
    def __init__(
        self,
        name: str = "pipeline",
        executor: Optional[PipelineExecutor] = None,
        context: Optional[PipelineContext] = None
    ):
        """
        Inicializar pipeline.
        
        Args:
            name: Nombre del pipeline
            executor: Ejecutor (por defecto: SequentialExecutor)
            context: Contexto compartido
        """
        self.name = name
        self.stages: List[PipelineStage] = []
        self.middleware: List[PipelineMiddleware] = []
        self.executor = executor or SequentialExecutor()
        self.context = context or PipelineContext()
        self.events = PipelineEventEmitter()
    
    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        """
        Agregar etapa al pipeline.
        
        Args:
            stage: Etapa
            
        Returns:
            Self para chaining
        """
        self.stages.append(stage)
        logger.debug(f"Etapa agregada: {stage.get_name()}")
        return self
    
    def add_middleware(self, middleware: PipelineMiddleware) -> 'Pipeline':
        """
        Agregar middleware.
        
        Args:
            middleware: Middleware
            
        Returns:
            Self para chaining
        """
        self.middleware.append(middleware)
        logger.debug(f"Middleware agregado: {middleware.__class__.__name__}")
        return self
    
    def set_executor(self, executor: PipelineExecutor) -> 'Pipeline':
        """
        Establecer ejecutor.
        
        Args:
            executor: Ejecutor
            
        Returns:
            Self para chaining
        """
        self.executor = executor
        return self
    
    def process(
        self,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Procesar datos a través del pipeline.
        
        Args:
            data: Datos a procesar
            context: Contexto adicional (opcional)
            
        Returns:
            Datos procesados
        """
        # Emitir evento before_pipeline
        self.events.emit(PipelineEvent.BEFORE_PIPELINE, data, context)
        
        # Fusionar contextos
        merged_context = self.context.to_dict()
        if context:
            merged_context.update(context)
        
        # Aplicar middleware antes de cada etapa
        current_data = data
        
        for stage in self.stages:
            stage_name = stage.get_name()
            
            # Emitir evento before_stage
            self.events.emit(
                PipelineEvent.BEFORE_STAGE,
                stage, current_data, merged_context,
                stage_name=stage_name
            )
            
            # Ejecutar middleware before
            for mw in self.middleware:
                current_data, merged_context = mw.before_stage(
                    stage, current_data, merged_context
                )
            
            # Ejecutar etapa
            result = None
            error = None
            try:
                result = stage.process(current_data, merged_context)
                # Emitir evento on_success
                self.events.emit(
                    PipelineEvent.ON_SUCCESS,
                    stage, result, merged_context,
                    stage_name=stage_name
                )
            except Exception as e:
                error = e
                logger.error(f"Error en etapa '{stage.get_name()}': {e}")
                # Emitir evento on_error
                self.events.emit(
                    PipelineEvent.ON_ERROR,
                    stage, error, current_data, merged_context,
                    stage_name=stage_name
                )
            
            # Ejecutar middleware after
            for mw in reversed(self.middleware):
                result = mw.after_stage(
                    stage, current_data, merged_context, result, error
                )
            
            # Emitir evento after_stage
            self.events.emit(
                PipelineEvent.AFTER_STAGE,
                stage, result, merged_context,
                stage_name=stage_name
            )
            
            if error:
                raise error
            
            current_data = result
        
        # Emitir evento after_pipeline
        self.events.emit(PipelineEvent.AFTER_PIPELINE, current_data, merged_context)
        
        return current_data
    
    def set_context(self, key: str, value: Any) -> 'Pipeline':
        """
        Establecer valor en contexto.
        
        Args:
            key: Clave
            value: Valor
            
        Returns:
            Self para chaining
        """
        self.context.set(key, value)
        return self
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor del contexto.
        
        Args:
            key: Clave
            default: Valor por defecto
            
        Returns:
            Valor
        """
        return self.context.get(key, default)
    
    def clear_context(self) -> 'Pipeline':
        """
        Limpiar contexto.
        
        Returns:
            Self para chaining
        """
        self.context.clear()
        return self
    
    def get_stage_count(self) -> int:
        """Obtener número de etapas."""
        return len(self.stages)
    
    def get_middleware_count(self) -> int:
        """Obtener número de middleware."""
        return len(self.middleware)
    
    async def process_async(
        self,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Procesar datos de forma asíncrona.
        
        Args:
            data: Datos a procesar
            context: Contexto adicional (opcional)
            
        Returns:
            Datos procesados
        """
        import asyncio
        from .stages import AsyncPipelineStage
        
        merged_context = self.context.to_dict()
        if context:
            merged_context.update(context)
        
        current_data = data
        
        for stage in self.stages:
            # Ejecutar middleware before
            for mw in self.middleware:
                current_data, merged_context = mw.before_stage(
                    stage, current_data, merged_context
                )
            
            # Ejecutar etapa (async o sync)
            result = None
            error = None
            try:
                if isinstance(stage, AsyncPipelineStage):
                    result = await stage.process_async(current_data, merged_context)
                else:
                    # Ejecutar sync en thread pool para no bloquear
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, stage.process, current_data, merged_context
                    )
            except Exception as e:
                error = e
                logger.error(f"Error en etapa '{stage.get_name()}': {e}")
            
            # Ejecutar middleware after
            for mw in reversed(self.middleware):
                result = mw.after_stage(
                    stage, current_data, merged_context, result, error
                )
            
            if error:
                raise error
            
            current_data = result
        
        return current_data
    
    def compose(self, other: 'Pipeline') -> 'Pipeline':
        """
        Componer con otro pipeline.
        
        Args:
            other: Otro pipeline
            
        Returns:
            Nuevo pipeline compuesto
        """
        composed = Pipeline(f"{self.name}_composed_{other.name}")
        composed.stages = self.stages + other.stages
        composed.middleware = self.middleware + other.middleware
        composed.context = self.context.copy()
        composed.context.update(other.context.to_dict())
        return composed
    
    def clone(self, name: Optional[str] = None) -> 'Pipeline':
        """
        Clonar pipeline.
        
        Args:
            name: Nuevo nombre (opcional)
            
        Returns:
            Pipeline clonado
        """
        cloned = Pipeline(name or f"{self.name}_clone")
        cloned.stages = list(self.stages)
        cloned.middleware = list(self.middleware)
        cloned.executor = self.executor
        cloned.context = self.context.copy()
        return cloned
    
    def remove_stage(self, stage: PipelineStage) -> 'Pipeline':
        """
        Remover etapa.
        
        Args:
            stage: Etapa a remover
            
        Returns:
            Self para chaining
        """
        if stage in self.stages:
            self.stages.remove(stage)
        return self
    
    def remove_middleware(self, middleware: PipelineMiddleware) -> 'Pipeline':
        """
        Remover middleware.
        
        Args:
            middleware: Middleware a remover
            
        Returns:
            Self para chaining
        """
        if middleware in self.middleware:
            self.middleware.remove(middleware)
        return self
    
    def on(
        self,
        event: PipelineEvent,
        callback: Callable,
        stage_filter: Optional[str] = None
    ) -> 'Pipeline':
        """
        Registrar hook para evento.
        
        Args:
            event: Tipo de evento
            callback: Función callback
            stage_filter: Filtrar por etapa (opcional)
            
        Returns:
            Self para chaining
        """
        self.events.on(event, callback, stage_filter)
        return self

