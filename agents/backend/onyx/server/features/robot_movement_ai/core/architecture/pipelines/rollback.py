"""
Pipeline Rollback
=================

Sistema de rollback y transacciones para pipelines.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from copy import deepcopy

from .stages import PipelineStage

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(frozen=True)
class RollbackPoint:
    """
    Punto de rollback.
    Inmutable para mejor seguridad.
    """
    stage_index: int
    stage_name: str
    data: Any
    context: Dict[str, Any]
    rollback_handler: Optional[Callable[[Any, Dict[str, Any]], Any]] = None


def _find_rollback_point(
    rollback_points: List[RollbackPoint],
    target_index: int
) -> Optional[RollbackPoint]:
    """
    Encontrar punto de rollback (función pura).
    
    Args:
        rollback_points: Lista de puntos de rollback
        target_index: Índice objetivo
        
    Returns:
        Punto de rollback o None
    """
    for point in reversed(rollback_points):
        if point.stage_index <= target_index:
            return point
    return None


def _execute_rollback_handlers(
    rollback_points: List[RollbackPoint],
    target_index: int
) -> None:
    """
    Ejecutar handlers de rollback (función pura).
    
    Args:
        rollback_points: Lista de puntos de rollback
        target_index: Índice objetivo
    """
    for point in reversed(rollback_points):
        if point.stage_index > target_index and point.rollback_handler:
            try:
                point.rollback_handler(point.data, point.context)
            except Exception as e:
                logger.error(f"Error in rollback handler: {e}", exc_info=True)


class RollbackManager:
    """
    Gestor de rollback para pipelines.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializar gestor."""
        self.rollback_points: List[RollbackPoint] = []
        self.transaction_stack: List[List[RollbackPoint]] = []
    
    def create_rollback_point(
        self,
        stage_index: int,
        stage_name: str,
        data: T,
        context: Dict[str, Any],
        rollback_handler: Optional[Callable[[T, Dict[str, Any]], T]] = None
    ) -> None:
        """
        Crear punto de rollback.
        
        Args:
            stage_index: Índice de la etapa
            stage_name: Nombre de la etapa
            data: Datos
            context: Contexto
            rollback_handler: Handler de rollback (opcional)
        """
        if stage_index < 0:
            raise ValueError("Stage index cannot be negative")
        
        if not stage_name:
            raise ValueError("Stage name cannot be empty")
        
        point = RollbackPoint(
            stage_index=stage_index,
            stage_name=stage_name,
            data=deepcopy(data),
            context=deepcopy(context),
            rollback_handler=rollback_handler
        )
        self.rollback_points.append(point)
    
    def begin_transaction(self) -> None:
        """Iniciar transacción."""
        self.transaction_stack.append([])
    
    def commit_transaction(self) -> None:
        """Confirmar transacción."""
        if self.transaction_stack:
            self.transaction_stack.pop()
    
    def rollback_to(
        self,
        stage_index: int,
        execute_handlers: bool = True
    ) -> Optional[RollbackPoint]:
        """
        Hacer rollback a una etapa específica.
        
        Args:
            stage_index: Índice de la etapa objetivo
            execute_handlers: Si ejecutar handlers de rollback
            
        Returns:
            Punto de rollback o None
        """
        if stage_index < 0:
            raise ValueError("Stage index cannot be negative")
        
        target_point = _find_rollback_point(self.rollback_points, stage_index)
        
        if not target_point:
            logger.warning(
                f"No rollback point found for index {stage_index}"
            )
            return None
        
        if execute_handlers:
            _execute_rollback_handlers(self.rollback_points, target_point.stage_index)
        
        self.rollback_points = [
            p for p in self.rollback_points
            if p.stage_index <= target_point.stage_index
        ]
        
        logger.info(
            f"Rollback to stage {target_point.stage_name} "
            f"(index {target_point.stage_index})"
        )
        return target_point
    
    def rollback_last(
        self,
        execute_handlers: bool = True
    ) -> Optional[RollbackPoint]:
        """
        Hacer rollback a la última etapa.
        
        Args:
            execute_handlers: Si ejecutar handlers
            
        Returns:
            Punto de rollback o None
        """
        if not self.rollback_points:
            return None
        
        if len(self.rollback_points) == 1:
            return None
        
        target_index = self.rollback_points[-2].stage_index
        return self.rollback_to(target_index, execute_handlers)
    
    def clear(self) -> None:
        """Limpiar todos los puntos de rollback."""
        self.rollback_points.clear()
        self.transaction_stack.clear()
    
    def get_rollback_points(self) -> List[RollbackPoint]:
        """
        Obtener todos los puntos de rollback.
        
        Returns:
            Lista de puntos de rollback (copia)
        """
        return list(self.rollback_points)


class RollbackMiddleware:
    """
    Middleware para rollback automático.
    Optimizado con mejor validación y manejo de errores.
    """
    
    def __init__(
        self,
        rollback_manager: RollbackManager,
        auto_rollback_on_error: bool = True
    ) -> None:
        """
        Inicializar middleware de rollback.
        
        Args:
            rollback_manager: Gestor de rollback
            auto_rollback_on_error: Hacer rollback automático en error
        """
        if not rollback_manager:
            raise ValueError("rollback_manager cannot be None")
        
        self.rollback_manager = rollback_manager
        self.auto_rollback_on_error = auto_rollback_on_error
    
    def before_stage(
        self,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[T, Optional[Dict[str, Any]]]:
        """
        Crear punto de rollback antes de la etapa.
        
        Args:
            stage: Etapa
            data: Datos
            context: Contexto
            
        Returns:
            Tupla (datos, contexto) sin modificar
        """
        if not context:
            return data, context
        
        stage_index = context.get('_stage_index', 0)
        
        self.rollback_manager.create_rollback_point(
            stage_index=stage_index,
            stage_name=stage.get_name(),
            data=data,
            context=context
        )
        
        return data, context
    
    def after_stage(
        self,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None,
        result: Optional[T] = None,
        error: Optional[Exception] = None
    ) -> Optional[T]:
        """
        Manejar rollback en caso de error.
        
        Args:
            stage: Etapa
            data: Datos de entrada
            context: Contexto
            result: Resultado de la etapa
            error: Error si hubo
            
        Returns:
            Resultado o datos del rollback
        """
        if error and self.auto_rollback_on_error:
            logger.warning("Error detected, performing automatic rollback")
            rollback_point = self.rollback_manager.rollback_last()
            if rollback_point:
                return rollback_point.data
        
        return result


class TransactionalPipeline:
    """
    Pipeline con soporte para transacciones y rollback.
    Optimizado con mejor validación y manejo de errores.
    """
    
    def __init__(
        self,
        name: str = "pipeline",
        rollback_manager: Optional[RollbackManager] = None,
        enable_rollback: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Inicializar pipeline transaccional.
        
        Args:
            name: Nombre del pipeline
            rollback_manager: Gestor de rollback
            enable_rollback: Habilitar rollback
            **kwargs: Argumentos adicionales
        """
        from .pipeline import Pipeline
        
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        
        self._pipeline = Pipeline(name, **kwargs)
        self.rollback_manager = rollback_manager or RollbackManager()
        self.enable_rollback = enable_rollback
        
        if enable_rollback:
            self._pipeline.add_middleware(
                RollbackMiddleware(self.rollback_manager)
            )
    
    def begin_transaction(self) -> 'TransactionalPipeline':
        """
        Iniciar transacción.
        
        Returns:
            Self para chaining
        """
        self.rollback_manager.begin_transaction()
        return self
    
    def commit(self) -> 'TransactionalPipeline':
        """
        Confirmar transacción.
        
        Returns:
            Self para chaining
        """
        self.rollback_manager.commit_transaction()
        return self
    
    def rollback_to_stage(
        self,
        stage_index: int,
        execute_handlers: bool = True
    ) -> Optional[Any]:
        """
        Hacer rollback a una etapa específica.
        
        Args:
            stage_index: Índice de la etapa
            execute_handlers: Si ejecutar handlers
            
        Returns:
            Datos del punto de rollback o None
        """
        point = self.rollback_manager.rollback_to(
            stage_index,
            execute_handlers
        )
        return point.data if point else None
    
    def rollback_last(self, execute_handlers: bool = True) -> Optional[Any]:
        """
        Hacer rollback a la última etapa.
        
        Args:
            execute_handlers: Si ejecutar handlers
            
        Returns:
            Datos del punto de rollback o None
        """
        point = self.rollback_manager.rollback_last(execute_handlers)
        return point.data if point else None
    
    @property
    def stages(self) -> List[PipelineStage]:
        """Obtener etapas del pipeline."""
        return self._pipeline.stages
