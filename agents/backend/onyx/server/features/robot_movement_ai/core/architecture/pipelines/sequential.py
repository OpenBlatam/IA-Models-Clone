"""
Sequential Pipeline Implementation
===================================

Implementación de pipeline secuencial donde las etapas se ejecutan una tras otra.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Any, Dict, List, Optional, TypeVar
from datetime import datetime, timezone

from .base import (
    BasePipeline,
    PipelineStage,
    PipelineContext,
    PipelineResult,
    PipelineStatus
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def _should_skip_stage(stage: PipelineStage) -> bool:
    """
    Verificar si se debe saltar una etapa (función pura).
    
    Args:
        stage: Etapa a verificar
        
    Returns:
        True si se debe saltar
    """
    return not stage.enabled


def _calculate_execution_time(start_time: datetime) -> float:
    """
    Calcular tiempo de ejecución (función pura).
    
    Args:
        start_time: Tiempo de inicio
        
    Returns:
        Tiempo de ejecución en segundos
    """
    return (datetime.now(timezone.utc) - start_time).total_seconds()


class SequentialPipeline(BasePipeline[T, R]):
    """
    Pipeline secuencial optimizado.
    
    Ejecuta las etapas en orden, pasando el resultado de una a la siguiente.
    Si una etapa falla, el pipeline se detiene (si stop_on_error=True).
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        stages: Optional[List[PipelineStage]] = None,
        stop_on_error: bool = True
    ) -> None:
        """
        Inicializar pipeline secuencial.
        
        Args:
            name: Nombre del pipeline
            description: Descripción del pipeline
            stages: Lista de etapas
            stop_on_error: Si detener en caso de error
        """
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        
        super().__init__(name, description, stages)
        self.stop_on_error = stop_on_error
    
    async def execute(self, input_data: T) -> PipelineResult[R]:
        """
        Ejecutar pipeline secuencialmente.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Resultado de la ejecución
        """
        start_time = datetime.now(timezone.utc)
        
        self.context = PipelineContext(
            pipeline_id=f"{self.name}_{start_time.timestamp()}",
            status=PipelineStatus.RUNNING
        )
        
        try:
            current_data = await self._apply_input_middleware(input_data)
            current_data = await self._execute_stages(current_data, start_time)
            current_data = await self._apply_output_middleware(current_data)
            
            return self._create_success_result(current_data, start_time)
            
        except Exception as e:
            logger.error(f"Pipeline '{self.name}' failed: {e}", exc_info=True)
            return self._create_error_result(str(e), start_time)
    
    async def _apply_input_middleware(self, data: T) -> T:
        """
        Aplicar middleware de entrada.
        
        Args:
            data: Datos de entrada
            
        Returns:
            Datos procesados
        """
        current_data = data
        for middleware in self._middleware:
            current_data = await middleware(current_data, self.context)
        return current_data
    
    async def _apply_output_middleware(self, data: R) -> R:
        """
        Aplicar middleware de salida.
        
        Args:
            data: Datos de salida
            
        Returns:
            Datos procesados
        """
        current_data = data
        for middleware in reversed(self._middleware):
            current_data = await middleware(current_data, self.context)
        return current_data
    
    async def _execute_stages(self, initial_data: T, start_time: datetime) -> R:
        """
        Ejecutar todas las etapas secuencialmente.
        
        Args:
            initial_data: Datos iniciales
            start_time: Tiempo de inicio
            
        Returns:
            Resultado final
            
        Raises:
            Exception: Si una etapa falla y stop_on_error=True
        """
        current_data: Any = initial_data
        
        for i, stage in enumerate(self.stages):
            if _should_skip_stage(stage):
                logger.debug(f"Skipping disabled stage: {stage.name}")
                continue
            
            logger.debug(
                f"Executing stage {i+1}/{len(self.stages)}: {stage.name}"
            )
            
            try:
                current_data = await self._execute_stage(stage, current_data)
            except Exception as e:
                if self.stop_on_error:
                    raise
                logger.warning(f"Continuing after stage failure: {stage.name}")
        
        return current_data
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        data: Any
    ) -> Any:
        """
        Ejecutar una etapa individual.
        
        Args:
            stage: Etapa a ejecutar
            data: Datos de entrada
            
        Returns:
            Datos de salida
        """
        stage_start = datetime.now(timezone.utc)
        
        try:
            result = await stage.execute(data, self.context)
            stage_duration = _calculate_execution_time(stage_start)
            
            stage._execution_count += 1
            stage._total_execution_time += stage_duration
            
            logger.debug(
                f"Stage '{stage.name}' completed in {stage_duration:.3f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Stage '{stage.name}' failed: {e}", exc_info=True)
            raise
    
    def _create_success_result(
        self,
        data: R,
        start_time: datetime
    ) -> PipelineResult[R]:
        """
        Crear resultado exitoso (función pura).
        
        Args:
            data: Datos de salida
            start_time: Tiempo de inicio
            
        Returns:
            Resultado exitoso
        """
        self.context.status = PipelineStatus.COMPLETED
        self.context.updated_at = datetime.now(timezone.utc)
        
        execution_time = _calculate_execution_time(start_time)
        
        return PipelineResult(
            success=True,
            data=data,
            execution_time=execution_time,
            metadata={"pipeline_id": self.context.pipeline_id}
        )
    
    def _create_error_result(
        self,
        error: str,
        start_time: datetime
    ) -> PipelineResult[R]:
        """
        Crear resultado de error (función pura).
        
        Args:
            error: Mensaje de error
            start_time: Tiempo de inicio
            
        Returns:
            Resultado de error
        """
        self.context.status = PipelineStatus.FAILED
        self.context.error = error
        self.context.updated_at = datetime.now(timezone.utc)
        
        execution_time = _calculate_execution_time(start_time)
        
        return PipelineResult(
            success=False,
            error=error,
            execution_time=execution_time
        )
