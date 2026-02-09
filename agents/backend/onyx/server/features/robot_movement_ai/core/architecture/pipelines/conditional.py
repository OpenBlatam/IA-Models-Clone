"""
Conditional Pipeline Implementation
====================================

Implementación de pipeline condicional donde las etapas se ejecutan basadas en condiciones.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar
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


def _is_conditional_stage(stage: PipelineStage) -> bool:
    """
    Verificar si una etapa es condicional (función pura).
    
    Args:
        stage: Etapa a verificar
        
    Returns:
        True si es ConditionalStage
    """
    return isinstance(stage, ConditionalStage)


class ConditionalStage(PipelineStage):
    """
    Etapa condicional (optimizado).
    
    Solo se ejecuta si se cumple una condición.
    """
    
    def __init__(
        self,
        name: str,
        stage: PipelineStage,
        condition: Callable[[Any, PipelineContext], bool],
        description: Optional[str] = None
    ) -> None:
        """
        Inicializar etapa condicional.
        
        Args:
            name: Nombre de la etapa
            stage: Etapa a ejecutar condicionalmente
            condition: Función que retorna True si debe ejecutarse
            description: Descripción
        """
        if not name:
            raise ValueError("Stage name cannot be empty")
        if stage is None:
            raise ValueError("Stage cannot be None")
        if condition is None:
            raise ValueError("Condition cannot be None")
        
        super().__init__(name, description)
        self.stage = stage
        self.condition = condition
    
    async def execute(self, input_data: Any, context: PipelineContext) -> Any:
        """
        Ejecutar etapa si se cumple la condición (optimizado).
        
        Args:
            input_data: Datos de entrada
            context: Contexto del pipeline
            
        Returns:
            Resultado de la etapa o input_data si no se ejecuta
        """
        if self.condition(input_data, context):
            logger.debug(f"Condition met for stage '{self.name}', executing")
            return await self.stage.execute(input_data, context)
        else:
            logger.debug(f"Condition not met for stage '{self.name}', skipping")
            return input_data


class ConditionalPipeline(BasePipeline[T, R]):
    """
    Pipeline condicional (optimizado).
    
    Ejecuta etapas basadas en condiciones evaluadas dinámicamente.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        stages: Optional[List[PipelineStage]] = None
    ) -> None:
        """
        Inicializar pipeline condicional.
        
        Args:
            name: Nombre del pipeline
            description: Descripción del pipeline
            stages: Lista de etapas (pueden ser ConditionalStage)
        """
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        
        super().__init__(name, description, stages)
    
    async def execute(self, input_data: T) -> PipelineResult[R]:
        """
        Ejecutar pipeline con evaluación condicional.
        
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
            current_data, executed_stages = await self._execute_conditional_stages(current_data, start_time)
            current_data = await self._apply_output_middleware(current_data)
            
            return self._create_success_result(current_data, executed_stages, start_time)
            
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
    
    async def _execute_conditional_stages(
        self,
        initial_data: T,
        start_time: datetime
    ) -> tuple[Any, List[str]]:
        """
        Ejecutar etapas condicionalmente.
        
        Args:
            initial_data: Datos iniciales
            start_time: Tiempo de inicio
            
        Returns:
            Tupla de (datos procesados, lista de etapas ejecutadas)
        """
        current_data: Any = initial_data
        executed_stages: List[str] = []
        
        for stage in self.stages:
            if _should_skip_stage(stage):
                logger.debug(f"Skipping disabled stage: {stage.name}")
                continue
            
            if _is_conditional_stage(stage):
                current_data, executed = await self._execute_conditional_stage(
                    stage,
                    current_data
                )
                if executed:
                    executed_stages.append(stage.name)
            else:
                current_data = await self._execute_normal_stage(stage, current_data, start_time)
                executed_stages.append(stage.name)
            
            logger.debug(f"Stage '{stage.name}' processed")
        
        return current_data, executed_stages
    
    async def _execute_conditional_stage(
        self,
        stage: 'ConditionalStage',
        data: Any
    ) -> tuple[Any, bool]:
        """
        Ejecutar etapa condicional.
        
        Args:
            stage: Etapa condicional
            data: Datos de entrada
            
        Returns:
            Tupla de (datos procesados, si se ejecutó)
        """
        logger.debug(f"Evaluating condition for stage: {stage.name}")
        
        if stage.condition(data, self.context):
            logger.debug(f"Condition met for stage '{stage.name}', executing")
            stage_start = datetime.now(timezone.utc)
            result = await stage.execute(data, self.context)
            stage_duration = _calculate_execution_time(stage_start)
            
            stage._execution_count += 1
            stage._total_execution_time += stage_duration
            
            return result, True
        else:
            logger.debug(f"Condition not met, skipping stage: {stage.name}")
            return data, False
    
    async def _execute_normal_stage(
        self,
        stage: PipelineStage,
        data: Any,
        start_time: datetime
    ) -> Any:
        """
        Ejecutar etapa normal.
        
        Args:
            stage: Etapa a ejecutar
            data: Datos de entrada
            start_time: Tiempo de inicio
            
        Returns:
            Datos procesados
        """
        stage_start = datetime.now(timezone.utc)
        result = await stage.execute(data, self.context)
        stage_duration = _calculate_execution_time(stage_start)
        
        stage._execution_count += 1
        stage._total_execution_time += stage_duration
        
        return result
    
    def _create_success_result(
        self,
        data: R,
        executed_stages: List[str],
        start_time: datetime
    ) -> PipelineResult[R]:
        """
        Crear resultado exitoso (función pura).
        
        Args:
            data: Datos de salida
            executed_stages: Lista de etapas ejecutadas
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
            metadata={
                "pipeline_id": self.context.pipeline_id,
                "executed_stages": executed_stages
            }
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

