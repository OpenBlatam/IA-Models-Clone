"""
Parallel Pipeline Implementation
================================

Implementación de pipeline paralelo donde las etapas se ejecutan simultáneamente.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import asyncio
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


def _get_enabled_stages(stages: List[PipelineStage]) -> List[PipelineStage]:
    """
    Filtrar etapas habilitadas (función pura).
    
    Args:
        stages: Lista de etapas
        
    Returns:
        Lista de etapas habilitadas
    """
    return [s for s in stages if s.enabled]


def _calculate_execution_time(start_time: datetime) -> float:
    """
    Calcular tiempo de ejecución (función pura).
    
    Args:
        start_time: Tiempo de inicio
        
    Returns:
        Tiempo de ejecución en segundos
    """
    return (datetime.now(timezone.utc) - start_time).total_seconds()


def _count_successful_stages(results: List[Any]) -> int:
    """
    Contar etapas exitosas (función pura).
    
    Args:
        results: Lista de resultados
        
    Returns:
        Número de etapas exitosas
    """
    return len([
        r for r in results
        if isinstance(r, dict) and r.get("success")
    ])


class ParallelPipeline(BasePipeline[T, Dict[str, Any]]):
    """
    Pipeline paralelo (optimizado).
    
    Ejecuta múltiples etapas en paralelo y combina los resultados.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        stages: Optional[List[PipelineStage]] = None,
        combine_results: bool = True,
        fail_fast: bool = False
    ) -> None:
        """
        Inicializar pipeline paralelo.
        
        Args:
            name: Nombre del pipeline
            description: Descripción del pipeline
            stages: Lista de etapas a ejecutar en paralelo
            combine_results: Si combinar resultados en un dict
            fail_fast: Si fallar inmediatamente si una etapa falla
        """
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        
        super().__init__(name, description, stages)
        self.combine_results = combine_results
        self.fail_fast = fail_fast
    
    async def execute(self, input_data: T) -> PipelineResult[Dict[str, Any]]:
        """
        Ejecutar pipeline en paralelo.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Resultado combinado de todas las etapas
        """
        start_time = datetime.now(timezone.utc)
        
        self.context = PipelineContext(
            pipeline_id=f"{self.name}_{start_time.timestamp()}",
            status=PipelineStatus.RUNNING
        )
        
        try:
            current_data = await self._apply_input_middleware(input_data)
            enabled_stages = _get_enabled_stages(self.stages)
            
            if not enabled_stages:
                logger.warning("No enabled stages to execute")
                return self._create_empty_result(start_time)
            
            results = await self._execute_stages_parallel(enabled_stages, current_data)
            combined_results, errors = self._process_results(results)
            
            if errors and self.fail_fast:
                return self._create_failure_result(errors, combined_results, start_time)
            
            combined_results = await self._apply_output_middleware(combined_results)
            
            return self._create_success_result(
                combined_results,
                errors,
                enabled_stages,
                results,
                start_time
            )
            
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
    
    async def _apply_output_middleware(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def _execute_stages_parallel(
        self,
        stages: List[PipelineStage],
        input_data: T
    ) -> List[Dict[str, Any]]:
        """
        Ejecutar etapas en paralelo.
        
        Args:
            stages: Lista de etapas habilitadas
            input_data: Datos de entrada
            
        Returns:
            Lista de resultados de etapas
        """
        async def execute_stage(stage: PipelineStage) -> Dict[str, Any]:
            """Ejecutar una etapa individual."""
            try:
                stage_start = datetime.now(timezone.utc)
                result = await stage.execute(input_data, self.context)
                stage_duration = _calculate_execution_time(stage_start)
                
                stage._execution_count += 1
                stage._total_execution_time += stage_duration
                
                return {
                    "stage": stage.name,
                    "success": True,
                    "result": result,
                    "execution_time": stage_duration
                }
            except Exception as e:
                logger.error(f"Stage '{stage.name}' failed: {e}", exc_info=True)
                return {
                    "stage": stage.name,
                    "success": False,
                    "error": str(e),
                    "execution_time": 0.0
                }
        
        return await asyncio.gather(
            *[execute_stage(stage) for stage in stages],
            return_exceptions=True
        )
    
    def _process_results(
        self,
        results: List[Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Procesar resultados de etapas (función pura).
        
        Args:
            results: Lista de resultados
            
        Returns:
            Tupla de (resultados combinados, lista de errores)
        """
        combined_results = {}
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    f"Stage execution raised exception: {result}",
                    exc_info=True
                )
                errors.append(str(result))
                continue
            
            stage_name = result["stage"]
            if result["success"]:
                if self.combine_results:
                    combined_results[stage_name] = result["result"]
                else:
                    combined_results[stage_name] = result
            else:
                error_msg = f"{stage_name}: {result.get('error', 'Unknown error')}"
                errors.append(error_msg)
                if self.fail_fast:
                    raise RuntimeError(f"Stage '{stage_name}' failed: {result.get('error')}")
        
        return combined_results, errors
    
    def _create_empty_result(
        self,
        start_time: datetime
    ) -> PipelineResult[Dict[str, Any]]:
        """
        Crear resultado vacío (función pura).
        
        Args:
            start_time: Tiempo de inicio
            
        Returns:
            Resultado vacío
        """
        execution_time = _calculate_execution_time(start_time)
        return PipelineResult(
            success=True,
            data={},
            execution_time=execution_time
        )
    
    def _create_failure_result(
        self,
        errors: List[str],
        combined_results: Dict[str, Any],
        start_time: datetime
    ) -> PipelineResult[Dict[str, Any]]:
        """
        Crear resultado de fallo (función pura).
        
        Args:
            errors: Lista de errores
            combined_results: Resultados parciales
            start_time: Tiempo de inicio
            
        Returns:
            Resultado de fallo
        """
        self.context.status = PipelineStatus.FAILED
        self.context.error = "; ".join(errors)
        self.context.updated_at = datetime.now(timezone.utc)
        
        execution_time = _calculate_execution_time(start_time)
        return PipelineResult(
            success=False,
            error="; ".join(errors),
            execution_time=execution_time,
            metadata={"stage_results": combined_results}
        )
    
    def _create_success_result(
        self,
        combined_results: Dict[str, Any],
        errors: List[str],
        enabled_stages: List[PipelineStage],
        results: List[Any],
        start_time: datetime
    ) -> PipelineResult[Dict[str, Any]]:
        """
        Crear resultado exitoso (función pura).
        
        Args:
            combined_results: Resultados combinados
            errors: Lista de errores (puede estar vacía)
            enabled_stages: Etapas habilitadas
            results: Resultados de etapas
            start_time: Tiempo de inicio
            
        Returns:
            Resultado exitoso
        """
        self.context.status = PipelineStatus.COMPLETED
        self.context.updated_at = datetime.now(timezone.utc)
        
        execution_time = _calculate_execution_time(start_time)
        
        return PipelineResult(
            success=len(errors) == 0,
            data=combined_results,
            error="; ".join(errors) if errors else None,
            execution_time=execution_time,
            metadata={
                "pipeline_id": self.context.pipeline_id,
                "stage_count": len(enabled_stages),
                "successful_stages": _count_successful_stages(results),
                "failed_stages": len(errors)
            }
        )
    
    def _create_error_result(
        self,
        error: str,
        start_time: datetime
    ) -> PipelineResult[Dict[str, Any]]:
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

