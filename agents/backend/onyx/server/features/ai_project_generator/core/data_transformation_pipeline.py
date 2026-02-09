"""
Data Transformation Pipeline - Pipeline de Transformación de Datos
===================================================================

Pipeline de transformación de datos:
- Multi-stage transformations
- Pipeline chaining
- Conditional transformations
- Transformation caching
- Error handling in pipeline
"""

from typing import Optional, Dict, Any, List, Callable
from enum import Enum

from .shared_utils import get_logger

logger = get_logger(__name__)


class TransformationStage:
    """Etapa de transformación"""
    
    def __init__(
        self,
        stage_id: str,
        transformer: Callable,
        condition: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> None:
        self.stage_id = stage_id
        self.transformer = transformer
        self.condition = condition
        self.on_error = on_error
        self.execution_count = 0
        self.error_count = 0
    
    async def execute(self, data: Any) -> Any:
        """Ejecuta transformación"""
        # Verificar condición
        if self.condition:
            try:
                if asyncio.iscoroutinefunction(self.condition):
                    should_execute = await self.condition(data)
                else:
                    should_execute = self.condition(data)
                
                if not should_execute:
                    return data
            except Exception as e:
                logger.error(f"Condition check failed for {self.stage_id}: {e}")
                return data
        
        # Ejecutar transformación
        try:
            self.execution_count += 1
            
            if asyncio.iscoroutinefunction(self.transformer):
                result = await self.transformer(data)
            else:
                result = self.transformer(data)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Transformation failed at {self.stage_id}: {e}")
            
            # Ejecutar handler de error si existe
            if self.on_error:
                try:
                    if asyncio.iscoroutinefunction(self.on_error):
                        return await self.on_error(data, e)
                    else:
                        return self.on_error(data, e)
                except Exception as error_handler_error:
                    logger.error(f"Error handler failed for {self.stage_id}: {error_handler_error}")
            
            # Re-lanzar error si no hay handler
            raise


import asyncio


class DataTransformationPipeline:
    """
    Pipeline de transformación de datos.
    """
    
    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id
        self.stages: List[TransformationStage] = []
        self.cache: Dict[str, Any] = {}
        self.enable_caching = False
    
    def add_stage(
        self,
        stage_id: str,
        transformer: Callable,
        condition: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> None:
        """Agrega etapa al pipeline"""
        stage = TransformationStage(stage_id, transformer, condition, on_error)
        self.stages.append(stage)
        logger.info(f"Stage added to pipeline {self.pipeline_id}: {stage_id}")
    
    async def execute(
        self,
        data: Any,
        cache_key: Optional[str] = None
    ) -> Any:
        """Ejecuta pipeline"""
        # Verificar cache
        if self.enable_caching and cache_key:
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {cache_key}")
                return self.cache[cache_key]
        
        # Ejecutar etapas
        result = data
        for stage in self.stages:
            result = await stage.execute(result)
        
        # Guardar en cache
        if self.enable_caching and cache_key:
            self.cache[cache_key] = result
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del pipeline"""
        return {
            "pipeline_id": self.pipeline_id,
            "stages_count": len(self.stages),
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "execution_count": stage.execution_count,
                    "error_count": stage.error_count
                }
                for stage in self.stages
            ],
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self) -> None:
        """Limpia cache"""
        self.cache.clear()
        logger.info(f"Cache cleared for pipeline {self.pipeline_id}")


def create_transformation_pipeline(pipeline_id: str) -> DataTransformationPipeline:
    """Crea pipeline de transformación"""
    return DataTransformationPipeline(pipeline_id)




