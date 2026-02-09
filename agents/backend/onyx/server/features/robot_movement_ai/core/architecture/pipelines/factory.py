"""
Pipeline Factory (optimizado)

Factory para crear pipelines de forma fácil y consistente.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from .base import BasePipeline, PipelineStage
from .sequential import SequentialPipeline
from .parallel import ParallelPipeline
from .conditional import ConditionalPipeline

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class PipelineFactory:
    """
    Factory para crear pipelines (optimizado).
    
    Proporciona métodos convenientes para crear diferentes tipos de pipelines.
    """
    
    @staticmethod
    def create_sequential(
        name: str,
        stages: Optional[List[PipelineStage]] = None,
        description: Optional[str] = None,
        stop_on_error: bool = True
    ) -> SequentialPipeline:
        """
        Crear pipeline secuencial (optimizado).
        
        Args:
            name: Nombre del pipeline
            stages: Lista de etapas
            description: Descripción
            stop_on_error: Si detener en error
            
        Returns:
            SequentialPipeline configurado
        """
        return SequentialPipeline(
            name=name,
            description=description,
            stages=stages or [],
            stop_on_error=stop_on_error
        )
    
    @staticmethod
    def create_parallel(
        name: str,
        stages: Optional[List[PipelineStage]] = None,
        description: Optional[str] = None,
        combine_results: bool = True,
        fail_fast: bool = False
    ) -> ParallelPipeline:
        """
        Crear pipeline paralelo (optimizado).
        
        Args:
            name: Nombre del pipeline
            stages: Lista de etapas
            description: Descripción
            combine_results: Si combinar resultados
            fail_fast: Si fallar rápido
            
        Returns:
            ParallelPipeline configurado
        """
        return ParallelPipeline(
            name=name,
            description=description,
            stages=stages or [],
            combine_results=combine_results,
            fail_fast=fail_fast
        )
    
    @staticmethod
    def create_conditional(
        name: str,
        stages: Optional[List[PipelineStage]] = None,
        description: Optional[str] = None
    ) -> ConditionalPipeline:
        """
        Crear pipeline condicional (optimizado).
        
        Args:
            name: Nombre del pipeline
            stages: Lista de etapas
            description: Descripción
            
        Returns:
            ConditionalPipeline configurado
        """
        return ConditionalPipeline(
            name=name,
            description=description,
            stages=stages or []
        )
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> BasePipeline:
        """
        Crear pipeline desde configuración (optimizado).
        
        Args:
            config: Diccionario de configuración con:
                - type: "sequential", "parallel", o "conditional"
                - name: Nombre del pipeline
                - description: Descripción (opcional)
                - stages: Lista de configuraciones de etapas (opcional)
                - options: Opciones adicionales según el tipo
                
        Returns:
            Pipeline configurado
            
        Raises:
            ValueError: Si la configuración es inválida
        """
        pipeline_type = config.get("type", "sequential")
        name = config.get("name")
        
        if not name:
            raise ValueError("Pipeline name is required in config")
        
        description = config.get("description")
        stages_config = config.get("stages", [])
        options = config.get("options", {})
        
        # Crear etapas desde configuración (simplificado)
        stages = []
        # TODO: Implementar creación de etapas desde config
        
        if pipeline_type == "sequential":
            return PipelineFactory.create_sequential(
                name=name,
                description=description,
                stages=stages,
                stop_on_error=options.get("stop_on_error", True)
            )
        elif pipeline_type == "parallel":
            return PipelineFactory.create_parallel(
                name=name,
                description=description,
                stages=stages,
                combine_results=options.get("combine_results", True),
                fail_fast=options.get("fail_fast", False)
            )
        elif pipeline_type == "conditional":
            return PipelineFactory.create_conditional(
                name=name,
                description=description,
                stages=stages
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

