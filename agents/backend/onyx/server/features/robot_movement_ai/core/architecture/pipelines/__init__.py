"""
Pipelines Module (optimizado)

Sistema modular de pipelines para procesamiento de datos y tareas.

Este módulo proporciona:
- Pipelines secuenciales
- Pipelines paralelos
- Pipelines condicionales
- Etapas reutilizables
- Middleware y decoradores
- Manejo de errores robusto
- Monitoreo y observabilidad
- Factory para creación fácil

Uso básico:
    from .sequential import SequentialPipeline
    from .base import PipelineStage
    from .factory import PipelineFactory
    
    # Opción 1: Directamente
    pipeline = SequentialPipeline("my_pipeline")
    pipeline.add_stage(MyStage("stage1"))
    result = await pipeline.execute(input_data)
    
    # Opción 2: Con factory
    pipeline = PipelineFactory.create_sequential("my_pipeline")
    result = await pipeline.execute(input_data)
"""

from .base import (
    BasePipeline,
    PipelineStage,
    PipelineContext,
    PipelineResult,
    PipelineStatus
)

from .sequential import SequentialPipeline
from .parallel import ParallelPipeline
from .conditional import ConditionalPipeline, ConditionalStage
from .stages import (
    TransformStage,
    FilterStage,
    ValidationStage,
    LoggingStage,
    ErrorHandlingStage
)
from .factory import PipelineFactory
from .monitoring import PipelineMonitor, get_monitor
from . import middleware

__all__ = [
    # Base classes
    "BasePipeline",
    "PipelineStage",
    "PipelineContext",
    "PipelineResult",
    "PipelineStatus",
    # Implementations
    "SequentialPipeline",
    "ParallelPipeline",
    "ConditionalPipeline",
    "ConditionalStage",
    # Common stages
    "TransformStage",
    "FilterStage",
    "ValidationStage",
    "LoggingStage",
    "ErrorHandlingStage",
    # Factory and utilities
    "PipelineFactory",
    "PipelineMonitor",
    "get_monitor",
    "middleware"
]
