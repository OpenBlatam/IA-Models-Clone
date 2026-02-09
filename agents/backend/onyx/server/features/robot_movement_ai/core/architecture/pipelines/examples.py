"""
Pipeline Examples
================

Ejemplos de uso del sistema de pipelines modular.
"""

import asyncio
import logging
from typing import Dict, Any, List

from .pipeline import Pipeline
from .builders import PipelineBuilder, PipelineFactory
from .stages import FunctionStage, CompositeStage
from .middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    RetryMiddleware,
    ValidationMiddleware,
    RateLimitMiddleware,
    TimeoutMiddleware,
    CircuitBreakerMiddleware
)
from .executors import (
    SequentialExecutor,
    ParallelExecutor,
    BatchExecutor,
    StreamExecutor
)
from .hooks import PipelineEvent, PipelineEventEmitter
from .metrics import MetricsCollector
from .decorators import pipeline_stage, retry_stage, cache_stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Ejemplo 1: Pipeline Básico
# ============================================================================

def ejemplo_basico():
    """Pipeline básico con funciones."""
    def multiplicar_por_dos(data: int, context: Dict[str, Any] = None) -> int:
        return data * 2
    
    def sumar_diez(data: int, context: Dict[str, Any] = None) -> int:
        return data + 10
    
    pipeline = Pipeline("basico")
    pipeline.add_stage(FunctionStage(multiplicar_por_dos))
    pipeline.add_stage(FunctionStage(sumar_diez))
    
    resultado = pipeline.process(5)
    print(f"Ejemplo básico: {resultado}")  # 20


# ============================================================================
# Ejemplo 2: Pipeline con Builder
# ============================================================================

def ejemplo_builder():
    """Pipeline construido con builder."""
    pipeline = (
        PipelineBuilder("con_builder")
        .add_stage_func(lambda x, ctx: x * 2, name="multiplicar")
        .add_stage_func(lambda x, ctx: x + 10, name="sumar")
        .with_logging()
        .with_retry(max_retries=3)
        .build()
    )
    
    resultado = pipeline.process(5)
    print(f"Ejemplo builder: {resultado}")  # 20


# ============================================================================
# Ejemplo 3: Pipeline con Factory
# ============================================================================

def ejemplo_factory():
    """Pipeline creado con factory."""
    # Pipeline robusto pre-configurado
    pipeline = PipelineFactory.create_robust_pipeline("robusto", max_retries=5)
    
    pipeline.add_stage_func(lambda x, ctx: x * 2)
    pipeline.add_stage_func(lambda x, ctx: x + 10)
    
    resultado = pipeline.process(5)
    print(f"Ejemplo factory: {resultado}")  # 20


# ============================================================================
# Ejemplo 4: Pipeline con Métricas
# ============================================================================

def ejemplo_metricas():
    """Pipeline con métricas."""
    collector = MetricsCollector()
    
    pipeline = (
        PipelineBuilder("con_metricas")
        .add_stage_func(lambda x, ctx: x * 2)
        .add_stage_func(lambda x, ctx: x + 10)
        .with_metrics(collector)
        .build()
    )
    
    # Procesar múltiples veces
    for i in range(10):
        pipeline.process(i)
    
    # Obtener métricas
    metrics = collector.get_pipeline_metrics("con_metricas")
    print(f"Ejecuciones: {metrics.execution_count}")
    print(f"Duración promedio: {metrics.average_duration:.4f}s")
    print(f"Tasa de éxito: {metrics.success_rate:.2%}")


# ============================================================================
# Ejemplo 5: Pipeline Paralelo
# ============================================================================

def ejemplo_paralelo():
    """Pipeline con ejecución paralela."""
    def etapa1(data: int, context: Dict[str, Any] = None) -> int:
        return data * 2
    
    def etapa2(data: int, context: Dict[str, Any] = None) -> int:
        return data + 10
    
    def etapa3(data: int, context: Dict[str, Any] = None) -> int:
        return data - 5
    
    pipeline = Pipeline(
        "paralelo",
        executor=ParallelExecutor(max_workers=3)
    )
    
    pipeline.add_stage(FunctionStage(etapa1))
    pipeline.add_stage(FunctionStage(etapa2))
    pipeline.add_stage(FunctionStage(etapa3))
    
    # Nota: ParallelExecutor requiere merge_strategy personalizada
    # para combinar resultados correctamente


# ============================================================================
# Ejemplo 6: Pipeline con Caché
# ============================================================================

def ejemplo_cache():
    """Pipeline con caché."""
    cache = {}
    
    def operacion_costosa(data: int, context: Dict[str, Any] = None) -> int:
        # Simular operación costosa
        import time
        time.sleep(0.1)
        return data * 2
    
    pipeline = (
        PipelineBuilder("con_cache")
        .add_stage_func(operacion_costosa)
        .with_caching(cache)
        .build()
    )
    
    # Primera ejecución (lento)
    resultado1 = pipeline.process(5)
    
    # Segunda ejecución (rápido, desde caché)
    resultado2 = pipeline.process(5)
    
    print(f"Resultado 1: {resultado1}, Resultado 2: {resultado2}")


# ============================================================================
# Ejemplo 7: Pipeline con Rate Limiting
# ============================================================================

def ejemplo_rate_limit():
    """Pipeline con rate limiting."""
    pipeline = (
        PipelineBuilder("con_rate_limit")
        .add_stage_func(lambda x, ctx: x * 2)
        .with_rate_limit(max_calls=5, time_window=10.0)
        .build()
    )
    
    # Procesar múltiples veces
    for i in range(10):
        try:
            pipeline.process(i)
        except RuntimeError as e:
            print(f"Rate limit excedido: {e}")
            break


# ============================================================================
# Ejemplo 8: Pipeline con Timeout
# ============================================================================

def ejemplo_timeout():
    """Pipeline con timeout."""
    def operacion_lenta(data: int, context: Dict[str, Any] = None) -> int:
        import time
        time.sleep(5)  # Operación lenta
        return data * 2
    
    pipeline = (
        PipelineBuilder("con_timeout")
        .add_stage_func(operacion_lenta)
        .with_timeout(timeout=2.0)
        .build()
    )
    
    try:
        pipeline.process(5)
    except TimeoutError as e:
        print(f"Timeout: {e}")


# ============================================================================
# Ejemplo 9: Pipeline con Circuit Breaker
# ============================================================================

def ejemplo_circuit_breaker():
    """Pipeline con circuit breaker."""
    def etapa_inestable(data: int, context: Dict[str, Any] = None) -> int:
        import random
        if random.random() < 0.5:  # 50% de fallos
            raise ValueError("Error simulado")
        return data * 2
    
    pipeline = (
        PipelineBuilder("con_circuit_breaker")
        .add_stage_func(etapa_inestable)
        .with_circuit_breaker(failure_threshold=3, timeout=10.0)
        .build()
    )
    
    # Intentar múltiples veces
    for i in range(10):
        try:
            resultado = pipeline.process(i)
            print(f"Éxito: {resultado}")
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")


# ============================================================================
# Ejemplo 10: Pipeline con Hooks/Eventos
# ============================================================================

def ejemplo_hooks():
    """Pipeline con hooks y eventos."""
    pipeline = Pipeline("con_hooks")
    
    # Registrar hooks
    pipeline.on(
        PipelineEvent.BEFORE_STAGE,
        lambda stage, data, ctx: print(f"Antes de: {stage.get_name()}")
    )
    
    pipeline.on(
        PipelineEvent.AFTER_STAGE,
        lambda stage, result, ctx: print(f"Después de: {stage.get_name()}, resultado: {result}")
    )
    
    pipeline.on(
        PipelineEvent.ON_ERROR,
        lambda stage, error, data, ctx: print(f"Error en {stage.get_name()}: {error}")
    )
    
    pipeline.add_stage_func(lambda x, ctx: x * 2)
    pipeline.add_stage_func(lambda x, ctx: x + 10)
    
    resultado = pipeline.process(5)
    print(f"Resultado: {resultado}")


# ============================================================================
# Ejemplo 11: Pipeline Asíncrono
# ============================================================================

async def ejemplo_async():
    """Pipeline asíncrono."""
    from .stages import AsyncPipelineStage
    
    class AsyncStage(AsyncPipelineStage):
        async def process_async(
            self,
            data: int,
            context: Dict[str, Any] = None
        ) -> int:
            await asyncio.sleep(0.1)
            return data * 2
    
    pipeline = Pipeline("async")
    pipeline.add_stage(AsyncStage())
    
    resultado = await pipeline.process_async(5)
    print(f"Resultado async: {resultado}")


# ============================================================================
# Ejemplo 12: Pipeline con Decoradores
# ============================================================================

def ejemplo_decoradores():
    """Pipeline con decoradores."""
    @pipeline_stage(name="multiplicar")
    @retry_stage(max_retries=3)
    @cache_stage()
    def multiplicar_por_dos(data: int, context: Dict[str, Any] = None) -> int:
        return data * 2
    
    pipeline = Pipeline("con_decoradores")
    pipeline.add_stage(multiplicar_por_dos)
    
    resultado = pipeline.process(5)
    print(f"Resultado con decoradores: {resultado}")


# ============================================================================
# Ejemplo 13: Pipeline Compuesto
# ============================================================================

def ejemplo_composicion():
    """Composición de pipelines."""
    pipeline1 = (
        PipelineBuilder("pipeline1")
        .add_stage_func(lambda x, ctx: x * 2)
        .build()
    )
    
    pipeline2 = (
        PipelineBuilder("pipeline2")
        .add_stage_func(lambda x, ctx: x + 10)
        .build()
    )
    
    # Componer pipelines
    pipeline_compuesto = pipeline1.compose(pipeline2)
    
    resultado = pipeline_compuesto.process(5)
    print(f"Resultado compuesto: {resultado}")  # 20


# ============================================================================
# Ejemplo 14: Pipeline con Batch Processing
# ============================================================================

def ejemplo_batch():
    """Pipeline con procesamiento por lotes."""
    pipeline = Pipeline(
        "batch",
        executor=BatchExecutor(batch_size=5)
    )
    
    pipeline.add_stage_func(lambda x, ctx: x * 2)
    pipeline.add_stage_func(lambda x, ctx: x + 10)
    
    # Procesar lista de datos
    datos = list(range(20))
    resultados = pipeline.process(datos)
    print(f"Resultados batch: {len(resultados)} elementos")


# ============================================================================
# Ejemplo 15: Pipeline Completo con Todas las Características
# ============================================================================

def ejemplo_completo():
    """Pipeline completo con todas las características."""
    collector = MetricsCollector()
    cache = {}
    
    pipeline = (
        PipelineBuilder("completo")
        .add_stage_func(lambda x, ctx: x * 2, name="multiplicar")
        .add_stage_func(lambda x, ctx: x + 10, name="sumar")
        .with_logging()
        .with_metrics(collector)
        .with_caching(cache)
        .with_retry(max_retries=3)
        .with_validation()
        .with_rate_limit(max_calls=100, time_window=60.0)
        .with_timeout(timeout=30.0)
        .with_circuit_breaker(failure_threshold=5, timeout=60.0)
        .build()
    )
    
    # Registrar hooks
    pipeline.on(
        PipelineEvent.BEFORE_PIPELINE,
        lambda data, ctx: print(f"Iniciando pipeline con datos: {data}")
    )
    
    pipeline.on(
        PipelineEvent.AFTER_PIPELINE,
        lambda result, ctx: print(f"Pipeline completado, resultado: {result}")
    )
    
    # Procesar
    resultado = pipeline.process(5)
    
    # Obtener métricas
    metrics = collector.get_pipeline_metrics("completo")
    print(f"\nMétricas:")
    print(f"  Ejecuciones: {metrics.execution_count}")
    print(f"  Duración promedio: {metrics.average_duration:.4f}s")
    print(f"  Tasa de éxito: {metrics.success_rate:.2%}")
    
    return resultado


if __name__ == "__main__":
    print("=== Ejemplos de Pipelines Modulares ===\n")
    
    ejemplo_basico()
    ejemplo_builder()
    ejemplo_factory()
    ejemplo_metricas()
    ejemplo_cache()
    ejemplo_hooks()
    ejemplo_decoradores()
    ejemplo_composicion()
    ejemplo_completo()
    
    print("\n=== Ejemplos completados ===")

