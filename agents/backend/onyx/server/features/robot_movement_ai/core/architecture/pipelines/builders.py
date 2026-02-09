"""
Pipeline Builders
=================

Builders y factories para construir pipelines de forma declarativa.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, TypeVar

from .stages import PipelineStage, FunctionStage
from .pipeline import Pipeline
from .context import PipelineContext
from .executors import (
    PipelineExecutor,
    SequentialExecutor,
    ParallelExecutor,
    ConditionalExecutor,
    BatchExecutor,
    StreamExecutor
)
from .middleware import (
    PipelineMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    RetryMiddleware,
    ValidationMiddleware
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PipelineBuilder:
    """
    Builder para construir pipelines de forma declarativa.
    """
    
    def __init__(self, name: str = "pipeline"):
        """
        Inicializar builder.
        
        Args:
            name: Nombre del pipeline
        """
        self.name = name
        self.stages: List[PipelineStage] = []
        self.middleware: List[PipelineMiddleware] = []
        self.executor: Optional[PipelineExecutor] = None
        self.context = PipelineContext()
    
    def add_stage(self, stage: PipelineStage) -> 'PipelineBuilder':
        """
        Agregar etapa.
        
        Args:
            stage: Etapa
            
        Returns:
            Self para chaining
        """
        self.stages.append(stage)
        return self
    
    def add_stage_func(
        self,
        func: Callable[[T, Optional[Dict[str, Any]]], T],
        name: Optional[str] = None
    ) -> 'PipelineBuilder':
        """
        Agregar función como etapa.
        
        Args:
            func: Función
            name: Nombre (opcional)
            
        Returns:
            Self para chaining
        """
        stage = FunctionStage(func, name)
        return self.add_stage(stage)
    
    def add_middleware(self, middleware: PipelineMiddleware) -> 'PipelineBuilder':
        """
        Agregar middleware.
        
        Args:
            middleware: Middleware
            
        Returns:
            Self para chaining
        """
        self.middleware.append(middleware)
        return self
    
    def with_logging(self, log_level: int = logging.DEBUG) -> 'PipelineBuilder':
        """
        Agregar middleware de logging.
        
        Args:
            log_level: Nivel de logging
            
        Returns:
            Self para chaining
        """
        return self.add_middleware(LoggingMiddleware(log_level))
    
    def with_metrics(self, metrics_collector: Optional[Any] = None) -> 'PipelineBuilder':
        """
        Agregar middleware de métricas.
        
        Args:
            metrics_collector: Colector de métricas
            
        Returns:
            Self para chaining
        """
        return self.add_middleware(MetricsMiddleware(metrics_collector))
    
    def with_caching(self, cache: Optional[Dict[str, Any]] = None) -> 'PipelineBuilder':
        """
        Agregar middleware de caché.
        
        Args:
            cache: Diccionario de caché
            
        Returns:
            Self para chaining
        """
        return self.add_middleware(CachingMiddleware(cache))
    
    def with_retry(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> 'PipelineBuilder':
        """
        Agregar middleware de reintentos.
        
        Args:
            max_retries: Número máximo de reintentos
            retry_delay: Delay entre reintentos
            
        Returns:
            Self para chaining
        """
        return self.add_middleware(RetryMiddleware(max_retries, retry_delay))
    
    def with_validation(self) -> 'PipelineBuilder':
        """
        Agregar middleware de validación.
        
        Returns:
            Self para chaining
        """
        return self.add_middleware(ValidationMiddleware())
    
    def with_rate_limit(
        self,
        max_calls: int = 100,
        time_window: float = 60.0
    ) -> 'PipelineBuilder':
        """
        Agregar middleware de rate limiting.
        
        Args:
            max_calls: Número máximo de llamadas
            time_window: Ventana de tiempo en segundos
            
        Returns:
            Self para chaining
        """
        from .middleware import RateLimitMiddleware
        return self.add_middleware(RateLimitMiddleware(max_calls, time_window))
    
    def with_timeout(self, timeout: float = 30.0) -> 'PipelineBuilder':
        """
        Agregar middleware de timeout.
        
        Args:
            timeout: Timeout en segundos
            
        Returns:
            Self para chaining
        """
        from .middleware import TimeoutMiddleware
        return self.add_middleware(TimeoutMiddleware(timeout))
    
    def with_circuit_breaker(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0
    ) -> 'PipelineBuilder':
        """
        Agregar middleware de circuit breaker.
        
        Args:
            failure_threshold: Umbral de fallos
            timeout: Timeout en estado abierto
            
        Returns:
            Self para chaining
        """
        from .middleware import CircuitBreakerMiddleware
        return self.add_middleware(
            CircuitBreakerMiddleware(failure_threshold, timeout)
        )
    
    def with_parallel_executor(
        self,
        max_workers: Optional[int] = None
    ) -> 'PipelineBuilder':
        """
        Usar ejecutor paralelo.
        
        Args:
            max_workers: Número máximo de workers
            
        Returns:
            Self para chaining
        """
        self.executor = ParallelExecutor(max_workers)
        return self
    
    def with_batch_executor(self, batch_size: int = 10) -> 'PipelineBuilder':
        """
        Usar ejecutor por lotes.
        
        Args:
            batch_size: Tamaño del lote
            
        Returns:
            Self para chaining
        """
        self.executor = BatchExecutor(batch_size)
        return self
    
    def with_stream_executor(self) -> 'PipelineBuilder':
        """
        Usar ejecutor de streams.
        
        Returns:
            Self para chaining
        """
        self.executor = StreamExecutor()
        return self
    
    def with_conditional_executor(
        self,
        condition: Callable[[T, Dict[str, Any]], bool],
        true_stages: List[PipelineStage],
        false_stages: Optional[List[PipelineStage]] = None
    ) -> 'PipelineBuilder':
        """
        Usar ejecutor condicional.
        
        Args:
            condition: Función de condición
            true_stages: Etapas si condición es verdadera
            false_stages: Etapas si condición es falsa
            
        Returns:
            Self para chaining
        """
        self.executor = ConditionalExecutor(condition, true_stages, false_stages)
        return self
    
    def with_context(self, key: str, value: Any) -> 'PipelineBuilder':
        """
        Agregar al contexto.
        
        Args:
            key: Clave
            value: Valor
            
        Returns:
            Self para chaining
        """
        self.context.set(key, value)
        return self
    
    def build(self) -> Pipeline:
        """
        Construir pipeline.
        
        Returns:
            Pipeline construido
        """
        pipeline = Pipeline(
            name=self.name,
            executor=self.executor,
            context=self.context
        )
        
        for stage in self.stages:
            pipeline.add_stage(stage)
        
        for middleware in self.middleware:
            pipeline.add_middleware(middleware)
        
        return pipeline


class PipelineFactory:
    """
    Factory para crear pipelines pre-configurados.
    """
    
    @staticmethod
    def create_basic_pipeline(name: str = "basic_pipeline") -> Pipeline:
        """
        Crear pipeline básico.
        
        Args:
            name: Nombre del pipeline
            
        Returns:
            Pipeline básico
        """
        return PipelineBuilder(name).build()
    
    @staticmethod
    def create_logged_pipeline(name: str = "logged_pipeline") -> Pipeline:
        """
        Crear pipeline con logging.
        
        Args:
            name: Nombre del pipeline
            
        Returns:
            Pipeline con logging
        """
        return (
            PipelineBuilder(name)
            .with_logging()
            .build()
        )
    
    @staticmethod
    def create_robust_pipeline(
        name: str = "robust_pipeline",
        max_retries: int = 3
    ) -> Pipeline:
        """
        Crear pipeline robusto con logging, retry y validación.
        
        Args:
            name: Nombre del pipeline
            max_retries: Número máximo de reintentos
            
        Returns:
            Pipeline robusto
        """
        return (
            PipelineBuilder(name)
            .with_logging()
            .with_retry(max_retries)
            .with_validation()
            .build()
        )
    
    @staticmethod
    def create_performance_pipeline(
        name: str = "performance_pipeline",
        cache: Optional[Dict[str, Any]] = None,
        metrics_collector: Optional[Any] = None
    ) -> Pipeline:
        """
        Crear pipeline optimizado para rendimiento.
        
        Args:
            name: Nombre del pipeline
            cache: Diccionario de caché
            metrics_collector: Colector de métricas
            
        Returns:
            Pipeline optimizado
        """
        return (
            PipelineBuilder(name)
            .with_logging()
            .with_caching(cache)
            .with_metrics(metrics_collector)
            .build()
        )


class PipelineRegistry:
    """
    Registro de pipelines para reutilización.
    """
    
    def __init__(self):
        """Inicializar registro."""
        self._pipelines: Dict[str, Pipeline] = {}
    
    def register(self, name: str, pipeline: Pipeline) -> None:
        """
        Registrar pipeline.
        
        Args:
            name: Nombre del pipeline
            pipeline: Pipeline
        """
        self._pipelines[name] = pipeline
        logger.info(f"Pipeline registrado: {name}")
    
    def get(self, name: str) -> Optional[Pipeline]:
        """
        Obtener pipeline.
        
        Args:
            name: Nombre del pipeline
            
        Returns:
            Pipeline o None
        """
        return self._pipelines.get(name)
    
    def unregister(self, name: str) -> None:
        """
        Desregistrar pipeline.
        
        Args:
            name: Nombre del pipeline
        """
        if name in self._pipelines:
            del self._pipelines[name]
            logger.info(f"Pipeline desregistrado: {name}")
    
    def list(self) -> List[str]:
        """
        Listar nombres de pipelines.
        
        Returns:
            Lista de nombres
        """
        return list(self._pipelines.keys())
    
    def clear(self) -> None:
        """Limpiar registro."""
        self._pipelines.clear()

