"""
Unified Orchestrator
====================

Orquestador unificado que integra todos los sistemas de pipelines,
deep learning, y arquitectura modular.
"""

import logging
from typing import Dict, Any, Optional, List, TypeVar, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PipelineType(Enum):
    """Tipos de pipelines disponibles."""
    MODULAR = "modular"  # Sistema modular de pipelines
    DEEP_LEARNING = "deep_learning"  # Pipelines de entrenamiento/inferencia
    TRAINING = "training"  # Pipeline de entrenamiento
    INFERENCE = "inference"  # Pipeline de inferencia
    TRANSFORMER = "transformer"  # Pipeline con Transformers
    DIFFUSION = "diffusion"  # Pipeline con Diffusion models


@dataclass
class OrchestrationConfig:
    """Configuración del orquestador."""
    enable_checkpointing: bool = True
    enable_rollback: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    checkpoint_dir: str = "./checkpoints"
    max_concurrent_pipelines: int = 10
    default_timeout: float = 300.0
    retry_attempts: int = 3


class UnifiedOrchestrator:
    """
    Orquestador unificado que integra todos los sistemas de pipelines.
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        """
        Inicializar orquestador.
        
        Args:
            config: Configuración del orquestador
        """
        self.config = config or OrchestrationConfig()
        self.pipelines: Dict[str, Any] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.metrics: Dict[str, Any] = {}
        
        # Inicializar componentes según configuración
        self._initialize_components()
        
        # Integrar con registro si está disponible
        try:
            from .pipeline_registry import get_registry
            self.registry = get_registry()
        except ImportError:
            self.registry = None
    
    def _initialize_components(self) -> None:
        """Inicializar componentes según configuración."""
        if self.config.enable_checkpointing:
            try:
                from .architecture.pipelines.checkpointing import CheckpointManager
                self.checkpoint_manager = CheckpointManager(
                    checkpoint_dir=self.config.checkpoint_dir
                )
            except ImportError:
                logger.warning("CheckpointManager no disponible")
                self.checkpoint_manager = None
        else:
            self.checkpoint_manager = None
        
        if self.config.enable_metrics:
            try:
                from .architecture.pipelines.metrics import MetricsCollector
                self.metrics_collector = MetricsCollector()
            except ImportError:
                logger.warning("MetricsCollector no disponible")
                self.metrics_collector = None
        else:
            self.metrics_collector = None
    
    def register_pipeline(
        self,
        name: str,
        pipeline: Any,
        pipeline_type: PipelineType = PipelineType.MODULAR
    ) -> None:
        """
        Registrar pipeline.
        
        Args:
            name: Nombre del pipeline
            pipeline: Instancia del pipeline
            pipeline_type: Tipo de pipeline
        """
        self.pipelines[name] = {
            'pipeline': pipeline,
            'type': pipeline_type,
            'registered_at': asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else None
        }
        
        # Registrar también en el registro global si está disponible
        if self.registry:
            self.registry.register(
                name,
                pipeline,
                pipeline_type.value
            )
        
        logger.info(f"Pipeline registrado: {name} ({pipeline_type.value})")
    
    def get_pipeline(self, name: str) -> Optional[Any]:
        """
        Obtener pipeline registrado.
        
        Args:
            name: Nombre del pipeline
            
        Returns:
            Pipeline o None
        """
        pipeline_info = self.pipelines.get(name)
        return pipeline_info['pipeline'] if pipeline_info else None
    
    async def execute_pipeline(
        self,
        name: str,
        data: T,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> T:
        """
        Ejecutar pipeline de forma asíncrona.
        
        Args:
            name: Nombre del pipeline
            data: Datos de entrada
            context: Contexto adicional
            timeout: Timeout en segundos
            
        Returns:
            Resultado del pipeline
            
        Raises:
            ValueError: Si el pipeline no está registrado
            TimeoutError: Si se excede el timeout
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' no está registrado")
        
        pipeline_info = self.pipelines[name]
        pipeline = pipeline_info['pipeline']
        pipeline_type = pipeline_info['type']
        
        timeout = timeout or self.config.default_timeout
        
        # Ejecutar según tipo de pipeline
        if pipeline_type == PipelineType.MODULAR:
            # Pipeline modular
            if hasattr(pipeline, 'process_async'):
                result = await asyncio.wait_for(
                    pipeline.process_async(data, context),
                    timeout=timeout
                )
            else:
                # Ejecutar en thread pool si no es async
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        pipeline.process,
                        data,
                        context
                    ),
                    timeout=timeout
                )
        
        elif pipeline_type == PipelineType.TRAINING:
            # Pipeline de entrenamiento
            if hasattr(pipeline, 'train_async'):
                result = await asyncio.wait_for(
                    pipeline.train_async(data, context),
                    timeout=timeout
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        pipeline.train,
                        data
                    ),
                    timeout=timeout
                )
        
        elif pipeline_type == PipelineType.INFERENCE:
            # Pipeline de inferencia
            if hasattr(pipeline, 'infer_async'):
                result = await asyncio.wait_for(
                    pipeline.infer_async(data, context),
                    timeout=timeout
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        pipeline.infer,
                        data
                    ),
                    timeout=timeout
                )
        
        else:
            # Otros tipos - ejecución genérica
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: pipeline(data) if callable(pipeline) else pipeline.process(data),
                    None
                ),
                timeout=timeout
            )
        
        # Registrar métricas y monitoreo
        if self.metrics_collector:
            self.metrics_collector.record_pipeline_execution(name, True, timeout)
        
        # Registrar en monitor si está disponible
        try:
            from .pipeline_monitor import get_monitor
            monitor = get_monitor()
            monitor.record_execution(name, True, timeout)
        except ImportError:
            pass
        
        return result
    
    def execute_pipeline_sync(
        self,
        name: str,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Ejecutar pipeline de forma síncrona.
        
        Args:
            name: Nombre del pipeline
            data: Datos de entrada
            context: Contexto adicional
            
        Returns:
            Resultado del pipeline
        """
        return asyncio.run(self.execute_pipeline(name, data, context))
    
    async def execute_pipeline_with_retry(
        self,
        name: str,
        data: T,
        context: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None
    ) -> T:
        """
        Ejecutar pipeline con reintentos.
        
        Args:
            name: Nombre del pipeline
            data: Datos de entrada
            context: Contexto adicional
            max_retries: Número máximo de reintentos
            
        Returns:
            Resultado del pipeline
        """
        max_retries = max_retries or self.config.retry_attempts
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self.execute_pipeline(name, data, context)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Intento {attempt + 1}/{max_retries + 1} falló para pipeline '{name}': {e}"
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Todos los intentos fallaron para pipeline '{name}'")
        
        raise last_error
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """
        Listar todos los pipelines registrados.
        
        Returns:
            Lista de información de pipelines
        """
        return [
            {
                'name': name,
                'type': info['type'].value,
                'registered_at': info.get('registered_at')
            }
            for name, info in self.pipelines.items()
        ]
    
    def get_pipeline_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información de un pipeline.
        
        Args:
            name: Nombre del pipeline
            
        Returns:
            Información del pipeline o None
        """
        if name not in self.pipelines:
            return None
        
        info = self.pipelines[name]
        pipeline = info['pipeline']
        
        result = {
            'name': name,
            'type': info['type'].value,
            'registered_at': info.get('registered_at')
        }
        
        # Agregar información específica según tipo
        if hasattr(pipeline, 'get_stage_count'):
            result['stage_count'] = pipeline.get_stage_count()
        
        if hasattr(pipeline, 'get_middleware_count'):
            result['middleware_count'] = pipeline.get_middleware_count()
        
        if hasattr(pipeline, 'get_version'):
            result['version'] = pipeline.get_version()
        
        return result
    
    async def shutdown(self) -> None:
        """Cerrar orquestador y limpiar recursos."""
        # Cancelar tareas activas
        for task_name, task in self.active_tasks.items():
            if not task.done():
                logger.info(f"Cancelando tarea: {task_name}")
                task.cancel()
        
        # Esperar a que terminen
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        logger.info("Orquestador cerrado")


# Instancia global del orquestador
_global_orchestrator: Optional[UnifiedOrchestrator] = None


def get_orchestrator(config: Optional[OrchestrationConfig] = None) -> UnifiedOrchestrator:
    """
    Obtener instancia global del orquestador.
    
    Args:
        config: Configuración (solo se usa en la primera llamada)
        
    Returns:
        Instancia del orquestador
    """
    global _global_orchestrator
    
    if _global_orchestrator is None:
        _global_orchestrator = UnifiedOrchestrator(config)
    
    return _global_orchestrator


def register_pipeline(
    name: str,
    pipeline: Any,
    pipeline_type: PipelineType = PipelineType.MODULAR
) -> None:
    """
    Registrar pipeline en el orquestador global.
    
    Args:
        name: Nombre del pipeline
        pipeline: Instancia del pipeline
        pipeline_type: Tipo de pipeline
    """
    orchestrator = get_orchestrator()
    orchestrator.register_pipeline(name, pipeline, pipeline_type)

