"""
Builder Pattern
===============

Builders para construir servicios y pipelines complejos.
"""

import logging
from typing import Dict, Any, Optional, List

from .services import RouteService, TrainingService, InferenceService
from .repositories import RouteRepository, ModelRepository
from .events import EventBus
from .interfaces import IRouteStrategy, IInferenceEngine, ITrainingPipeline

logger = logging.getLogger(__name__)


class RouteServiceBuilder:
    """
    Builder para RouteService.
    """
    
    def __init__(self):
        """Inicializar builder."""
        self._strategies: Dict[str, IRouteStrategy] = {}
        self._repository: Optional[RouteRepository] = None
        self._event_bus: Optional[EventBus] = None
    
    def add_strategy(self, name: str, strategy: IRouteStrategy) -> 'RouteServiceBuilder':
        """
        Agregar estrategia.
        
        Args:
            name: Nombre de la estrategia
            strategy: Instancia de estrategia
            
        Returns:
            Self para chaining
        """
        self._strategies[name] = strategy
        return self
    
    def with_repository(self, repository: RouteRepository) -> 'RouteServiceBuilder':
        """
        Configurar repositorio.
        
        Args:
            repository: Repositorio
            
        Returns:
            Self para chaining
        """
        self._repository = repository
        return self
    
    def with_event_bus(self, event_bus: EventBus) -> 'RouteServiceBuilder':
        """
        Configurar event bus.
        
        Args:
            event_bus: Event bus
            
        Returns:
            Self para chaining
        """
        self._event_bus = event_bus
        return self
    
    def build(self) -> RouteService:
        """
        Construir servicio.
        
        Returns:
            Instancia de RouteService
        """
        if not self._strategies:
            raise ValueError("Al menos una estrategia debe ser agregada")
        
        return RouteService(
            strategy_registry=self._strategies,
            repository=self._repository,
            event_bus=self._event_bus
        )


class TrainingPipelineBuilder:
    """
    Builder para TrainingPipeline.
    """
    
    def __init__(self):
        """Inicializar builder."""
        self._pipeline: Optional[ITrainingPipeline] = None
        self._event_bus: Optional[EventBus] = None
    
    def with_pipeline(self, pipeline: ITrainingPipeline) -> 'TrainingPipelineBuilder':
        """
        Configurar pipeline.
        
        Args:
            pipeline: Pipeline de entrenamiento
            
        Returns:
            Self para chaining
        """
        self._pipeline = pipeline
        return self
    
    def with_event_bus(self, event_bus: EventBus) -> 'TrainingPipelineBuilder':
        """
        Configurar event bus.
        
        Args:
            event_bus: Event bus
            
        Returns:
            Self para chaining
        """
        self._event_bus = event_bus
        return self
    
    def build(self) -> TrainingService:
        """
        Construir servicio.
        
        Returns:
            Instancia de TrainingService
        """
        if not self._pipeline:
            raise ValueError("Pipeline debe ser configurado")
        
        return TrainingService(
            pipeline=self._pipeline,
            event_bus=self._event_bus
        )


class InferencePipelineBuilder:
    """
    Builder para InferencePipeline.
    """
    
    def __init__(self):
        """Inicializar builder."""
        self._inference_engine: Optional[IInferenceEngine] = None
        self._event_bus: Optional[EventBus] = None
    
    def with_inference_engine(
        self,
        engine: IInferenceEngine
    ) -> 'InferencePipelineBuilder':
        """
        Configurar motor de inferencia.
        
        Args:
            engine: Motor de inferencia
            
        Returns:
            Self para chaining
        """
        self._inference_engine = engine
        return self
    
    def with_event_bus(self, event_bus: EventBus) -> 'InferencePipelineBuilder':
        """
        Configurar event bus.
        
        Args:
            event_bus: Event bus
            
        Returns:
            Self para chaining
        """
        self._event_bus = event_bus
        return self
    
    def build(self) -> InferenceService:
        """
        Construir servicio.
        
        Returns:
            Instancia de InferenceService
        """
        if not self._inference_engine:
            raise ValueError("Motor de inferencia debe ser configurado")
        
        return InferenceService(
            inference_engine=self._inference_engine,
            event_bus=self._event_bus
        )

