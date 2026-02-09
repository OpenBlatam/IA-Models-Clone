"""
Factory Pattern
===============

Factories para crear instancias de modelos, estrategias y pipelines.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable

from .interfaces import IRouteModel, IRouteStrategy, ITrainingPipeline

logger = logging.getLogger(__name__)


class RouteModelFactory:
    """
    Factory para crear modelos de routing.
    """
    
    _registry: Dict[str, Type[IRouteModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[IRouteModel]):
        """
        Registrar clase de modelo.
        
        Args:
            name: Nombre del modelo
            model_class: Clase del modelo
        """
        cls._registry[name] = model_class
        logger.debug(f"Modelo registrado: {name}")
    
    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> IRouteModel:
        """
        Crear modelo.
        
        Args:
            name: Nombre del modelo
            config: Configuración (opcional)
            
        Returns:
            Instancia del modelo
        """
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(f"Modelo '{name}' no registrado")
        
        config = config or {}
        try:
            return model_class(**config)
        except Exception as e:
            logger.error(f"Error creando modelo '{name}': {e}")
            raise
    
    @classmethod
    def list_models(cls) -> list:
        """Listar modelos disponibles."""
        return list(cls._registry.keys())


class StrategyFactory:
    """
    Factory para crear estrategias de routing.
    """
    
    _registry: Dict[str, Type[IRouteStrategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[IRouteStrategy]):
        """
        Registrar clase de estrategia.
        
        Args:
            name: Nombre de la estrategia
            strategy_class: Clase de la estrategia
        """
        cls._registry[name] = strategy_class
        logger.debug(f"Estrategia registrada: {name}")
    
    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> IRouteStrategy:
        """
        Crear estrategia.
        
        Args:
            name: Nombre de la estrategia
            config: Configuración (opcional)
            
        Returns:
            Instancia de la estrategia
        """
        strategy_class = cls._registry.get(name)
        if not strategy_class:
            raise ValueError(f"Estrategia '{name}' no registrada")
        
        config = config or {}
        try:
            return strategy_class(**config)
        except Exception as e:
            logger.error(f"Error creando estrategia '{name}': {e}")
            raise
    
    @classmethod
    def list_strategies(cls) -> list:
        """Listar estrategias disponibles."""
        return list(cls._registry.keys())


class PipelineFactory:
    """
    Factory para crear pipelines.
    """
    
    _registry: Dict[str, Type[ITrainingPipeline]] = {}
    
    @classmethod
    def register(cls, name: str, pipeline_class: Type[ITrainingPipeline]):
        """
        Registrar clase de pipeline.
        
        Args:
            name: Nombre del pipeline
            pipeline_class: Clase del pipeline
        """
        cls._registry[name] = pipeline_class
        logger.debug(f"Pipeline registrado: {name}")
    
    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ITrainingPipeline:
        """
        Crear pipeline.
        
        Args:
            name: Nombre del pipeline
            config: Configuración (opcional)
            
        Returns:
            Instancia del pipeline
        """
        pipeline_class = cls._registry.get(name)
        if not pipeline_class:
            raise ValueError(f"Pipeline '{name}' no registrado")
        
        config = config or {}
        try:
            return pipeline_class(**config)
        except Exception as e:
            logger.error(f"Error creando pipeline '{name}': {e}")
            raise
    
    @classmethod
    def list_pipelines(cls) -> list:
        """Listar pipelines disponibles."""
        return list(cls._registry.keys())

