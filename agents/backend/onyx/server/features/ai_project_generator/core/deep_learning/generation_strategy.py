"""
Generation Strategy - Estrategias de Generación
================================================

Define diferentes estrategias para generar componentes según el tipo de proyecto.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from typing import Dict, Any, List, Protocol
from pathlib import Path

from ...shared_utils import get_logger

logger = get_logger(__name__)


class GenerationStrategy(Protocol):
    """Protocolo para estrategias de generación."""
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """
        Determina si esta estrategia debe ejecutarse.
        
        Args:
            keywords: Keywords del proyecto
            
        Returns:
            True si debe ejecutarse, False en caso contrario
        """
        ...
    
    def get_generators(self) -> List[str]:
        """
        Retorna la lista de generadores a usar.
        
        Returns:
            Lista de claves de generadores
        """
        ...


def _should_generate_core(keywords: Dict[str, Any]) -> bool:
    """
    Determina si se debe generar el modelo core (función pura).
    
    Args:
        keywords: Keywords del proyecto
        
    Returns:
        True si debe generarse, False en caso contrario
    """
    return bool(keywords.get("is_deep_learning") or keywords.get("requires_pytorch"))


def _should_generate_training(keywords: Dict[str, Any]) -> bool:
    """
    Determina si se debe generar entrenamiento (función pura).
    
    Args:
        keywords: Keywords del proyecto
        
    Returns:
        True si debe generarse, False en caso contrario
    """
    return bool(keywords.get("requires_training") or keywords.get("is_deep_learning"))


def _should_generate_interface(keywords: Dict[str, Any]) -> bool:
    """
    Determina si se debe generar interfaz (función pura).
    
    Args:
        keywords: Keywords del proyecto
        
    Returns:
        True si debe generarse, False en caso contrario
    """
    return bool(keywords.get("requires_gradio", False))


def _should_generate_config(keywords: Dict[str, Any]) -> bool:
    """
    Determina si se debe generar configuración (función pura).
    
    Args:
        keywords: Keywords del proyecto
        
    Returns:
        True si debe generarse, False en caso contrario
    """
    return bool(keywords.get("is_deep_learning", False))


class CoreModelStrategy:
    """
    Estrategia para generar componentes core del modelo.
    Optimizado con funciones puras.
    """
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """
        Determina si esta estrategia debe ejecutarse.
        
        Args:
            keywords: Keywords del proyecto
            
        Returns:
            True si debe ejecutarse, False en caso contrario
        """
        return _should_generate_core(keywords)
    
    def get_generators(self) -> List[str]:
        """
        Retorna la lista de generadores a usar.
        
        Returns:
            Lista de claves de generadores
        """
        return ["model"]


class TrainingStrategy:
    """
    Estrategia para generar componentes de entrenamiento.
    Optimizado con funciones puras.
    """
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """
        Determina si esta estrategia debe ejecutarse.
        
        Args:
            keywords: Keywords del proyecto
            
        Returns:
            True si debe ejecutarse, False en caso contrario
        """
        return _should_generate_training(keywords)
    
    def get_generators(self) -> List[str]:
        """
        Retorna la lista de generadores a usar.
        
        Returns:
            Lista de claves de generadores
        """
        from .generator_config import TRAINING_UTILS
        return TRAINING_UTILS


class InterfaceStrategy:
    """
    Estrategia para generar interfaces.
    Optimizado con funciones puras.
    """
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """
        Determina si esta estrategia debe ejecutarse.
        
        Args:
            keywords: Keywords del proyecto
            
        Returns:
            True si debe ejecutarse, False en caso contrario
        """
        return _should_generate_interface(keywords)
    
    def get_generators(self) -> List[str]:
        """
        Retorna la lista de generadores a usar.
        
        Returns:
            Lista de claves de generadores
        """
        return ["interface"]


class ConfigStrategy:
    """
    Estrategia para generar configuraciones.
    Optimizado con funciones puras.
    """
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """
        Determina si esta estrategia debe ejecutarse.
        
        Args:
            keywords: Keywords del proyecto
            
        Returns:
            True si debe ejecutarse, False en caso contrario
        """
        return _should_generate_config(keywords)
    
    def get_generators(self) -> List[str]:
        """
        Retorna la lista de generadores a usar.
        
        Returns:
            Lista de claves de generadores
        """
        return ["config"]


def _get_unique_generators(generators: List[str]) -> List[str]:
    """
    Obtiene generadores únicos preservando orden (función pura).
    
    Args:
        generators: Lista de generadores
        
    Returns:
        Lista de generadores únicos
    """
    return list(dict.fromkeys(generators))


class StrategyOrchestrator:
    """
    Orquestador que ejecuta múltiples estrategias de generación.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self, strategies: List[GenerationStrategy]) -> None:
        """
        Inicializa el orquestador.
        
        Args:
            strategies: Lista de estrategias a ejecutar
            
        Raises:
            ValueError: Si strategies está vacío
        """
        if not strategies:
            raise ValueError("strategies cannot be empty")
        
        self.strategies = strategies
    
    def get_generators_to_run(self, keywords: Dict[str, Any]) -> List[str]:
        """
        Determina qué generadores ejecutar según las estrategias activas.
        
        Args:
            keywords: Keywords del proyecto
            
        Returns:
            Lista de claves de generadores a ejecutar
            
        Raises:
            ValueError: Si keywords está vacío
        """
        if not keywords:
            raise ValueError("keywords cannot be empty")
        
        generators: List[str] = []
        
        for strategy in self.strategies:
            try:
                if strategy.should_generate(keywords):
                    generators.extend(strategy.get_generators())
            except Exception as e:
                logger.warning(f"Error in strategy {type(strategy).__name__}: {e}")
                continue
        
        return _get_unique_generators(generators)
    
    @classmethod
    def create_default(cls) -> "StrategyOrchestrator":
        """
        Crea un orquestador con las estrategias por defecto.
        
        Returns:
            Instancia de StrategyOrchestrator configurada
        """
        strategies = [
            CoreModelStrategy(),
            TrainingStrategy(),
            InterfaceStrategy(),
            ConfigStrategy(),
        ]
        return cls(strategies)
