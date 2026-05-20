"""
Strategy Interface Module
========================

Interfaz base para todas las estrategias de papers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseStrategy(ABC):
    """
    Interfaz base para todas las estrategias de papers.
    
    Todas las estrategias deben implementar este interface
    para garantizar consistencia y facilitar testing.
    """
    
    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecutar la estrategia.
        
        Args:
            task: Descripción de la tarea
            context: Contexto adicional (opcional)
            
        Returns:
            Resultado de la ejecución
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Obtener información sobre la estrategia.
        
        Returns:
            Dict con información de la estrategia
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Verificar si la estrategia está habilitada.
        
        Returns:
            True si está habilitada
        """
        return True
    
    def get_paper_name(self) -> str:
        """
        Obtener nombre del paper en el que se basa.
        
        Returns:
            Nombre del paper
        """
        return "Unknown"


# Type hint para estrategias
from typing import Protocol

class StrategyProtocol(Protocol):
    """Protocolo para estrategias."""
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ejecutar estrategia."""
        ...
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Obtener información de estrategia."""
        ...
