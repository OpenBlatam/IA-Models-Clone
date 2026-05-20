"""
Strategy Manager Module
======================

Gestión centralizada de estrategias del agente.
Proporciona acceso unificado a todas las estrategias y sus estados.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Gestor centralizado de estrategias.
    
    Proporciona acceso unificado a todas las estrategias del agente
    y métodos para consultar su estado y disponibilidad.
    """
    
    def __init__(
        self,
        strategies: Dict[str, Any],
        enabled_flags: Dict[str, bool]
    ):
        """
        Inicializar gestor de estrategias.
        
        Args:
            strategies: Diccionario de estrategias
            enabled_flags: Diccionario de flags de habilitación
        """
        self.strategies = strategies
        self.enabled_flags = enabled_flags
    
    def get_strategy(self, strategy_name: str) -> Optional[Any]:
        """
        Obtener una estrategia por nombre.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Estrategia o None si no existe
        """
        strategy_key = f"{strategy_name}_strategy"
        return self.strategies.get(strategy_key)
    
    def is_enabled(self, strategy_name: str) -> bool:
        """
        Verificar si una estrategia está habilitada.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            True si está habilitada
        """
        enabled_key = f"{strategy_name}_enabled"
        return self.enabled_flags.get(enabled_key, False)
    
    def is_available(self, strategy_name: str) -> bool:
        """
        Verificar si una estrategia está disponible (habilitada y creada).
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            True si está disponible
        """
        return self.is_enabled(strategy_name) and self.get_strategy(strategy_name) is not None
    
    def get_available_strategies(self) -> List[str]:
        """
        Obtener lista de estrategias disponibles.
        
        Returns:
            Lista de nombres de estrategias disponibles
        """
        available = []
        strategy_names = ["react", "lats", "tot", "tom", "personality", "toolformer"]
        
        for name in strategy_names:
            if self.is_available(name):
                available.append(name)
        
        return available
    
    def get_enabled_strategies(self) -> List[str]:
        """
        Obtener lista de estrategias habilitadas.
        
        Returns:
            Lista de nombres de estrategias habilitadas
        """
        enabled = []
        strategy_names = ["react", "lats", "tot", "tom", "personality", "toolformer"]
        
        for name in strategy_names:
            if self.is_enabled(name):
                enabled.append(name)
        
        return enabled
    
    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener información sobre todas las estrategias.
        
        Returns:
            Dict con información de cada estrategia
        """
        info = {}
        strategy_names = ["react", "lats", "tot", "tom", "personality", "toolformer"]
        
        for name in strategy_names:
            strategy = self.get_strategy(name)
            info[name] = {
                "enabled": self.is_enabled(name),
                "available": strategy is not None,
                "has_instance": strategy is not None
            }
        
        return info
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """
        Habilitar una estrategia (si existe).
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            True si se habilitó exitosamente
        """
        enabled_key = f"{strategy_name}_enabled"
        if enabled_key in self.enabled_flags:
            self.enabled_flags[enabled_key] = True
            logger.info(f"Strategy {strategy_name} enabled")
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """
        Deshabilitar una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            True si se deshabilitó exitosamente
        """
        enabled_key = f"{strategy_name}_enabled"
        if enabled_key in self.enabled_flags:
            self.enabled_flags[enabled_key] = False
            logger.info(f"Strategy {strategy_name} disabled")
            return True
        return False


def create_strategy_manager(
    strategies: Dict[str, Any],
    enabled_flags: Dict[str, bool]
) -> StrategyManager:
    """
    Factory function para crear StrategyManager.
    
    Args:
        strategies: Diccionario de estrategias
        enabled_flags: Diccionario de flags de habilitación
        
    Returns:
        Instancia de StrategyManager
    """
    return StrategyManager(strategies, enabled_flags)


