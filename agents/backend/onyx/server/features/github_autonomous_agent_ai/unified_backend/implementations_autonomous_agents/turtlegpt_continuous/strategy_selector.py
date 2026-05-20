"""
Strategy Selector Module
========================

Selecciona automáticamente la estrategia más apropiada según la tarea.
"""

import logging
from typing import Dict, Any, Optional
from .models import AgentTask

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Selecciona la estrategia más apropiada para una tarea.
    
    Basado en:
    - Prioridad de la tarea
    - Configuración del agente
    - Complejidad estimada
    """
    
    def __init__(
        self,
        react_enabled: bool = True,
        lats_enabled: bool = True,
        tot_enabled: bool = True,
        react_threshold: int = 7,
        tot_threshold: int = 8,
        lats_threshold: int = 9
    ):
        """
        Inicializar selector de estrategias.
        
        Args:
            react_enabled: Si ReAct está habilitado
            lats_enabled: Si LATS está habilitado
            tot_enabled: Si Tree of Thoughts está habilitado
            react_threshold: Prioridad mínima para usar ReAct
            tot_threshold: Prioridad mínima para usar ToT
            lats_threshold: Prioridad mínima para usar LATS
        """
        self.react_enabled = react_enabled
        self.lats_enabled = lats_enabled
        self.tot_enabled = tot_enabled
        self.react_threshold = react_threshold
        self.tot_threshold = tot_threshold
        self.lats_threshold = lats_threshold
    
    def select_strategy(self, task: AgentTask) -> Dict[str, Any]:
        """
        Seleccionar estrategia para una tarea.
        
        Args:
            task: Tarea a procesar
            
        Returns:
            Dict con información de la estrategia seleccionada:
            {
                "strategy": "lats" | "tot" | "react" | "standard",
                "reason": "razón de la selección",
                "priority": prioridad_usada
            }
        """
        priority = task.priority
        
        # LATS: Máxima prioridad y complejidad
        if self.lats_enabled and priority >= self.lats_threshold:
            return {
                "strategy": "lats",
                "reason": f"High priority task (priority={priority}) requires advanced tree search",
                "priority": priority,
                "enabled": True
            }
        
        # Tree of Thoughts: Alta prioridad, razonamiento deliberado
        if self.tot_enabled and priority >= self.tot_threshold:
            return {
                "strategy": "tot",
                "reason": f"High priority task (priority={priority}) requires deliberate reasoning",
                "priority": priority,
                "enabled": True
            }
        
        # ReAct: Prioridad media-alta, reasoning-acting interleaved
        if self.react_enabled and priority >= self.react_threshold:
            return {
                "strategy": "react",
                "reason": f"Medium-high priority task (priority={priority}) requires reasoning-acting cycle",
                "priority": priority,
                "enabled": True
            }
        
        # Standard: Prioridad normal, método estándar (Generative Agents)
        return {
            "strategy": "standard",
            "reason": f"Standard priority task (priority={priority}) uses default Generative Agents approach",
            "priority": priority,
            "enabled": True
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Obtener información sobre las estrategias disponibles."""
        return {
            "react": {
                "enabled": self.react_enabled,
                "threshold": self.react_threshold,
                "description": "ReAct: Reasoning-Acting interleaved cycle"
            },
            "tot": {
                "enabled": self.tot_enabled,
                "threshold": self.tot_threshold,
                "description": "Tree of Thoughts: Deliberate problem solving"
            },
            "lats": {
                "enabled": self.lats_enabled,
                "threshold": self.lats_threshold,
                "description": "LATS: Unified reasoning, acting, and planning"
            },
            "standard": {
                "enabled": True,
                "threshold": 0,
                "description": "Generative Agents: Standard approach with memory and reflection"
            }
        }
