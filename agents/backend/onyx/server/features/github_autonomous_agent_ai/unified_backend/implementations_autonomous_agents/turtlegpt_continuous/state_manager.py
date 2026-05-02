"""
State Manager Module
====================

Gestión centralizada del estado del agente.
Proporciona transiciones de estado estructuradas y tracking de cambios.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from ..common.agent_base import AgentState, AgentStatus

logger = logging.getLogger(__name__)


class StateTransition(Enum):
    """Tipos de transiciones de estado."""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    IDLE = "idle"
    ERROR = "error"
    COMPLETED = "completed"


class StateManager:
    """
    Gestor centralizado del estado del agente.
    
    Proporciona transiciones de estado estructuradas y tracking
    de cambios de estado.
    """
    
    def __init__(self, agent_state: AgentState):
        """
        Inicializar gestor de estado.
        
        Args:
            agent_state: Estado del agente
        """
        self.agent_state = agent_state
        self.state_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
    
    def transition_to(
        self,
        status: AgentStatus,
        context: Optional[Dict[str, Any]] = None,
        add_step: bool = True
    ) -> None:
        """
        Transicionar a un nuevo estado.
        
        Args:
            status: Nuevo estado
            context: Contexto adicional
            add_step: Si se debe agregar un paso al historial
        """
        previous_status = self.agent_state.status
        
        self.agent_state.status = status
        
        if add_step:
            step_data = {
                "status": status.value if hasattr(status, 'value') else str(status),
                "previous_status": previous_status.value if previous_status and hasattr(previous_status, 'value') else str(previous_status),
                "timestamp": datetime.now().isoformat(),
                "context": context or {}
            }
            self._add_to_history(step_data)
            
            logger.debug(
                f"State transition: {previous_status} -> {status}",
                extra={"previous": str(previous_status), "current": str(status)}
            )
    
    def add_step(
        self,
        step_type: str,
        data: Dict[str, Any],
        update_status: Optional[AgentStatus] = None
    ) -> None:
        """
        Agregar un paso al historial del estado.
        
        Args:
            step_type: Tipo de paso (thinking, action, observation, etc.)
            data: Datos del paso
            update_status: Estado opcional para actualizar
        """
        if update_status:
            self.transition_to(update_status, context=data, add_step=False)
        
        step = {
            "type": step_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "status": self.agent_state.status.value if hasattr(self.agent_state.status, 'value') else str(self.agent_state.status)
        }
        
        if hasattr(self.agent_state, 'add_step'):
            self.agent_state.add_step(step_type, data)
        
        self._add_to_history(step)
    
    def set_current_task(self, task: str) -> None:
        """
        Establecer tarea actual.
        
        Args:
            task: Descripción de la tarea
        """
        if hasattr(self.agent_state, 'current_task'):
            self.agent_state.current_task = task
            logger.debug(f"Current task set: {task}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual completo.
        
        Returns:
            Dict con estado actual
        """
        status = {
            "status": self.agent_state.status.value if hasattr(self.agent_state.status, 'value') else str(self.agent_state.status),
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(self.agent_state, 'current_task'):
            status["current_task"] = self.agent_state.current_task
        
        if hasattr(self.agent_state, 'to_dict'):
            try:
                state_dict = self.agent_state.to_dict()
                status.update(state_dict)
            except Exception as e:
                logger.warning(f"Error getting state dict: {e}")
        
        return status
    
    def get_state_history(
        self,
        limit: Optional[int] = None,
        step_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de estado.
        
        Args:
            limit: Límite de entradas a retornar
            step_type: Filtrar por tipo de paso
            
        Returns:
            Lista de entradas del historial
        """
        history = self.state_history.copy()
        
        if step_type:
            history = [h for h in history if h.get("type") == step_type]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_recent_steps(
        self,
        step_type: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Obtener pasos recientes de un tipo específico.
        
        Args:
            step_type: Tipo de paso
            count: Número de pasos a obtener
            
        Returns:
            Lista de pasos
        """
        steps = [
            h for h in self.state_history
            if h.get("type") == step_type
        ]
        return steps[-count:]
    
    def clear_history(self) -> None:
        """Limpiar historial de estado."""
        self.state_history.clear()
        logger.debug("State history cleared")
    
    def _add_to_history(self, entry: Dict[str, Any]) -> None:
        """
        Agregar entrada al historial.
        
        Args:
            entry: Entrada a agregar
        """
        self.state_history.append(entry)
        
        # Limitar tamaño del historial
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
    
    def get_state_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del estado.
        
        Returns:
            Dict con estadísticas
        """
        stats = {
            "current_status": str(self.agent_state.status),
            "history_size": len(self.state_history),
            "step_types": {}
        }
        
        # Contar pasos por tipo
        for entry in self.state_history:
            step_type = entry.get("type", "unknown")
            stats["step_types"][step_type] = stats["step_types"].get(step_type, 0) + 1
        
        return stats
    
    def is_in_status(self, status: AgentStatus) -> bool:
        """
        Verificar si el agente está en un estado específico.
        
        Args:
            status: Estado a verificar
            
        Returns:
            True si está en el estado
        """
        return self.agent_state.status == status
    
    def can_transition_to(self, target_status: AgentStatus) -> bool:
        """
        Verificar si se puede transicionar a un estado.
        
        Args:
            target_status: Estado objetivo
            
        Returns:
            True si la transición es válida
        """
        # Lógica básica: permitir todas las transiciones
        # Se puede extender con reglas específicas
        current = self.agent_state.status
        
        # No permitir transición al mismo estado
        if current == target_status:
            return False
        
        # Permitir todas las demás transiciones
        return True


def create_state_manager(agent_state: AgentState) -> StateManager:
    """
    Factory function para crear StateManager.
    
    Args:
        agent_state: Estado del agente
        
    Returns:
        Instancia de StateManager
    """
    return StateManager(agent_state)


