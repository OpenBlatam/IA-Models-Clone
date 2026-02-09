"""
Agent Registry - Registro de agentes disponibles
"""

from typing import Dict, Optional
from .base import BaseAgent


class AgentRegistry:
    """Registro de agentes disponibles"""

    def __init__(self):
        """Inicializa el registro"""
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, name: str, agent: BaseAgent) -> None:
        """Registra un agente"""
        self._agents[name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        """Obtiene un agente por nombre"""
        return self._agents.get(name)

    def list(self) -> list:
        """Lista todos los agentes registrados"""
        return list(self._agents.keys())

