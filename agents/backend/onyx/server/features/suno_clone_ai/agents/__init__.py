"""
Agents Module - Agentes de IA y Orquestación
Define y gestiona agentes de IA, orquestación de tareas complejas y workflows inteligentes.

Rol en el Ecosistema IA:
- Orquestación de tareas complejas, workflows
- Agentes especializados, multi-step reasoning, tool use
- Automatización inteligente de procesos

Reglas de Importación:
- Puede importar: llm, tools, prompts, tracing
- NO debe importar: chat, server (evitar ciclos)
- Usa inyección de dependencias
"""

from .base import BaseAgent
from .service import AgentService
from .orchestrator import AgentOrchestrator
from .registry import AgentRegistry
from .main import (
    get_agent_service,
    execute_task,
    register_agent,
    initialize_agents,
)

__all__ = [
    # Clases principales
    "BaseAgent",
    "AgentService",
    "AgentOrchestrator",
    "AgentRegistry",
    # Funciones de acceso rápido
    "get_agent_service",
    "execute_task",
    "register_agent",
    "initialize_agents",
]

