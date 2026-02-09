"""
Agents Main - Funciones base y entry points del módulo de agentes

Rol en el Ecosistema IA:
- Orquestación de tareas complejas, workflows
- Agentes especializados, multi-step reasoning, tool use
- Automatización inteligente de procesos
"""

from typing import Optional, Dict, Any
from .service import AgentService
from .orchestrator import AgentOrchestrator
from .registry import AgentRegistry
from llm.main import get_llm_service
from tools.main import get_tool_service
from prompts.main import get_prompt_service
from tracing.main import get_tracing_service


# Instancia global del servicio
_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """
    Obtiene la instancia global del servicio de agentes.
    
    Returns:
        AgentService: Servicio de agentes
    """
    global _agent_service
    if _agent_service is None:
        llm_service = get_llm_service()
        tool_service = get_tool_service()
        prompt_service = get_prompt_service()
        tracing_service = get_tracing_service()
        _agent_service = AgentService(
            llm_service=llm_service,
            tool_service=tool_service,
            prompt_service=prompt_service,
            tracing_service=tracing_service
        )
    return _agent_service


async def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta una tarea usando agentes.
    
    Args:
        task: Diccionario con la tarea a ejecutar
        
    Returns:
        Resultado de la tarea
    """
    service = get_agent_service()
    return await service.execute_task(task)


def register_agent(name: str, agent) -> None:
    """
    Registra un agente en el sistema.
    
    Args:
        name: Nombre del agente
        agent: Instancia del agente
    """
    service = get_agent_service()
    service.register_agent(name, agent)


def initialize_agents() -> AgentService:
    """
    Inicializa el sistema de agentes.
    
    Returns:
        AgentService: Servicio inicializado
    """
    return get_agent_service()

