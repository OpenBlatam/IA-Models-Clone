"""
Agent Service - Servicio de orquestación de agentes
"""

from typing import Any, Dict, Optional
from .base import BaseAgent
from .orchestrator import AgentOrchestrator
from .registry import AgentRegistry
from llm.service import LLMService
from tools.service import ToolService
from prompts.service import PromptService
from tracing.service import TracingService


class AgentService:
    """Servicio para gestionar agentes"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        tool_service: Optional[ToolService] = None,
        prompt_service: Optional[PromptService] = None,
        tracing_service: Optional[TracingService] = None
    ):
        """Inicializa el servicio de agentes"""
        self.registry = AgentRegistry()
        self.orchestrator = AgentOrchestrator(
            self.registry,
            llm_service,
            tool_service,
            prompt_service,
            tracing_service
        )

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Registra un agente"""
        self.registry.register(name, agent)

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una tarea usando agentes"""
        return await self.orchestrator.execute(task)

