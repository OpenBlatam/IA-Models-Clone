"""
Agent Orchestrator - Orquestador de agentes
"""

from typing import Any, Dict, Optional
from .registry import AgentRegistry
from llm.service import LLMService
from tools.service import ToolService
from prompts.service import PromptService
from tracing.service import TracingService


class AgentOrchestrator:
    """Orquestador de agentes para tareas complejas"""

    def __init__(
        self,
        registry: AgentRegistry,
        llm_service: Optional[LLMService] = None,
        tool_service: Optional[ToolService] = None,
        prompt_service: Optional[PromptService] = None,
        tracing_service: Optional[TracingService] = None
    ):
        """Inicializa el orquestador"""
        self.registry = registry
        self.llm_service = llm_service
        self.tool_service = tool_service
        self.prompt_service = prompt_service
        self.tracing_service = tracing_service

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una tarea orquestando agentes"""
        agent_name = task.get("agent")
        agent = self.registry.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        return await agent.execute(task)

