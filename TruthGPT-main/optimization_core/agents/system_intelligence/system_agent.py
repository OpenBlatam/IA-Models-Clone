
import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..arquitecturas_fundamentales.base_agent import BaseAgent
from ..razonamiento_planificacion.orchestrator import MultiUserReActAgent
from ..models import AgentResponse, AgentConfig
from .system_tools import (
    ListPapersTool,
    PaperInfoTool,
    SystemHealthTool,
    RunOptimizationTool,
    ModelInferenceTool,
    ModelTrainTool
)
from ..razonamiento_planificacion.tools import (
    FileReadTool,
    FileWriteTool,
    DirectoryListTool
)

logger = logging.getLogger(__name__)

class SystemAgent(BaseAgent):
    """
    Agente de Inteligencia de Sistema (TruthGPT Administrator).
    Especializado en gestionar la infraestructura, modelos, papers y herramientas de TruthGPT.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_engine: Optional[Any] = None,
    ) -> None:
        super().__init__(
            name="TruthGPTAgent",
            role="Administrador de Sistemas y ML Ops de TruthGPT",
        )

        self._react = MultiUserReActAgent(
            config=config,
            llm_engine=llm_engine,
            name=self.name,
            custom_system_instructions=(
                "Eres el Agente Administrador de TruthGPT (TruthGPTAgent).\n"
                "Tu misión es gestionar el ecosistema de IA, incluyendo la ejecución de modelos locales, "
                "la consulta de artículos de investigación (SOTA), y la supervisión de la salud del sistema.\n"
                "Tienes acceso directo a las funciones del Command Center de TruthGPT.\n"
                "Usa tus herramientas para responder dudas técnicas sobre el sistema o ejecutar tareas de mantenimiento."
            )
        )

        # Register system tools
        self._react.register_tool(ListPapersTool())
        self._react.register_tool(PaperInfoTool())
        self._react.register_tool(SystemHealthTool())
        self._react.register_tool(RunOptimizationTool())
        self._react.register_tool(ModelInferenceTool())
        self._react.register_tool(ModelTrainTool())
        
        # Add core system interaction tools
        self._react.register_tool(FileReadTool())
        self._react.register_tool(FileWriteTool())
        self._react.register_tool(DirectoryListTool())

    async def process(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Procesa la petición del sistema usando el bucle ReAct."""
        user_id = (context or {}).get("user_id", "sys_admin")
        logger.info("[%s] System request: %s", self.name, query[:80])
        start = time.monotonic()

        try:
            response = await self._react.process_message(user_id, query)
            self.add_to_memory("user", query)
            
            content = response.content if hasattr(response, 'content') else str(response)
            self.add_to_memory("assistant", content)

            latency_ms = (time.monotonic() - start) * 1000
            
            prompt_tokens = getattr(response, 'metadata', {}).get("prompt_tokens", 0)
            completion_tokens = getattr(response, 'metadata', {}).get("completion_tokens", 0)
            
            response.metadata.update({
                "agent": self.name,
                "latency_ms": round(latency_ms, 2),
                "system_ops": True,
                "telemetry": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            })
            
            # Explicit garbage collection to prevent memory leaks in long-running terminal sessions
            import gc
            if hasattr(self, 'memory') and len(self.memory) > 100:
                self.memory = self.memory[-50:]
                gc.collect()
                
            return response
        except Exception as e:
            import json
            logger.error("[%s] Error: %s", self.name, e, exc_info=True)
            error_payload = {
                "error": str(e),
                "context": "Execution failed during ReAct loop.",
                "suggestion": "Check inputs and tool parameters for hallucinations."
            }
            return AgentResponse(
                content=json.dumps(error_payload),
                action_type="final_answer",
            )
