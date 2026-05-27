import logging
from typing import Dict, Any

from agents.razonamiento_planificacion.config import settings
from agents.models import AgentAction

logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self, agent_instance):
        """
        Takes the agent instance to access its tools, memory, and SOTA module flags.
        """
        self.agent = agent_instance
        
        # Cache imports and base templates to avoid overhead in the hot loop
        from agents.prompts.prompt_manager import prompt_manager
        self.prompt_manager = prompt_manager
        
        self.base_prompt = prompt_manager.get_prompt("base_agent", name=settings.AGENT_NAME, role="Enterprise AI Assistant")
        self.react_prompt = prompt_manager.get_prompt("react_core")
        self.json_schema_prompt = prompt_manager.get_prompt("json_output", schema=AgentAction.model_json_schema())
        
        self.internet_override = (
            "[CRITICAL OVERRIDE]: TIENES ACCESO TOTAL A INTERNET a través de la herramienta 'web_search'.\n"
            "Si un usuario te pide buscar algo reciente (ej. 'papers of this week', 'noticias', 'tendencias'), "
            "¡DEBES USAR LA HERRAMIENTA web_search!\n"
            "Bajo NINGUNA circunstancia respondas 'I cannot access the internet' o 'no tengo acceso a internet'.\n"
        )
        
        self._sota_cache = self._build_sota_cache()

    def _build_sota_cache(self) -> str:
        sota_parts = []
        
        if settings.USE_CHAIN_OF_DRAFT and getattr(self.agent, 'chain_of_draft', None):
            cod_template = self.agent.chain_of_draft.get_template(settings.CHAIN_OF_DRAFT_VARIANT)
            sota_parts.append(
                "[Paper 2506.10987v1 - Chain of Draft]\n"
                "Antes de responder, genera pasos de razonamiento concisos (≤5 palabras por paso):\n"
                f"{cod_template}\n"
            )
            
        if settings.USE_ELASTIC_REASONING and getattr(self.agent, 'elastic_reasoning', None):
            sota_parts.append(
                "[Paper 2505.05315v2 - Elastic Reasoning]\n"
                f"Tu razonamiento tiene un presupuesto de {settings.THINK_BUDGET} tokens para <think> "
                f"y {settings.SOLUTION_BUDGET} tokens para la solución. "
                "Usa <think>...</think> para tu razonamiento interno, luego da la solución "
                "dentro del presupuesto asignado.\n"
            )
            
        if settings.USE_SELF_CONSISTENCY and getattr(self.agent, 'self_consistency', None):
            sota_parts.append(
                "[Paper 2203.11171 - Self-Consistency]\n"
                "Para preguntas de razonamiento complejo, se muestrearán múltiples "
                "cadenas de pensamiento y se seleccionará la respuesta por voto mayoritario. "
                "Muestra tu razonamiento paso a paso.\n"
            )
            
        if settings.USE_MCTS_REASONING and getattr(self.agent, 'mcts_reasoner', None):
            sota_parts.append(
                "[Paper 2305.20050 - MCTS Reasoning]\n"
                "Para problemas complejos, se realizará una búsqueda en árbol MCTS. "
                "Genera pasos de razonamiento concretos y verificables.\n"
            )
            
        if getattr(self.agent, 'custom_system_instructions', None):
            sota_parts.append(self.agent.custom_system_instructions)
            
        return "\n".join(sota_parts) + "\n\n" if sota_parts else ""

    def _get_system_instructions(self) -> str:
        """Pipeline paso 1: Instrucciones principales del sistema."""
        tools_list = "\n".join([f"- {t.name}: {t.description}" for t in self.agent.tools.values()])
        
        parts = [
            self.base_prompt,
            self.internet_override,
            self.react_prompt,
            f"Tienes acceso a estas herramientas:\n{tools_list}\n",
            self._sota_cache,
            self.json_schema_prompt
        ]
        
        return "\n\n".join(parts)

    async def _get_memory_context(self, user_id: str) -> str:
        """Pipeline paso 2: Contexto de memoria a corto y medio plazo."""
        history = await self.agent.memory.get_history(user_id, limit=10)
        core_context = await self.agent.core_memory.get_formatted_context(user_id) if hasattr(self.agent, 'core_memory') else ""
        
        formatted = f"{core_context}\n--- MEMORIA PRIVADA ({user_id}) ---\n"
        for msg in history:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
        formatted += "--------------------------------------\n"
        return formatted

    async def _get_semantic_context(self, message: str, user_id: str) -> str:
        """Pipeline paso 3: Memoria vectorial a largo plazo."""
        if hasattr(self.agent, 'vector_memory') and self.agent.vector_memory:
            try:
                return await self.agent.vector_memory.get_context_for_prompt(message, user_id)
            except Exception as e:
                logger.error(f"Error fetching vector memory: {e}")
        return ""

    def _get_budget_constraint(self) -> str:
        """Pipeline paso 4: Reglas dinámicas (Elastic Reasoning)."""
        if settings.USE_ELASTIC_REASONING and getattr(self.agent, 'elastic_reasoning', None):
            return (
                f"<think>Resuelve con un máximo de {settings.THINK_BUDGET} tokens de pensamiento "
                f"y {settings.SOLUTION_BUDGET} tokens de solución.</think>\n"
            )
        return ""

    async def build_initial_prompt(self, objective: str, user_id: str, semantic_context: str = "") -> str:
        sys_instructions = self._get_system_instructions()
        budget = self._get_budget_constraint()
        
        # Try to load CoALA context if available, fallback to legacy
        memory_context = ""
        try:
            from agents.razonamiento_planificacion.coala_memory import CoALAMemoryManager
            coala = CoALAMemoryManager()
            memory_context = coala.get_full_coala_context(semantic_context)
        except ImportError:
            # Fallback to legacy
            memory_context = await self._get_memory_context(user_id)
            if semantic_context:
                memory_context += f"\nContexto Semántico:\n{semantic_context}\n"
        
        prompt = (
            f"{sys_instructions}\n"
            f"{budget}\n"
            f"{memory_context}\n"
            "=========================================\n"
            f"OBJETIVO ACTUAL: {objective}\n"
            "=========================================\n"
            "TRUTHGPT: <think>\n"
            "Analizaré los lóbulos de memoria y el objetivo actual."
        )
        return prompt
