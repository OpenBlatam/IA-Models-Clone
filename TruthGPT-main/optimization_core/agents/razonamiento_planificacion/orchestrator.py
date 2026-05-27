from __future__ import annotations
import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.memoria_aprendizaje.sqlite_memory import SQLiteMemory, BaseMemory
    from agents.memoria_aprendizaje.vector_memory import VectorMemory
    from agents.memoria_aprendizaje.core_memory import CoreMemory
    from agents.razonamiento_planificacion.tools import BaseTool, ToolResult
    from agents.engines import AsyncLLMEngine

from agents.models import AgentAction, AgentResponse, AgentConfig
from agents.razonamiento_planificacion.config import settings
from agents.engines import safe_llm_call
from .prompt_builder import PromptBuilder
from .action_parser import parse_and_recover_action

# ===== Optional Integrations =====
try:
    from papers.chain_of_draft import ChainOfDraft
    from papers.elastic_reasoning import ElasticReasoning
    from papers.fp16_stability import FP16Stability
    from papers.self_consistency import SelfConsistency
    from papers.speculative_decoding import SpeculativeDecoder
    from papers.mcts_reasoning import MCTSReasoner, RewardEstimator
    SOTA_AVAILABLE = True
except ImportError:
    SOTA_AVAILABLE = False
    ChainOfDraft = ElasticReasoning = FP16Stability = SelfConsistency = SpeculativeDecoder = MCTSReasoner = RewardEstimator = None

try:
    from interface.cc_style import cc_agent_done
    CC_AVAILABLE = True
except ImportError:
    CC_AVAILABLE = False

try:
    from agents.observability import global_tracer
except ImportError:
    from ..observability import global_tracer

logger = logging.getLogger(__name__)

class MultiUserReActAgent:
    """
    Orquestador ReAct Multi-Usuario — Platinum Edition.
    """
    def __init__(
        self, 
        config: AgentConfig,
        llm_engine: Optional[AsyncLLMEngine] = None, 
        memory: Optional[BaseMemory] = None,
        vector_memory: Optional[VectorMemory] = None,
        custom_system_instructions: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        name: Optional[str] = None
    ):
        from agents.memoria_aprendizaje.sqlite_memory import SQLiteMemory
        from agents.memoria_aprendizaje.l1_l2_memory import L1L2TieredMemory
        from agents.memoria_aprendizaje.core_memory import CoreMemory
        from agents.memoria_aprendizaje.core_memory_tools import CoreMemoryAppendTool, CoreMemoryReplaceTool

        self.config = config
        self.llm = llm_engine or config.llm_engine
        self.memory = memory or L1L2TieredMemory(db_path=config.memory_db_path)
        self.vector_memory = vector_memory
        self.core_memory = CoreMemory()
        self.tools: Dict[str, BaseTool] = {}
        self.custom_system_instructions = custom_system_instructions
        self.use_reflexion = config.use_reflexion
        self.name = name or "MultiUserReActAgent"
        self.persistent = getattr(config, "persistent", True)

        # Base tools
        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        # Memory self-update tools
        self.register_tool(CoreMemoryAppendTool(self.core_memory))
        self.register_tool(CoreMemoryReplaceTool(self.core_memory))

        # Nexus OS Kernel Tool
        try:
            from agents.razonamiento_planificacion.tools.nexus import NexusTool
            self.register_tool(NexusTool())
        except ImportError:
            logger.warning("NexusTool no pudo cargarse.")

        # SOTA and Execution Initialization
        self._sota_initialized = False
        self._init_sota_modules()
        
        self.prompt_builder = PromptBuilder(self)
        from .tool_executor import ToolExecutor
        self.tool_executor = ToolExecutor(self.tools, self.core_memory)

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool
        logger.info(f"Agente {self.name}: Herramienta '{tool.name}' registrada.")

    async def load_mcp_tools(self, server_url: str):
        from agents.mcp_client import MCPClient
        from agents.razonamiento_planificacion.tools.mcp import MCPTool

        logger.info(f"Cargando herramientas MCP desde {server_url}...")
        client = MCPClient(server_url)
        tools_info = await client.list_tools()
        for t_info in tools_info:
            self.register_tool(MCPTool(client, t_info))
        logger.info(f"Se cargaron {len(tools_info)} herramientas MCP.")

    def _init_sota_modules(self):
        """Inicializa los módulos SOTA (State Of The Art) de manera simplificada."""
        if not SOTA_AVAILABLE:
            logger.warning("SOTA papers not available — running without advanced features")
            return

        self.chain_of_draft = ChainOfDraft() if ChainOfDraft else None
        self.fp16_stability = FP16Stability() if FP16Stability else None
        
        self.elastic_reasoning = ElasticReasoning(
            t_budget=settings.THINK_BUDGET, s_budget=settings.SOLUTION_BUDGET
        ) if settings.USE_ELASTIC_REASONING and ElasticReasoning else None

        self.self_consistency = SelfConsistency(
            n_samples=settings.SELF_CONSISTENCY_SAMPLES,
            temperature=settings.SELF_CONSISTENCY_TEMPERATURE,
            answer_extraction="json"
        ) if settings.USE_SELF_CONSISTENCY and SelfConsistency else None

        self.speculative_decoder = SpeculativeDecoder(
            confidence_threshold=settings.SPECULATIVE_CONFIDENCE_THRESHOLD
        ) if settings.USE_SPECULATIVE_DECODING and SpeculativeDecoder else None

        if settings.USE_MCTS_REASONING and MCTSReasoner:
            self.mcts_reasoner = MCTSReasoner(
                max_iterations=settings.MCTS_MAX_ITERATIONS,
                max_depth=settings.MCTS_MAX_DEPTH,
                branching_factor=settings.MCTS_BRANCHING_FACTOR
            )
            self.reward_estimator = RewardEstimator()
        else:
            self.mcts_reasoner = None
            self.reward_estimator = None

        self._sota_initialized = True
        logger.info("SOTA modules initialized successfully.")

    async def process_message(self, user_id: str, message: str) -> AgentResponse:
        logger.info(f"Iniciando proceso asíncrono para {user_id}")
        await self.memory.add_message(user_id, "user", message)

        current_prompt = await self.prompt_builder.build_initial_prompt(objective=message, user_id=user_id)
        trace_id = global_tracer.start_trace(name="process_message", agent_name=self.name)
        task_id = str(uuid.uuid4())[:8]

        # State tracking for the reasoning loop
        loop_state = {
            "logical_step": 0,
            "json_retry_count": 0,
            "consecutive_errors": 0,
            "failed_search_count": 0,
            "recent_tool_keys": [],
            "last_tool_key": None
        }

        try:
            return await self._reasoning_loop(user_id, current_prompt, trace_id, task_id, loop_state)
        except Exception as critical_err:
            logger.error(f"Critical error in process_message: {critical_err}")
            if CC_AVAILABLE:
                cc_agent_done(self.name, ok=False)
            return AgentResponse(content=f"Error interno: {str(critical_err)[:200]}", action_type="error")
        finally:
            try:
                global_tracer.finish_trace(trace_id)
            except Exception:
                pass

    async def _reasoning_loop(self, user_id: str, current_prompt: str, trace_id: str, task_id: str, state: dict) -> AgentResponse:
        MAX_ITERATIONS = getattr(settings, 'MAX_ITERATIONS', 30)
        NO_PROGRESS_LIMIT = 100
        MAX_JSON_RETRIES = 10
        CONSECUTIVE_ERRORS_LIMIT = 15

        while state["logical_step"] < MAX_ITERATIONS:
            if state["logical_step"] >= NO_PROGRESS_LIMIT:
                return await self._abort_with_message(user_id, trace_id, task_id, "Límite de progreso alcanzado. Por favor, replantea tu petición.")

            # 1. Llamar al LLM
            response = await safe_llm_call(self.llm, current_prompt, trace_id)
            clean_resp = response.strip() if response else ""

            # Validación de integridad
            if not clean_resp or '{' not in clean_resp or "Echo from" in clean_resp or "Mock" in clean_resp:
                fallback = "⚠️ Motor de inferencia no configurado." if "Echo" in clean_resp or "Mock" in clean_resp else clean_resp[:300]
                return await self._abort_with_message(user_id, trace_id, task_id, fallback)

            # 2. Parseo de Acción JSON
            try:
                action, clean_resp_stripped = parse_and_recover_action(clean_resp)
                state["json_retry_count"] = 0
                state["consecutive_errors"] = 0
            except Exception as parse_err:
                state["json_retry_count"] += 1
                state["consecutive_errors"] += 1
                logger.warning(f"JSON parse error: {type(parse_err).__name__}")
                
                if state["json_retry_count"] >= MAX_JSON_RETRIES or state["consecutive_errors"] >= CONSECUTIVE_ERRORS_LIMIT:
                    return await self._abort_with_message(user_id, trace_id, task_id, f"JSON Parse Error: {str(parse_err)}")
                
                current_prompt = self._handle_parse_error(current_prompt, clean_resp, parse_err)
                continue

            state["logical_step"] += 1

            # 3. Enrutamiento de la Acción
            status, current_prompt, final_response = await self._route_action(
                action, clean_resp_stripped, current_prompt, trace_id, task_id, user_id, state
            )

            if status == "done":
                return final_response
            elif status == "approval_required":
                return final_response
            elif status == "continue":
                continue

        return await self._abort_with_message(user_id, trace_id, task_id, "Límite de razonamiento iterativo alcanzado.")

    async def _route_action(self, action: AgentAction, clean_resp: str, current_prompt: str, trace_id: str, task_id: str, user_id: str, state: dict) -> Tuple[str, str, Optional[AgentResponse]]:
        """Dirige el flujo según el tipo de acción decidida por el agente."""
        if action.tool:
            status, result, new_prompt, f_count, r_keys, l_key = await self.tool_executor.process_tool_action(
                action, clean_resp, current_prompt, trace_id, user_id, 
                state["failed_search_count"], state["recent_tool_keys"], state["last_tool_key"]
            )
            state.update({"failed_search_count": f_count, "recent_tool_keys": r_keys, "last_tool_key": l_key})
            
            if status == "continue":
                return "continue", new_prompt, None
            elif status == "approval_required":
                await self.memory.add_message(user_id, "assistant", result)
                return "approval_required", current_prompt, AgentResponse(content=result, action_type="approval_required")

        elif action.final_answer:
            return await self._evaluate_final_answer(action, current_prompt, clean_resp, trace_id, task_id, user_id)

        elif action.handoff:
            await self.memory.add_message(user_id, "assistant", f"Transferring to {action.handoff}...")
            global_tracer.finish_trace(trace_id)
            return "done", current_prompt, AgentResponse(content=f"Transferring to {action.handoff}...", action_type="handoff", handoff_target=action.handoff)

        raise ValueError("JSON sin 'tool', 'final_answer' ni 'handoff'.")

    async def _evaluate_final_answer(self, action: AgentAction, current_prompt: str, clean_resp_stripped: str, trace_id: str, task_id: str, user_id: str) -> Tuple[str, str, Optional[AgentResponse]]:
        if "{" in action.final_answer and '"tool"' in action.final_answer:
            error_msg = "[SISTEMA]: Error - Has puesto tu respuesta JSON dentro del campo 'final_answer'."
            return "continue", current_prompt + f"{clean_resp_stripped}\nTOOL_RESULT: {error_msg}\nTRUTHGPT: ", None

        is_system_error = "Inference error:" in action.final_answer or "⚠️ Motor" in action.final_answer
        if self.use_reflexion and not is_system_error:
            passed, critique = await self._reflexion_check(current_prompt, clean_resp_stripped, trace_id)
            if not passed:
                return "continue", f"{current_prompt}\n{clean_resp_stripped}\n[REFLEXION]: {critique}\nTRUTHGPT: ", None

        await self.memory.add_message(user_id, "assistant", action.final_answer)
        final_resp = await self._finalize_response(trace_id, task_id, action.final_answer)
        return "done", current_prompt, final_resp

    def _handle_parse_error(self, current_prompt: str, clean_resp: str, parse_err: Exception) -> str:
        if current_prompt.endswith("TRUTHGPT: "):
            current_prompt = current_prompt[:-10]
        return current_prompt + f"Tu respuesta anterior fue:\n{clean_resp}\n[ERROR de parseo: {str(parse_err)}]: Responde SOLO con JSON válido.\nTRUTHGPT: "

    async def _abort_with_message(self, user_id: str, trace_id: str, task_id: str, message: str) -> AgentResponse:
        await self.memory.add_message(user_id, "assistant", message)
        return await self._finalize_response(trace_id, task_id, message)

    async def _finalize_response(self, trace_id: str, task_id: str, final_answer: str) -> AgentResponse:
        if CC_AVAILABLE:
            cc_agent_done(self.name, ok=True)
        return AgentResponse(content=final_answer, action_type="final_answer")

    async def _reflexion_check(self, current_prompt: str, clean_resp: str, trace_id: str) -> tuple:
        critique_prompt = f"{current_prompt}\n{clean_resp}\n[SISTEMA]: Si la respuesta es correcta, responde '<final>APROBADO</final>'. Si no, critica."
        critique = await safe_llm_call(self.llm, critique_prompt, trace_id)
        return "<final>APROBADO</final>" in critique, critique

    async def astream_process_message(self, user_id: str, message: str):
        import json as _json
        try:
            result = await self.process_message(user_id, message)
            yield _json.dumps({"event": "final_answer", "content": result.content}) + "\n"
        except Exception as e:
            logger.exception("Error during fake-stream process_message")
            yield _json.dumps({"event": "error", "content": str(e)}) + "\n"

    async def resume_task(self, task_id: str) -> AgentResponse:
        try:
            from modules.persistence.task_manager import get_persistence_manager
            task = await get_persistence_manager().get_task(task_id)
            if task:
                return await self.process_message(task["user_id"], task["message"])
        except Exception as e:
            logger.error(f"Resume failed: {e}")
        return AgentResponse(content="No se pudo reanudar.", action_type="error")
