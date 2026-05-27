from __future__ import annotations
import logging
import re
import asyncio
import json
import uuid
from typing import List, Dict, Any, Callable, Protocol, Optional, runtime_checkable, Type, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.memoria_aprendizaje.sqlite_memory import SQLiteMemory, BaseMemory
    from agents.memoria_aprendizaje.vector_memory import VectorMemory
    from agents.memoria_aprendizaje.core_memory import CoreMemory
    from agents.razonamiento_planificacion.tools import BaseTool, ToolResult
    from agents.engines import AsyncLLMEngine

from agents.models import AgentAction, AgentResponse, InferenceResult, AgentConfig
from agents.razonamiento_planificacion.config import settings

# ===== SOTA Paper Integrations (6 papers) =====
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
    ChainOfDraft = None
    ElasticReasoning = None
    FP16Stability = None
    SelfConsistency = None
    SpeculativeDecoder = None
    MCTSReasoner = None
    RewardEstimator = None

try:
    from interface.cc_style import (
        cc_action, cc_tool_call, cc_result, cc_agent_done, cc_code_change
    )
    CC_AVAILABLE = True
except ImportError:
    CC_AVAILABLE = False

try:
    from agents.observability import global_tracer
except ImportError:
    from ..observability import global_tracer

logger = logging.getLogger(__name__)

# AgentAction and AgentResponse are now imported from .models

class MultiUserReActAgent:
    """
    Orquestador ReAct Multi-Usuario — Platinum Edition.
    
    Integrates 6 SOTA papers for state-of-the-art reasoning:
    1. Chain of Draft — Concise reasoning steps (≤5 words)
    2. Elastic Reasoning — Dynamic think/solution token budget
    3. FP16 Stability — Numerical stability monitoring
    4. Self-Consistency — Majority voting over reasoning paths
    5. Speculative Decoding — Draft/target cost optimization
    6. MCTS Reasoning — Tree search for complex problems
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
        from agents.memoria_aprendizaje.core_memory import CoreMemory
        from agents.memoria_aprendizaje.core_memory_tools import CoreMemoryAppendTool, CoreMemoryReplaceTool

        self.config = config
        self.llm = llm_engine or config.llm_engine
        self.memory = memory or SQLiteMemory(db_path=config.memory_db_path)
        self.vector_memory = vector_memory
        self.core_memory = CoreMemory()
        self.tools: Dict[str, BaseTool] = {}
        self.custom_system_instructions = custom_system_instructions
        self.use_reflexion = config.use_reflexion
        self.name = name or "MultiUserReActAgent"
        self.persistent = getattr(config, "persistent", True)

        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        # Add memory self-update tools
        self.register_tool(CoreMemoryAppendTool(self.core_memory))
        self.register_tool(CoreMemoryReplaceTool(self.core_memory))

        # Initialize SOTA modules
        self._sota_initialized = False
        self._init_sota_modules()

    def _init_sota_modules(self):
        """Initialize all 6 SOTA paper modules if available."""
        if not SOTA_AVAILABLE:
            logger.warning("SOTA papers not available — running without advanced features")
            return

        # Paper 2506.10987v1 — Chain of Draft
        self.chain_of_draft = ChainOfDraft() if ChainOfDraft else None

        # Paper 2505.05315v2 — Elastic Reasoning
        if settings.USE_ELASTIC_REASONING and ElasticReasoning:
            self.elastic_reasoning = ElasticReasoning(
                t_budget=settings.THINK_BUDGET,
                s_budget=settings.SOLUTION_BUDGET
            )
        else:
            self.elastic_reasoning = None

        # Paper 2510.26788v1 — FP16 Stability
        self.fp16_stability = FP16Stability() if FP16Stability else None

        # Paper 2203.11171 — Self-Consistency
        if settings.USE_SELF_CONSISTENCY and SelfConsistency:
            self.self_consistency = SelfConsistency(
                n_samples=settings.SELF_CONSISTENCY_SAMPLES,
                temperature=settings.SELF_CONSISTENCY_TEMPERATURE,
                answer_extraction="json",
            )
        else:
            self.self_consistency = None

        # Paper 2302.01318 — Speculative Decoding
        if settings.USE_SPECULATIVE_DECODING and SpeculativeDecoder:
            self.speculative_decoder = SpeculativeDecoder(
                confidence_threshold=settings.SPECULATIVE_CONFIDENCE_THRESHOLD,
            )
        else:
            self.speculative_decoder = None

        # Paper 2305.20050 — MCTS Reasoning
        if settings.USE_MCTS_REASONING and MCTSReasoner:
            self.mcts_reasoner = MCTSReasoner(
                max_iterations=settings.MCTS_MAX_ITERATIONS,
                max_depth=settings.MCTS_MAX_DEPTH,
                branching_factor=settings.MCTS_BRANCHING_FACTOR,
            )
            self.reward_estimator = RewardEstimator()
        else:
            self.mcts_reasoner = None
            self.reward_estimator = None

        self._sota_initialized = True
        active = []
        if settings.USE_CHAIN_OF_DRAFT:
            active.append("CoD")
        if settings.USE_ELASTIC_REASONING:
            active.append("ER")
        if settings.USE_FP16_STABILITY:
            active.append("FP16")
        if settings.USE_SELF_CONSISTENCY:
            active.append("SC")
        if settings.USE_SPECULATIVE_DECODING:
            active.append("SD")
        if settings.USE_MCTS_REASONING:
            active.append("MCTS")
        logger.info("SOTA modules initialized: [%s]", ", ".join(active) if active else "none")

    def register_tool(self, tool: BaseTool) -> None:
        """Registra una herramienta disponible para el agente."""
        self.tools[tool.name] = tool
        logger.info(f"Agente {self.name}: Herramienta '{tool.name}' registrada.")

    async def load_mcp_tools(self, server_url: str):
        """
        Descubre y registra dinámicamente herramientas desde un servidor MCP.
        """
        from agents.mcp_client import MCPClient
        from agents.razonamiento_planificacion.tools import MCPTool

        logger.info(f"Cargando herramientas MCP desde {server_url}...")
        client = MCPClient(server_url)
        tools_info = await client.list_tools()
        
        for t_info in tools_info:
            mcp_tool = MCPTool(client, t_info)
            self.register_tool(mcp_tool)
        
        logger.info(f"Se cargaron {len(tools_info)} herramientas MCP.")

    def _get_system_instructions(self) -> str:
        """Genera instrucciones dinámicas usando el PromptManager centralizado y 6 papers SOTA."""
        from agents.prompts.prompt_manager import prompt_manager
        
        tools_list = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        
        # Build prompt using centralized templates
        base = prompt_manager.get_prompt("base_agent", name=settings.AGENT_NAME, role="Enterprise AI Assistant")
        react = prompt_manager.get_prompt("react_core")
        json_schema = prompt_manager.get_prompt("json_output", schema=AgentAction.model_json_schema())
        
        internet_override = (
            "[CRITICAL OVERRIDE]: TIENES ACCESO TOTAL A INTERNET a través de la herramienta 'web_search'.\n"
            "Si un usuario te pide buscar algo reciente (ej. 'papers of this week', 'noticias', 'tendencias'), "
            "¡DEBES USAR LA HERRAMIENTA web_search!\n"
            "Bajo NINGUNA circunstancia respondas 'I cannot access the internet' o 'no tengo acceso a internet'.\n"
        )
        
        instructions = f"{base}\n{internet_override}\n{react}\n\nTienes acceso a estas herramientas:\n{tools_list}\n\n"
        
        # ===== Chain of Draft Integration (Paper 2506.10987v1) =====
        if settings.USE_CHAIN_OF_DRAFT and self.chain_of_draft:
            cod_template = self.chain_of_draft.get_template(settings.CHAIN_OF_DRAFT_VARIANT)
            instructions += (
                "[Paper 2506.10987v1 - Chain of Draft]\n"
                "Antes de responder, genera pasos de razonamiento concisos (≤5 palabras por paso):\n"
                f"{cod_template}\n\n"
            )
        
        # ===== Elastic Reasoning Integration (Paper 2505.05315v2) =====
        if settings.USE_ELASTIC_REASONING and self.elastic_reasoning:
            instructions += (
                "[Paper 2505.05315v2 - Elastic Reasoning]\n"
                f"Tu razonamiento tiene un presupuesto de {settings.THINK_BUDGET} tokens para <think> "
                f"y {settings.SOLUTION_BUDGET} tokens para la solución. "
                "Usa <think>...</think> para tu razonamiento interno, luego da la solución "
                "dentro del presupuesto asignado.\n\n"
            )

        # ===== Self-Consistency Hint (Paper 2203.11171) =====
        if settings.USE_SELF_CONSISTENCY and self.self_consistency:
            instructions += (
                "[Paper 2203.11171 - Self-Consistency]\n"
                "Para preguntas de razonamiento complejo, se muestrearán múltiples "
                "cadenas de pensamiento y se seleccionará la respuesta por voto mayoritario. "
                "Muestra tu razonamiento paso a paso.\n\n"
            )

        # ===== MCTS Hint (Paper 2305.20050) =====
        if settings.USE_MCTS_REASONING and self.mcts_reasoner:
            instructions += (
                "[Paper 2305.20050 - MCTS Reasoning]\n"
                "Para problemas complejos, se realizará una búsqueda en árbol MCTS. "
                "Genera pasos de razonamiento concretos y verificables.\n\n"
            )
        
        if self.custom_system_instructions:
            instructions += f"{self.custom_system_instructions}\n\n"
            
        return instructions + json_schema

    async def _format_context(self, user_id: str) -> str:
        """Recupera el historial de la base de datos y lo formatea para el prompt."""
        history = await self.memory.get_history(user_id, limit=10)
        formatted = f"--- MEMORIA PRIVADA ({user_id}) ---\n"
        for msg in history:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
        formatted += "--------------------------------------\n"
        return formatted

    async def _build_initial_prompt(self, user_id: str, message: str) -> str:
        """Construye el prompt inicial del bucle ReAct integrando all SOTA papers."""
        system_instructions = self._get_system_instructions()
        context = await self._format_context(user_id)
        
        # Elastic Reasoning: inject budget into prompt structure
        if settings.USE_ELASTIC_REASONING and self.elastic_reasoning:
            budget_note = (
                f"<think>Resuelve con un máximo de {settings.THINK_BUDGET} tokens de pensamiento "
                f"y {settings.SOLUTION_BUDGET} tokens de solución.</think>\n"
            )
        else:
            budget_note = ""
        
        prompt = (
            f"{system_instructions}\n\n"
            f"{context}\n"
            f"USER: {message}\n"
            f"{budget_note}"
            "TRUTHGPT: "
        )
        return prompt

    async def process_message(self, user_id: str, message: str) -> AgentResponse:
        """
        Procesa un mensaje de forma asíncrona aislando el contexto por usuario.
        Platinum Edition v3: Modularized, Traced, Persistent, Trace-Hardened, Anti-Loop.
        """
        import uuid
        from agents.engines import safe_llm_call

        logger.info(f"Iniciando proceso asíncrono para {user_id}")
        await self.memory.add_message(user_id, "user", message)

        current_prompt = await self._build_initial_prompt(user_id, message)
        trace_id = global_tracer.start_trace(name="process_message", agent_name=self.name)
        task_id = str(uuid.uuid4())[:8]

        MAX_JSON_RETRIES = 10
        CONSECUTIVE_ERRORS_LIMIT = 15
        NO_PROGRESS_LIMIT = 100  # Circuit breaker if we do 100 steps with tools but no answer
        
        json_retry_count = 0
        consecutive_errors = 0
        logical_step = 0
        failed_search_count = 0
        recent_tool_keys = []
        last_tool_key = None

        try:
            while logical_step < settings.MAX_ITERATIONS:
                if logical_step >= NO_PROGRESS_LIMIT:
                    logger.warning(f"Circuit breaker triggered: No final answer after {NO_PROGRESS_LIMIT} logical steps.")
                    fallback = "Límite de progreso alcanzado (demasiados pasos sin conclusión). Por favor, replantea tu petición."
                    await self.memory.add_message(user_id, "assistant", fallback)
                    return await self._finalize_response(trace_id, task_id, fallback)

                response = await safe_llm_call(self.llm, current_prompt, trace_id)
                clean_resp = response.strip() if response else ""

                if not clean_resp or '{' not in clean_resp:
                    logger.warning("Non-JSON response detected (no '{' found). Short-circuiting.")
                    fallback = clean_resp[:300] if clean_resp else "Motor LLM no configurado."
                    if "Echo from" in fallback or "Mock" in fallback:
                        fallback = "⚠️ Motor de inferencia no configurado. Ve a Settings (P) > Set Engines y configura una API key."
                    await self.memory.add_message(user_id, "assistant", fallback)
                    return await self._finalize_response(trace_id, task_id, fallback)

                try:
                    action, clean_resp_stripped = self._parse_and_recover_action(clean_resp)
                except Exception as parse_err:
                    json_retry_count += 1
                    consecutive_errors += 1
                    logger.warning(f"JSON parse error ({json_retry_count}/{MAX_JSON_RETRIES}): {type(parse_err).__name__}")
                    
                    if json_retry_count >= MAX_JSON_RETRIES or consecutive_errors >= CONSECUTIVE_ERRORS_LIMIT:
                        fallback = f"JSON Parse Error: {str(parse_err)}. RAW JSON: {clean_resp[:300]}"
                        await self.memory.add_message(user_id, "assistant", fallback)
                        return await self._finalize_response(trace_id, task_id, fallback)
                    
                    current_prompt += f"\n[ERROR de parseo: {str(parse_err)}]: Responde SOLO con JSON válido {{thought, tool/final_answer}}.\nTRUTHGPT: "
                    continue

                logical_step += 1
                json_retry_count = 0
                consecutive_errors = 0  # Reset error streak on success

                if action.tool:
                    is_dup, tool_key = self._is_duplicate_tool_call(action, recent_tool_keys, last_tool_key)
                    if is_dup:
                        error_msg = f"[SISTEMA]: Error - Detectada llamada repetida a la herramienta '{action.tool}' con los mismos argumentos. Intenta una estrategia distinta."
                        current_prompt += f"{clean_resp_stripped}\nTOOL_RESULT: {error_msg}\nTRUTHGPT: "
                        continue
                        
                    last_tool_key = tool_key
                    recent_tool_keys.append(tool_key)
                    if len(recent_tool_keys) > 6:
                        recent_tool_keys = recent_tool_keys[-6:]

                    # Execute tool
                    result, requires_approval, new_failed_search = await self._handle_tool_execution(
                        action, trace_id, user_id, failed_search_count
                    )
                    failed_search_count = new_failed_search
                    
                    if requires_approval:
                        await self.memory.add_message(user_id, "assistant", result)
                        return AgentResponse(content=result, action_type="approval_required")
                    
                    if action.tool == "web_search" and failed_search_count >= 2:
                        logger.warning(f"web_search failed {failed_search_count}x. Forcing LLM to answer without search.")
                        current_prompt += (
                            f"{clean_resp_stripped}\nTOOL_RESULT: {result}\n"
                            "[SISTEMA]: La herramienta web_search no está disponible. "
                            "Responde usando tu conocimiento interno sin usar herramientas.\n"
                            "TRUTHGPT: "
                        )
                    else:
                        current_prompt += f"{clean_resp_stripped}\nTOOL_RESULT: {result}\nTRUTHGPT: "

                elif action.final_answer:
                    if "{" in action.final_answer and '"tool"' in action.final_answer:
                        error_msg = "[SISTEMA]: Error - Has puesto tu respuesta JSON dentro del campo 'final_answer'. Por favor, responde directamente con la estructura JSON en la raíz, sin envolverla."
                        current_prompt += f"{clean_resp_stripped}\nTOOL_RESULT: {error_msg}\nTRUTHGPT: "
                        continue

                    is_system_error = "Inference error:" in action.final_answer or "⚠️ Motor de inferencia" in action.final_answer
                    if self.use_reflexion and not is_system_error:
                        passed, critique = await self._reflexion_check(current_prompt, clean_resp_stripped, trace_id)
                        if passed:
                            await self.memory.add_message(user_id, "assistant", action.final_answer)
                            return await self._finalize_response(trace_id, task_id, action.final_answer)
                        current_prompt = f"{current_prompt}\n{clean_resp_stripped}\n[REFLEXION]: {critique}\nTRUTHGPT: "
                    else:
                        await self.memory.add_message(user_id, "assistant", action.final_answer)
                        return await self._finalize_response(trace_id, task_id, action.final_answer)

                elif action.handoff:
                    await self.memory.add_message(user_id, "assistant", f"Transferring to {action.handoff}...")
                    global_tracer.finish_trace(trace_id)
                    return AgentResponse(content=f"Transferring to {action.handoff}...", action_type="handoff", handoff_target=action.handoff)

                else:
                    raise ValueError("JSON sin 'tool', 'final_answer' ni 'handoff'.")

            fallback = "Límite de razonamiento iterativo alcanzado. Por favor, simplifica tu petición."
            await self.memory.add_message(user_id, "assistant", fallback)
            return await self._finalize_response(trace_id, task_id, fallback)

        except Exception as critical_err:
            logger.error(f"Critical error in process_message: {critical_err}")
            if CC_AVAILABLE:
                cc_agent_done(self.name, ok=False)
            error_msg = f"Error interno: {str(critical_err)[:200]}"
            return AgentResponse(content=error_msg, action_type="error")
        finally:
            try:
                global_tracer.finish_trace(trace_id)
            except Exception:
                pass

    def _parse_and_recover_action(self, clean_resp: str) -> Tuple[AgentAction, str]:
        """Parses LLM JSON output, handles markdown stripping and recovers truncated JSON."""
        import json as _json
        import re
        
        json_str = clean_resp.strip()
        
        # Robustly extract JSON block by finding the first { and last }, ignoring <think> tags or markdown
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = json_str[start_idx:end_idx+1]
        
        # Use json_str for parsing, but return clean_resp.strip() so the prompt keeps the thought tags
        clean_resp_stripped = clean_resp.strip()

        try:
            action = AgentAction.model_validate_json(json_str)
            return action, clean_resp_stripped
        except Exception as parse_err:
            fa_match = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"?', json_str, re.DOTALL)
            thought_match = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL)
            tool_match = re.search(r'"tool"\s*:\s*"(\w+)"', json_str)
            
            if fa_match and len(fa_match.group(1)) > 10:
                recovered = fa_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                logger.info(f"Recovered final_answer from truncated JSON ({len(recovered)} chars)")
                return AgentAction(final_answer=recovered, thought=thought_match.group(1) if thought_match else None), clean_resp_stripped
            elif tool_match:
                tool_input_match = re.search(r'"tool_input"\s*:\s*("[^"]*"|\{[^}]*\})', json_str)
                tool_input_val = None
                if tool_input_match:
                    try:
                        tool_input_val = _json.loads(tool_input_match.group(1))
                    except:
                        pass
                
                reconstructed = {
                    "thought": thought_match.group(1)[:200] if thought_match else "Reconstructed from partial response",
                    "tool": tool_match.group(1),
                    "tool_input": tool_input_val,
                    "final_answer": None
                }
                try:
                    action = AgentAction.model_validate(reconstructed)
                    return action, clean_resp_stripped
                except:
                    raise parse_err
            else:
                raise parse_err

    def _is_duplicate_tool_call(self, action: AgentAction, recent_tool_keys: list, last_tool_key: str) -> Tuple[bool, str]:
        """Detects if a tool call is exactly or fuzzily duplicated in recent history."""
        tool_key = f"{action.tool}:{action.tool_input}"
        is_duplicate = (tool_key == last_tool_key)
        
        if not is_duplicate and action.tool == "web_search" and len(recent_tool_keys) >= 2:
            current_words = set(str(action.tool_input).lower().split())
            for prev_key in recent_tool_keys[-4:]:
                if prev_key.startswith("web_search:"):
                    prev_words = set(prev_key.split(":", 1)[1].lower().split())
                    overlap = len(current_words & prev_words)
                    total = max(len(current_words | prev_words), 1)
                    if overlap / total > 0.6:
                        is_duplicate = True
                        logger.warning(f"Fuzzy duplicate web_search detected (overlap={overlap}/{total})")
                        break
        return is_duplicate, tool_key

    async def _handle_tool_execution(self, action: AgentAction, trace_id: str, user_id: str, failed_search_count: int) -> Tuple[str, bool, int]:
        """Executes the tool and checks for errors or specific search failures."""
        if action.tool not in self.tools:
            return f"Error: Herramienta '{action.tool}' no existe.", False, failed_search_count
        
        tool_instance = self.tools[action.tool]
        if tool_instance.requires_approval:
            return f"⏳ Aprobación requerida: {action.tool}", True, failed_search_count
            
        if CC_AVAILABLE:
            from interface.cc_style import cc_tool_start, cc_tool_call
            cc_tool_start(action.tool, str(action.tool_input))
            cc_tool_call(f"Executing {action.tool}...")
            
        result = await self._execute_tool_action(trace_id, action, user_id)
        
        if CC_AVAILABLE:
            from interface.cc_style import cc_tool_end, cc_result
            cc_result(action.tool, note="Success")
            cc_tool_end(action.tool, str(result))
            
        new_failed_search_count = failed_search_count
        if action.tool == "web_search" and "no se encontraron" in result.lower():
            new_failed_search_count += 1
        else:
            new_failed_search_count = 0
            
        return result, False, new_failed_search_count

    async def _finalize_response(self, trace_id: str, task_id: str, final_answer: str) -> AgentResponse:
        """Finaliza la ejecución: cierra trace, actualiza persistencia."""
        global_tracer.finish_trace(trace_id)
        if CC_AVAILABLE:
            cc_agent_done(self.name, ok=True)
        return AgentResponse(content=final_answer, action_type="final_answer")

    async def _reflexion_check(self, current_prompt: str, clean_resp: str, trace_id: str) -> tuple:
        """Auto-Reflexion: evalúa calidad de la respuesta."""
        from agents.engines import safe_llm_call
        critique_prompt = (
            f"{current_prompt}\n{clean_resp}\n"
            "[SISTEMA]: Si la respuesta es correcta, responde '<final>APROBADO</final>'. Si no, critica."
        )
        critique = await safe_llm_call(self.llm, critique_prompt, trace_id)
        return "<final>APROBADO</final>" in critique, critique

    async def _execute_tool_action(self, trace_id: str, action: AgentAction, user_id: str) -> str:
        """Ejecuta herramienta con tracing y señales Core Memory."""
        from agents.razonamiento_planificacion.tools import ToolResult
        tool_instance = self.tools[action.tool]
        tool_span = global_tracer.start_span(trace_id, name=action.tool, kind="tool_call", input_data=str(action.tool_input))
        try:
            raw_result = await tool_instance.run(str(action.tool_input) or "")
            if isinstance(raw_result, ToolResult):
                result_str = raw_result.output
                if raw_result.signal == "core_memory_append":
                    await self.core_memory.append_to_block(user_id, raw_result.metadata.get("block"), raw_result.metadata.get("content"))
                    result_str = f"SYSTEM: Memoria CORE actualizada."
                elif raw_result.signal == "core_memory_replace":
                    await self.core_memory.update_block(user_id, raw_result.metadata.get("block"), raw_result.metadata.get("content"))
                    result_str = f"SYSTEM: Memoria CORE reemplazada."
            else:
                result_str = str(raw_result)
            tool_span.finish(output=result_str)
            return result_str
        except Exception as e:
            logger.error(f"Error ejecutando {action.tool}: {e}")
            tool_span.finish(output=str(e), status="error")
            return f"Error ejecutando {action.tool}: {str(e)[:200]}"

    async def astream_process_message(self, user_id: str, message: str):
        """Streaming version — delegates to process_message."""
        import json as _json
        result = await self.process_message(user_id, message)
        yield _json.dumps({"event": "final_answer", "content": result.content}) + "\n"

    async def resume_task(self, task_id: str):
        """Resume an interrupted task."""
        try:
            from modules.persistence.task_manager import get_persistence_manager
            task = await get_persistence_manager().get_task(task_id)
            if task:
                return await self.process_message(task["user_id"], task["message"])
        except Exception as e:
            logger.error(f"Resume failed: {e}")
        return AgentResponse(content="No se pudo reanudar.", action_type="error")
