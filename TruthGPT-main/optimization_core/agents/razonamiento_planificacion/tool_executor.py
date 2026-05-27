import logging
from typing import Tuple, Dict, Any

from agents.models import AgentAction
from agents.observability import global_tracer

logger = logging.getLogger(__name__)

CC_AVAILABLE = False
try:
    from interface import cc_style
    CC_AVAILABLE = True
except ImportError:
    pass

class ToolExecutor:
    """
    Handles the execution lifecycle of tools for the ReAct Agent.
    Includes duplicate detection, fuzzy search protections, tracing, 
    and core memory signal dispatching.
    """
    def __init__(self, tools: Dict[str, Any], core_memory: Any):
        self.tools = tools
        self.core_memory = core_memory
        
        try:
            from .meta_learner import PythonMetaLearner
            self.meta_learner = PythonMetaLearner()
        except ImportError:
            self.meta_learner = None
            logger.warning("No se pudo inicializar PythonMetaLearner")

    async def process_tool_action(
        self, 
        action: AgentAction, 
        clean_resp_stripped: str, 
        current_prompt: str, 
        trace_id: str, 
        user_id: str, 
        failed_search_count: int, 
        recent_tool_keys: list, 
        last_tool_key: str
    ) -> Tuple[str, str | None, str, int, list, str]:
        """
        Main entry point to execute a tool action and process the result.
        Returns: status ("continue" or "approval_required"), result_or_response, new_prompt, failed_search_count, recent_tool_keys, last_tool_key
        """
        is_dup, tool_key = self._is_duplicate_tool_call(action, recent_tool_keys, last_tool_key)
        if is_dup:
            error_msg = f"[SISTEMA]: Error - Detectada llamada repetida a la herramienta '{action.tool}' con los mismos argumentos. Intenta una estrategia distinta."
            return "continue", None, current_prompt + f"{clean_resp_stripped}\nTOOL_RESULT: {error_msg}\nTRUTHGPT: ", failed_search_count, recent_tool_keys, last_tool_key
            
        # Meta-Learning: Circuit Breaker
        if self.meta_learner and self.meta_learner.should_break_circuit(action.tool, action.tool_input):
            error_msg = f"[CRITICAL SYSTEM INTERVENTION]: El Sistema Inmune de Python detectó que la herramienta '{action.tool}' con estos argumentos ha fallado múltiples veces en el pasado. Se ha bloqueado su ejecución para evitar atascos. GENERATE UNA NUEVA ESTRATEGIA INMEDIATAMENTE."
            logger.warning(f"Circuit Breaker activado en Python para {action.tool}")
            return "continue", None, current_prompt + f"{clean_resp_stripped}\nTOOL_RESULT: {error_msg}\nTRUTHGPT: ", failed_search_count, recent_tool_keys, last_tool_key
            
        last_tool_key = tool_key
        recent_tool_keys.append(tool_key)
        if len(recent_tool_keys) > 6:
            recent_tool_keys = recent_tool_keys[-6:]

        result, requires_approval, new_failed_search = await self._handle_tool_execution(
            action, trace_id, user_id, failed_search_count
        )
        
        if requires_approval:
            return "approval_required", result, current_prompt, new_failed_search, recent_tool_keys, last_tool_key
        
        if action.tool == "web_search" and new_failed_search >= 2:
            logger.warning(f"web_search failed {new_failed_search}x. Forcing LLM to answer without search.")
            new_prompt = current_prompt + (
                f"{clean_resp_stripped}\nTOOL_RESULT: {result}\n"
                "[SISTEMA]: La herramienta web_search no está disponible. "
                "Responde usando tu conocimiento interno sin usar herramientas.\n"
                "TRUTHGPT: "
            )
        else:
            new_prompt = current_prompt + f"{clean_resp_stripped}\nTOOL_RESULT: {result}\nTRUTHGPT: "
            
        return "continue", None, new_prompt, new_failed_search, recent_tool_keys, last_tool_key

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
            
            # Registrar éxito en la memoria muscular
            if self.meta_learner and not isinstance(raw_result, ToolResult) or (isinstance(raw_result, ToolResult) and raw_result.success):
                self.meta_learner.record_result(action.tool, str(action.tool_input), success=True)
            elif self.meta_learner and isinstance(raw_result, ToolResult) and not raw_result.success:
                self.meta_learner.record_result(action.tool, str(action.tool_input), success=False)
                
            return result_str
        except Exception as e:
            logger.error(f"Error ejecutando {action.tool}: {e}")
            tool_span.finish(output=str(e), status="error")
            
            # Registrar fracaso en la memoria muscular
            if self.meta_learner:
                self.meta_learner.record_result(action.tool, str(action.tool_input), success=False)
                
            return f"Error ejecutando {action.tool}: {str(e)[:200]}"
