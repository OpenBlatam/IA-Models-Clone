import json as _json
import re
import logging
from typing import Tuple, Optional
from agents.models import AgentAction

logger = logging.getLogger(__name__)

def _extract_json_block(text: str) -> str:
    """Extrae el bloque JSON más externo del texto, ignorando <think> u otro markdown."""
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return text[start_idx:end_idx+1]
    return text

def _recover_fallback(json_str: str, original_err: Exception) -> AgentAction:
    """Usa expresiones regulares seguras para recuperar datos si Pydantic falla."""
    fa_match = re.search(r'"final_answer"\s*:\s*"(.*?)"?(?:\s*\}|\s*,)', json_str, re.DOTALL)
    thought_match = re.search(r'"thought"\s*:\s*"(.*?)"(?:\s*,)', json_str, re.DOTALL)
    tool_match = re.search(r'"tool"\s*:\s*"(\w+)"', json_str)
    
    thought_val = thought_match.group(1) if thought_match else None

    # Caso 1: Se encontró final_answer útil
    if fa_match and len(fa_match.group(1)) > 10:
        recovered = fa_match.group(1).replace('\\n', '\n').replace('\\"', '"')
        logger.info(f"Recuperado final_answer truncado ({len(recovered)} chars)")
        return AgentAction(final_answer=recovered, thought=thought_val)
        
    # Caso 2: Se encontró una llamada a tool
    elif tool_match:
        tool_input_match = re.search(r'"tool_input"\s*:\s*("[^"]*"|\{[^}]*\})', json_str)
        tool_input_val = None
        if tool_input_match:
            try:
                tool_input_val = _json.loads(tool_input_match.group(1))
            except (ValueError, _json.JSONDecodeError):
                pass
        
        reconstructed = {
            "thought": thought_val[:200] if thought_val else "Reconstructed from partial response",
            "tool": tool_match.group(1),
            "tool_input": tool_input_val,
            "final_answer": None
        }
        try:
            return AgentAction.model_validate(reconstructed)
        except Exception:
            raise original_err
            
    # Caso 3: Imposible recuperar, levantar el error original formateado
    else:
        from pydantic import ValidationError
        if isinstance(original_err, ValidationError):
            errors = original_err.errors()
            error_details = "; ".join([f"'{err.get('loc', [''])[0]}': {err.get('msg')}" for err in errors])
            raise ValueError(f"Validación estricta falló: {error_details}")
        raise original_err

def parse_and_recover_action(clean_resp: str) -> Tuple[AgentAction, str]:
    """
    Parses LLM JSON output.
    Returns the parsed AgentAction and the stripped original response.
    """
    clean_resp_stripped = clean_resp.strip()
    json_str = _extract_json_block(clean_resp_stripped)

    try:
        action = AgentAction.model_validate_json(json_str)
        return action, clean_resp_stripped
    except Exception as e:
        # Fallback a recuperación por Regex
        recovered_action = _recover_fallback(json_str, e)
        return recovered_action, clean_resp_stripped
