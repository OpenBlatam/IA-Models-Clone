"""
Utilidades compartidas para el core.
"""

import json
import asyncio
from typing import Dict, Any, Optional, Callable
from functools import wraps
from config.logging_config import get_logger
from core.constants import GitConfig

logger = get_logger(__name__)


def parse_json_field(value: Optional[str]) -> Optional[Any]:
    """
    Parsear un campo JSON de forma segura con mejor manejo de errores.
    
    Args:
        value: String JSON o None
        
    Returns:
        Objeto parseado o None
        
    Raises:
        ValueError: Si el valor no es None y no es un string válido
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        logger.warning(f"Intento de parsear JSON de tipo {type(value).__name__}: {value}")
        return None
    
    if not value.strip():
        return None
    
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        logger.warning(
            f"Error al parsear JSON (línea {e.lineno}, columna {e.colno}): {value[:100]}"
        )
        return None
    except TypeError as e:
        logger.warning(f"Error de tipo al parsear JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado al parsear JSON: {e}", exc_info=True)
        return None


def serialize_json_field(value: Optional[Any]) -> Optional[str]:
    """
    Serializar un objeto a JSON de forma segura con mejor manejo de errores.
    
    Args:
        value: Objeto a serializar
        
    Returns:
        String JSON o None
    """
    if value is None:
        return None
    
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError as e:
        logger.warning(
            f"Error de tipo al serializar JSON (tipo: {type(value).__name__}): {e}"
        )
        # Intentar serializar con default=str para tipos no serializables
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception as e2:
            logger.error(f"Error al serializar JSON con fallback: {e2}", exc_info=True)
            return None
    except ValueError as e:
        logger.warning(f"Error de valor al serializar JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado al serializar JSON: {e}", exc_info=True)
        return None


def handle_github_exception(func: Callable) -> Callable:
    """
    Decorador para manejar excepciones de GitHub de forma consistente.
    Soporta funciones síncronas y asíncronas.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def parse_instruction_params(instruction: str) -> Dict[str, Any]:
    """
    Parsear parámetros de una instrucción de forma más robusta.
    Soporta múltiples formatos de instrucciones.
    
    Args:
        instruction: Instrucción a parsear
        
    Returns:
        Diccionario con parámetros parseados
    """
    if not instruction or not isinstance(instruction, str):
        return {
            "file_path": None,
            "content": "",
            "branch": GitConfig.DEFAULT_BASE_BRANCH,
            "branch_name": None,
            "base_branch": GitConfig.DEFAULT_BASE_BRANCH,
            "title": None,
            "body": instruction or "",
            "head": GitConfig.DEFAULT_BASE_BRANCH,
            "base": GitConfig.DEFAULT_BASE_BRANCH
        }
    
    instruction_lower = instruction.lower()
    params = {
        "file_path": None,
        "content": "",
        "branch": GitConfig.DEFAULT_BASE_BRANCH,
        "branch_name": None,
        "base_branch": GitConfig.DEFAULT_BASE_BRANCH,
        "title": None,
        "body": instruction,
        "head": GitConfig.DEFAULT_BASE_BRANCH,
        "base": GitConfig.DEFAULT_BASE_BRANCH
    }
    
    # Buscar file path con mejor parsing
    file_keywords = ["file", "archivo", "path"]
    for keyword in file_keywords:
        if keyword in instruction_lower:
            idx = instruction_lower.find(keyword)
            remaining = instruction[idx:]
            parts = remaining.split()
            
            if len(parts) > 1:
                potential_path = parts[1]
                if "/" in potential_path or "." in potential_path:
                    params["file_path"] = potential_path
                    content_start_idx = instruction.find(potential_path) + len(potential_path)
                    if content_start_idx < len(instruction):
                        remaining_content = instruction[content_start_idx:].strip()
                        if remaining_content and not remaining_content.startswith("branch"):
                            params["content"] = remaining_content
            break
    
    # Buscar branch con mejor parsing
    branch_keywords = ["branch", "rama"]
    for keyword in branch_keywords:
        if keyword in instruction_lower:
            idx = instruction_lower.find(keyword)
            parts = instruction[idx:].split()
            if len(parts) > 1:
                branch_value = parts[1].strip()
                if branch_value:
                    if "from" in instruction_lower or "desde" in instruction_lower:
                        from_idx = instruction_lower.find("from") if "from" in instruction_lower else instruction_lower.find("desde")
                        if idx < from_idx:
                            params["branch_name"] = branch_value
                        else:
                            params["base_branch"] = branch_value
                    else:
                        params["branch"] = branch_value
                        params["branch_name"] = branch_value
            break
    
    # Buscar title con mejor parsing
    if "title" in instruction_lower:
        title_idx = instruction_lower.find("title")
        title_part = instruction[title_idx + 5:].strip()
        if '"' in title_part:
            start_quote = title_part.find('"')
            end_quote = title_part.find('"', start_quote + 1)
            if end_quote > start_quote:
                params["title"] = title_part[start_quote + 1:end_quote]
        elif "'" in title_part:
            start_quote = title_part.find("'")
            end_quote = title_part.find("'", start_quote + 1)
            if end_quote > start_quote:
                params["title"] = title_part[start_quote + 1:end_quote]
        else:
            parts = title_part.split()
            if parts:
                params["title"] = parts[0]
    
    # Buscar head y base para PRs
    if "head" in instruction_lower:
        head_idx = instruction_lower.find("head")
        parts = instruction[head_idx:].split()
        if len(parts) > 1:
            params["head"] = parts[1].strip()
    
    if "base" in instruction_lower:
        base_idx = instruction_lower.find("base")
        parts = instruction[base_idx:].split()
        if len(parts) > 1:
            params["base"] = parts[1].strip()
    
    return params

