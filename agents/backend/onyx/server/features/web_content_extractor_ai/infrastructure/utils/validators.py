"""
Validadores de datos
"""

from typing import Any, Dict, Optional
import json


def validate_json_response(content: str) -> Optional[Dict[str, Any]]:
    """
    Validar y parsear respuesta JSON
    
    Args:
        content: String que debería contener JSON
        
    Returns:
        Dict parseado o None si no es válido
    """
    if not content or not isinstance(content, str):
        return None
    
    content = content.strip()
    
    # Intentar extraer JSON si está en markdown code block
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end > start:
            content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end > start:
            content = content[start:end].strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def sanitize_content(content: str, max_length: int = 100000) -> str:
    """
    Sanitizar contenido para procesamiento
    
    Args:
        content: Contenido a sanitizar
        max_length: Longitud máxima
        
    Returns:
        Contenido sanitizado
    """
    if not content:
        return ""
    
    # Truncar si es muy largo
    if len(content) > max_length:
        content = content[:max_length] + "... [truncado]"
    
    # Remover caracteres de control problemáticos
    content = "".join(char for char in content if ord(char) >= 32 or char in "\n\t")
    
    return content








