"""
Helpers - Funciones auxiliares y utilidades
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_content(content: str, max_length: Optional[int] = None) -> str:
    """
    Sanitizar contenido removiendo caracteres problemáticos.

    Args:
        content: Contenido a sanitizar
        max_length: Longitud máxima (opcional)

    Returns:
        Contenido sanitizado
    """
    # Remover caracteres de control excepto newlines y tabs
    content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
    
    # Limitar longitud si se especifica
    if max_length and len(content) > max_length:
        content = content[:max_length]
        logger.warning(f"Contenido truncado a {max_length} caracteres")
    
    return content


def extract_sections(content: str, pattern: str = r'^#+\s') -> List[Dict[str, Any]]:
    """
    Extraer secciones del contenido basado en un patrón.

    Args:
        content: Contenido a analizar
        pattern: Patrón regex para identificar secciones

    Returns:
        Lista de secciones encontradas
    """
    sections = []
    lines = content.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            if current_section:
                sections.append(current_section)
            current_section = {
                "title": line.strip(),
                "start_line": i,
                "content": line + "\n"
            }
        elif current_section:
            current_section["content"] += line + "\n"
            current_section["end_line"] = i
    
    if current_section:
        sections.append(current_section)
    
    return sections


def calculate_text_statistics(content: str) -> Dict[str, Any]:
    """
    Calcular estadísticas del texto.

    Args:
        content: Contenido a analizar

    Returns:
        Diccionario con estadísticas
    """
    words = content.split()
    sentences = re.split(r'[.!?]+', content)
    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    
    return {
        "characters": len(content),
        "characters_no_spaces": len(content.replace(' ', '')),
        "words": len(words),
        "sentences": len([s for s in sentences if s.strip()]),
        "paragraphs": len(paragraphs),
        "lines": len(content.split('\n')),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    }


def find_and_replace(
    content: str,
    find: str,
    replace: str,
    case_sensitive: bool = True,
    max_replacements: Optional[int] = None
) -> Dict[str, Any]:
    """
    Buscar y reemplazar con información detallada.

    Args:
        content: Contenido donde buscar
        find: Texto a buscar
        replace: Texto de reemplazo
        case_sensitive: Si la búsqueda es sensible a mayúsculas
        max_replacements: Número máximo de reemplazos

    Returns:
        Diccionario con resultado y estadísticas
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.escape(find)
    
    matches = list(re.finditer(pattern, content, flags))
    replacement_count = len(matches) if not max_replacements else min(len(matches), max_replacements)
    
    if replacement_count > 0:
        result_content = re.sub(
            pattern,
            replace,
            content,
            count=max_replacements or 0,
            flags=flags
        )
    else:
        result_content = content
    
    return {
        "success": replacement_count > 0,
        "content": result_content,
        "replacements": replacement_count,
        "matches_found": len(matches),
        "original_length": len(content),
        "new_length": len(result_content)
    }


def validate_content_structure(content: str, format_type: str) -> Dict[str, Any]:
    """
    Validar estructura del contenido según su formato.

    Args:
        content: Contenido a validar
        format_type: Tipo de formato (json, markdown, etc.)

    Returns:
        Resultado de la validación
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    if format_type == "json":
        try:
            import json
            json.loads(content)
        except json.JSONDecodeError as e:
            validation["valid"] = False
            validation["errors"].append(f"JSON inválido: {str(e)}")
    
    elif format_type == "markdown":
        # Validaciones básicas de Markdown
        if content.count('```') % 2 != 0:
            validation["warnings"].append("Bloques de código no balanceados")
    
    return validation


def split_content_by_length(content: str, max_length: int) -> List[str]:
    """
    Dividir contenido en chunks de longitud máxima.

    Args:
        content: Contenido a dividir
        max_length: Longitud máxima por chunk

    Returns:
        Lista de chunks
    """
    chunks = []
    current_chunk = ""
    
    for line in content.split('\n'):
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks






