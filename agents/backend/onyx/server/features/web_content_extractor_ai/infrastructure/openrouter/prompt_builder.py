"""
Prompt builder for OpenRouter content extraction.

Centralizes prompt construction logic.
"""

from typing import Optional


def build_extraction_prompt(url: str, content_preview: str) -> str:
    """
    Build prompt for content extraction from web pages.
    
    Args:
        url: URL of the web page
        content_preview: Preview of the web content (truncated if needed)
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> prompt = build_extraction_prompt("https://example.com", content)
        >>> # Returns formatted prompt for OpenRouter
    """
    return f"""Extrae toda la información relevante de esta página web.

URL: {url}

Contenido:
{content_preview}

Extrae y estructura la siguiente información:
1. Título principal
2. Descripción/resumen
3. Contenido principal (texto completo)
4. Metadatos (autor, fecha, categorías)
5. Enlaces importantes
6. Imágenes y medios
7. Información de contacto (si existe)
8. Precios/productos (si aplica)
9. Cualquier otra información relevante

Devuelve la información en formato JSON estructurado."""


def truncate_content(content: str, max_length: int = 50000) -> str:
    """
    Truncate content if it exceeds max_length.
    
    Args:
        content: Content to truncate
        max_length: Maximum length (default: 50000)
        
    Returns:
        Truncated content if needed, original content otherwise
        
    Example:
        >>> short_content = truncate_content(long_content)
    """
    return content[:max_length] if len(content) > max_length else content

