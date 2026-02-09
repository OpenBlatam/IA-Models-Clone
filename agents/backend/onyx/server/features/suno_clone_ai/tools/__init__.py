"""
Tools Module - Herramientas y Utilidades
Herramientas reutilizables, funciones auxiliares, y utilidades específicas.

Rol en el Ecosistema IA:
- Funciones que los agentes pueden usar
- Búsqueda web, cálculos, llamadas a APIs, RAG
- Extensibilidad del sistema mediante herramientas

Reglas de Importación:
- Puede importar: utils, llm
- NO debe importar: agents, chat (evitar ciclos)
- Los agentes importan este módulo, no al revés
"""

from .base import BaseTool
from .service import ToolService
from .registry import ToolRegistry
from .main import (
    get_tool_service,
    execute_tool,
    register_tool,
    list_tools,
    initialize_tools,
)

__all__ = [
    # Clases principales
    "BaseTool",
    "ToolService",
    "ToolRegistry",
    # Funciones de acceso rápido
    "get_tool_service",
    "execute_tool",
    "register_tool",
    "list_tools",
    "initialize_tools",
]

