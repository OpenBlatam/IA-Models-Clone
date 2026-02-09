"""
Tools Main - Funciones base y entry points del módulo de herramientas

Rol en el Ecosistema IA:
- Funciones que los agentes pueden usar
- Búsqueda web, cálculos, llamadas a APIs, RAG
- Extensibilidad del sistema mediante herramientas
"""

from typing import Optional, Any
from .service import ToolService
from .registry import ToolRegistry
from utils.main import get_util_service
from llm.main import get_llm_service


# Instancia global del servicio
_tool_service: Optional[ToolService] = None


def get_tool_service() -> ToolService:
    """
    Obtiene la instancia global del servicio de herramientas.
    
    Returns:
        ToolService: Servicio de herramientas
    """
    global _tool_service
    if _tool_service is None:
        util_service = get_util_service()
        llm_service = get_llm_service()
        _tool_service = ToolService(
            util_service=util_service,
            llm_service=llm_service
        )
    return _tool_service


def execute_tool(name: str, *args, **kwargs) -> Any:
    """
    Ejecuta una herramienta.
    
    Args:
        name: Nombre de la herramienta
        *args: Argumentos posicionales
        **kwargs: Argumentos con nombre
        
    Returns:
        Resultado de la herramienta
    """
    service = get_tool_service()
    return service.execute_tool(name, *args, **kwargs)


def register_tool(name: str, tool) -> None:
    """
    Registra una herramienta en el sistema.
    
    Args:
        name: Nombre de la herramienta
        tool: Instancia de la herramienta
    """
    service = get_tool_service()
    service.register_tool(name, tool)


def list_tools() -> list:
    """
    Lista todas las herramientas registradas.
    
    Returns:
        Lista de nombres de herramientas
    """
    service = get_tool_service()
    return service.registry.list()


def initialize_tools() -> ToolService:
    """
    Inicializa el sistema de herramientas.
    
    Returns:
        ToolService: Servicio inicializado
    """
    return get_tool_service()

