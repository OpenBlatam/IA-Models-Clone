"""
Tool Service - Servicio de herramientas
"""

from typing import Any, Dict, Optional
from .base import BaseTool
from .registry import ToolRegistry
from utils.service import UtilService
from llm.service import LLMService


class ToolService:
    """Servicio para gestionar herramientas"""

    def __init__(
        self,
        util_service: Optional[UtilService] = None,
        llm_service: Optional[LLMService] = None
    ):
        """Inicializa el servicio de herramientas"""
        self.registry = ToolRegistry()
        self.util_service = util_service
        self.llm_service = llm_service

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Registra una herramienta"""
        self.registry.register(name, tool)

    def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """Ejecuta una herramienta"""
        tool = self.registry.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return tool.execute(*args, **kwargs)

