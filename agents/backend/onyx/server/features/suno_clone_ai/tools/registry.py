"""
Tool Registry - Registro de herramientas
"""

from typing import Dict, Optional
from .base import BaseTool


class ToolRegistry:
    """Registro de herramientas disponibles"""

    def __init__(self):
        """Inicializa el registro"""
        self._tools: Dict[str, BaseTool] = {}

    def register(self, name: str, tool: BaseTool) -> None:
        """Registra una herramienta"""
        self._tools[name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Obtiene una herramienta por nombre"""
        return self._tools.get(name)

    def list(self) -> list:
        """Lista todas las herramientas registradas"""
        return list(self._tools.keys())

