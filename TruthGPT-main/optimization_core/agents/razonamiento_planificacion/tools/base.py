import os
import sys
import json
import asyncio
import subprocess
import logging
import re
import httpx
from typing import Any, Callable, Dict, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
"""
System 5.9 — Agent Tools.

BaseTool ABC and built-in tool implementations for the ReAct agent loop.
"""



class ToolResult:
    """
    Standardized result from a tool execution.
    Can contain the final output string and optional internal signals for the orchestrator.
    """
    def __init__(
        self, 
        output: str, 
        metadata: Optional[Dict[Any, Any]] = None, 
        signal: Optional[str] = None
    ):
        self.output = output
        self.metadata = metadata or {}
        self.signal = signal  # e.g., "core_memory_update"

class BaseTool(ABC):
    """
    Clase base para herramientas automatizadas. 
    Permite que el agente obtenga la descripción automáticamente del docstring.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre único de la herramienta."""
        pass

    @property
    def description(self) -> str:
        """Description extracted from the docstring for LLM consumption."""
        return self.__doc__.strip() if self.__doc__ else "No description available."
        
    @property
    def risk_level(self) -> str:
        """Risk level (LOW, MEDIUM, HIGH) for execution permission handling."""
        return "LOW"
        
    @property
    def requires_approval(self) -> bool:
        """Si es True, la ejecución requerirá aprobación manual del usuario (HITL)."""
        return self.risk_level == "HIGH"

    @abstractmethod
    async def run(self, arg: str) -> Any:
        """
        Ejecución asíncrona de la herramienta. 
        Puede devolver un string simple o un objeto ToolResult.
        """
        pass
