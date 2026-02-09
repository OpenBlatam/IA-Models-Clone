"""
Tools Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


class Tool:
    """Tool definition"""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.created_at = datetime.utcnow()


class ToolBase(ABC):
    """Base interface for tools"""
    
    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute tool"""
        pass
    
    @abstractmethod
    async def register_tool(self, tool: Tool) -> bool:
        """Register tool"""
        pass
    
    @abstractmethod
    async def list_tools(self) -> list:
        """List available tools"""
        pass

