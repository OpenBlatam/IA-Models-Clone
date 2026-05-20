"""
Tool Service Implementation
"""

from typing import Dict, Any, List
import logging
import time

from .base import ToolBase, Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolService(ToolBase):
    """Tool service implementation"""
    
    def __init__(
        self,
        httpx_client=None,
        db=None,
        tracing_service=None
    ):
        """Initialize tool service"""
        self.httpx_client = httpx_client
        self.db = db
        self.tracing_service = tracing_service
        self._tools: dict = {}
        self._handlers: dict = {}
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute tool"""
        try:
            tool = self._tools.get(tool_name)
            if not tool:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Tool '{tool_name}' not found"
                )
            
            handler = self._handlers.get(tool_name)
            if not handler:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Handler for tool '{tool_name}' not found"
                )
            
            start_time = time.time()
            output = await handler(parameters)
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                output=output,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def register_tool(self, tool: Tool) -> bool:
        """Register tool"""
        try:
            self._tools[tool.name] = tool
            return True
            
        except Exception as e:
            logger.error(f"Error registering tool: {e}")
            return False
    
    def register_handler(self, tool_name: str, handler: callable):
        """Register tool handler"""
        self._handlers[tool_name] = handler
    
    async def list_tools(self) -> List[Tool]:
        """List available tools"""
        return list(self._tools.values())

