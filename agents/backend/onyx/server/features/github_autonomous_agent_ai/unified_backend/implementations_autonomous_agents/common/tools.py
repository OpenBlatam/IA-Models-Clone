"""
Tool System for Autonomous Agents
==================================

Tools allow agents to interact with external systems and perform actions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum


class ToolType(Enum):
    """Types of tools."""
    SEARCH = "search"
    CALCULATOR = "calculator"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    CUSTOM = "custom"


@dataclass
class Tool:
    """
    Represents a tool that an agent can use.
    
    Tools are functions that agents can call to perform actions.
    """
    name: str
    description: str
    tool_type: ToolType
    func: Callable
    parameters: Dict[str, Any]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Result dictionary with 'success', 'result', 'error', etc.
        """
        try:
            # Validate parameters
            for param, value in kwargs.items():
                if param not in self.parameters:
                    return {
                        "success": False,
                        "error": f"Unknown parameter: {param}",
                        "result": None
                    }
            
            # Execute tool
            result = self.func(**kwargs)
            
            return {
                "success": True,
                "result": result,
                "error": None,
                "tool": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "tool": self.name
            }


class ToolRegistry:
    """
    Registry for managing available tools.
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self.tools
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name (alias for get)."""
        tool = self.tools.get(name)
        if tool:
            return tool.func
        return None
    
    def list_tools(self) -> List[str]:

        """List all available tool names."""
        return list(self.tools.keys())
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool
            **kwargs: Tool parameters
            
        Returns:
            Result dictionary
        """
        tool = self.get(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "result": None
            }
        
        return tool.execute(**kwargs)


# Built-in tools

def search_tool(query: str) -> Dict[str, Any]:
    """Simple search tool (placeholder)."""
    # In production, integrate with actual search API
    return {
        "query": query,
        "results": [f"Result for: {query}"],
        "count": 1
    }


def calculator_tool(expression: str) -> Dict[str, Any]:
    """Calculator tool."""
    try:
        # Safe evaluation (in production, use a proper math parser)
        result = eval(expression.replace("^", "**"))
        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e)
        }


# Create default tool registry with built-in tools
default_tool_registry = ToolRegistry()
default_tool_registry.register(
    Tool(
        name="search",
        description="Search for information",
        tool_type=ToolType.SEARCH,
        func=search_tool,
        parameters={"query": {"type": "string", "required": True}}
    )
)
default_tool_registry.register(
    Tool(
        name="calculator",
        description="Perform mathematical calculations",
        tool_type=ToolType.CALCULATOR,
        func=calculator_tool,
        parameters={"expression": {"type": "string", "required": True}}
    )
)



