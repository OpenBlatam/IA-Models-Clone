"""
Tool Manager
============
Centralized manager for all tools.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base import BaseAPITool, ToolResult
from .config import get_config, ToolConfig
from .registry import get_registry, register_tool
from .utils import print_success, print_error, print_info


class ToolManager:
    """Manager for API tools."""
    
    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or get_config()
        self.registry = get_registry()
        self.tools: Dict[str, BaseAPITool] = {}
    
    def create_tool(self, name: str, **kwargs) -> Optional[BaseAPITool]:
        """Create tool instance."""
        tool_class = self.registry.get(name)
        if not tool_class:
            print_error(f"Tool '{name}' not found")
            return None
        
        # Merge config with kwargs
        tool_kwargs = {
            "base_url": self.config.base_url,
            "timeout": self.config.timeout
        }
        tool_kwargs.update(kwargs)
        
        tool = tool_class(**tool_kwargs)
        self.tools[name] = tool
        return tool
    
    def run_tool(self, name: str, **kwargs) -> Optional[ToolResult]:
        """Run a tool."""
        tool = self.create_tool(name, **kwargs)
        if not tool:
            return None
        
        print_info(f"Running tool: {name}")
        result = tool.run(**kwargs)
        
        if result.success:
            print_success(result.message)
        else:
            print_error(result.message)
        
        return result
    
    def list_available_tools(self) -> List[str]:
        """List all available tools."""
        return self.registry.list_tools()
    
    def print_tools_list(self):
        """Print list of available tools."""
        tools = self.list_available_tools()
        
        print("\n" + "=" * 60)
        print("📋 Available Tools")
        print("=" * 60)
        
        if not tools:
            print("No tools registered")
        else:
            for i, tool_name in enumerate(tools, 1):
                print(f"  {i}. {tool_name}")
        
        print("=" * 60)


# Register tools
from .refactored_health_checker import HealthChecker
from .refactored_benchmark import Benchmark
from .refactored_test_suite import TestSuite

register_tool("health")(HealthChecker)
register_tool("benchmark")(Benchmark)
register_tool("test")(TestSuite)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tool Manager")
    parser.add_argument("--list", action="store_true", help="List available tools")
    parser.add_argument("--tool", help="Tool to run")
    parser.add_argument("--url", help="API base URL")
    
    args = parser.parse_args()
    
    manager = ToolManager()
    
    if args.list:
        manager.print_tools_list()
    elif args.tool:
        config = get_config()
        if args.url:
            config.base_url = args.url
        
        result = manager.run_tool(args.tool)
        sys.exit(0 if result and result.success else 1)
    else:
        manager.print_tools_list()
        print("\nUse --tool <name> to run a tool")


if __name__ == "__main__":
    main()

