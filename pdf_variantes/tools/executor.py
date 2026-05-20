"""
Tool Executor
=============
Advanced executor with plugins, chaining, and parallel execution.
"""

import concurrent.futures
from typing import List, Dict, Any, Optional
from .base import BaseAPITool, ToolResult
from .factory import get_factory
from .plugins import get_plugin_manager, LoggingPlugin, MetricsPlugin
from .chain import ToolChain
from .utils import print_success, print_error, print_info


class ToolExecutor:
    """Advanced tool executor with plugins and parallel execution."""
    
    def __init__(self):
        self.factory = get_factory()
        self.plugin_manager = get_plugin_manager()
        self.metrics_plugin = MetricsPlugin()
        
        # Register default plugins
        self.plugin_manager.register(LoggingPlugin())
        self.plugin_manager.register(self.metrics_plugin)
    
    def execute(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """Execute a single tool with plugins."""
        tool = self.factory.create(tool_name, **kwargs)
        if not tool:
            return ToolResult(
                success=False,
                message=f"Tool '{tool_name}' not found"
            )
        
        # Apply before hooks
        kwargs = self.plugin_manager.apply_before(tool, **kwargs)
        
        # Execute tool
        result = tool.run(**kwargs)
        
        # Apply after hooks
        result = self.plugin_manager.apply_after(tool, result, **kwargs)
        
        return result
    
    def execute_parallel(
        self,
        tools: List[Dict[str, Any]],
        max_workers: int = 5
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        print_info(f"Executing {len(tools)} tools in parallel (max_workers: {max_workers})...")
        
        def execute_single(tool_config: Dict[str, Any]) -> ToolResult:
            tool_name = tool_config["name"]
            tool_kwargs = tool_config.get("kwargs", {})
            return self.execute(tool_name, **tool_kwargs)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(execute_single, tool_config) for tool_config in tools]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        successful = sum(1 for r in results if r.success)
        print_success(f"Parallel execution completed: {successful}/{len(results)} successful")
        
        return results
    
    def execute_chain(
        self,
        tool_configs: List[Dict[str, Any]],
        stop_on_error: bool = True
    ) -> List[ToolResult]:
        """Execute tools in sequence (chain)."""
        chain = ToolChain()
        
        for tool_config in tool_configs:
            chain.add_tool(tool_config["name"], **tool_config.get("kwargs", {}))
        
        return chain.execute(stop_on_error=stop_on_error)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get execution metrics."""
        return self.metrics_plugin.get_metrics()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tool Executor")
    parser.add_argument("--tool", help="Tool to execute")
    parser.add_argument("--parallel", nargs="+", help="Tools to execute in parallel")
    parser.add_argument("--chain", nargs="+", help="Tools to execute in sequence")
    
    args = parser.parse_args()
    
    executor = ToolExecutor()
    
    if args.tool:
        result = executor.execute(args.tool)
        print_success(result.message) if result.success else print_error(result.message)
    elif args.parallel:
        # Parse parallel tools (format: name:kwargs)
        tools = [{"name": name, "kwargs": {}} for name in args.parallel]
        results = executor.execute_parallel(tools)
    elif args.chain:
        # Parse chain tools
        tools = [{"name": name, "kwargs": {}} for name in args.chain]
        results = executor.execute_chain(tools)
    else:
        print("Use --tool, --parallel, or --chain")


if __name__ == "__main__":
    main()



