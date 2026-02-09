"""
Tool Chain
==========
Chain multiple tools together for complex workflows.
"""

from typing import List, Dict, Any, Optional
from .base import BaseAPITool, ToolResult
from .factory import get_factory
from .utils import print_success, print_error, print_info


class ToolChain:
    """Chain of tools to execute sequentially."""
    
    def __init__(self):
        self.factory = get_factory()
        self.tools: List[Dict[str, Any]] = []
        self.results: List[ToolResult] = []
    
    def add_tool(self, name: str, **kwargs):
        """Add tool to chain."""
        self.tools.append({
            "name": name,
            "kwargs": kwargs
        })
        return self
    
    def execute(self, stop_on_error: bool = True) -> List[ToolResult]:
        """Execute all tools in chain."""
        print_info(f"Executing tool chain with {len(self.tools)} tools...")
        
        for i, tool_config in enumerate(self.tools, 1):
            tool_name = tool_config["name"]
            tool_kwargs = tool_config["kwargs"]
            
            print_info(f"[{i}/{len(self.tools)}] Running {tool_name}...")
            
            tool = self.factory.create(tool_name, **tool_kwargs)
            if not tool:
                error_result = ToolResult(
                    success=False,
                    message=f"Tool '{tool_name}' not found"
                )
                self.results.append(error_result)
                print_error(f"Tool '{tool_name}' not found")
                
                if stop_on_error:
                    break
                continue
            
            result = tool.run(**tool_kwargs)
            self.results.append(result)
            
            if result.success:
                print_success(f"{tool_name}: {result.message}")
            else:
                print_error(f"{tool_name}: {result.message}")
                
                if stop_on_error:
                    break
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "results": self.results
        }
    
    def print_summary(self):
        """Print execution summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("📊 Tool Chain Summary")
        print("=" * 60)
        print(f"Total Tools: {summary['total']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print("=" * 60)


def create_chain() -> ToolChain:
    """Create a new tool chain."""
    return ToolChain()


# Example usage
if __name__ == "__main__":
    chain = create_chain()
    chain.add_tool("health", endpoints=["/health", "/"])
    chain.add_tool("benchmark", endpoint="/health", iterations=10)
    
    results = chain.execute()
    chain.print_summary()



