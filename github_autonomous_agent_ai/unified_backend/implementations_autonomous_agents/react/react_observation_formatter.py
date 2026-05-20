"""
ReAct Observation Formatter

Handles formatting of action results into natural language observations.
Provides consistent formatting across success and error cases.
"""

import json
from typing import Dict, Any

from .react_constants import ObservationTemplates


class ObservationFormatter:
    """
    Formatter for converting action results to observations.
    
    Converts structured action results into natural language observations
    that can be used in the next reasoning step.
    """
    
    def __init__(self):
        """Initialize observation formatter."""
        self.templates = ObservationTemplates()
    
    def format(self, action_result: Dict[str, Any]) -> str:
        """
        Format action result as observation.
        
        Args:
            action_result: Result from action execution
            
        Returns:
            Natural language observation
        """
        # Check if task is complete
        if action_result.get("complete"):
            return self.templates.TASK_COMPLETE
        
        tool = action_result.get("tool")
        result = action_result.get("result", {})
        
        # Format based on success/failure
        if result.get("success"):
            observation = self.format_success(tool, result)
        else:
            observation = self.format_error(tool, result)
        
        # Add context if available
        if "context" in result:
            observation += f" {self.templates.CONTEXT_PREFIX.format(context=result['context'])}"
        
        return observation
    
    def format_success(
        self, 
        tool: str, 
        result: Dict[str, Any]
    ) -> str:
        """
        Format successful action result as observation.
        
        Args:
            tool: Tool name that succeeded
            result: Tool execution result
            
        Returns:
            Formatted success observation
        """
        result_value = result.get("result")
        result_str = self._format_result_value(result_value)
        
        return self.templates.SUCCESS_TEMPLATE.format(
            tool=tool,
            result=result_str[:500]  # Limit length
        )
    
    def format_error(
        self, 
        tool: str, 
        result: Dict[str, Any]
    ) -> str:
        """
        Format failed action result as observation.
        
        Args:
            tool: Tool name that failed
            result: Tool execution result
            
        Returns:
            Formatted error observation
        """
        error = result.get("error", "Unknown error")
        
        return self.templates.ERROR_TEMPLATE.format(
            tool=tool,
            error=error
        )
    
    def _format_result_value(self, result_value: Any) -> str:
        """
        Format result value for observation.
        
        Handles different result types (dict, list, primitive).
        
        Args:
            result_value: Result value to format
            
        Returns:
            Formatted string representation
        """
        if isinstance(result_value, dict):
            return json.dumps(result_value, indent=2)
        elif isinstance(result_value, (list, tuple)):
            return f"[{len(result_value)} items]"
        else:
            return str(result_value)



