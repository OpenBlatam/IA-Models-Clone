"""
ReAct Result Builder

Standardized result dictionary construction for consistent error handling
and action results across the ReAct agent.
"""

from typing import Dict, Any, Optional, List


class ResultBuilder:
    """
    Builder for standardizing action result dictionaries.
    
    This class eliminates code duplication by providing consistent
    methods for constructing success, error, and finish results.
    """
    
    @staticmethod
    def success_result(
        tool: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        duration: float
    ) -> Dict[str, Any]:
        """
        Build success action result dictionary.
        
        Args:
            tool: Tool name that was executed
            parameters: Parameters passed to the tool
            result: Tool execution result
            duration: Execution duration in seconds
            
        Returns:
            Standardized success result dictionary
        """
        return {
            "action": "tool_call",
            "tool": tool,
            "parameters": parameters,
            "result": result,
            "complete": result.get("success", False) and result.get("final", False),
            "duration": duration
        }
    
    @staticmethod
    def error_result(
        tool: Optional[str],
        parameters: Dict[str, Any],
        error: str,
        duration: float = 0.0,
        available_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build error action result dictionary.
        
        Args:
            tool: Tool name (None if tool not found)
            parameters: Parameters that were attempted
            error: Error message
            duration: Execution duration in seconds
            available_tools: Optional list of available tools (for tool not found errors)
            
        Returns:
            Standardized error result dictionary
        """
        result = {
            "action": "error",
            "tool": tool,
            "parameters": parameters,
            "result": {
                "success": False,
                "error": error
            },
            "complete": False,
            "duration": duration
        }
        
        if available_tools:
            result["result"]["available_tools"] = available_tools
        
        return result
    
    @staticmethod
    def finish_result(reason: str = "No actionable step found in thought") -> Dict[str, Any]:
        """
        Build finish action result dictionary.
        
        Used when no action is needed or task is complete.
        
        Args:
            reason: Reason for finishing
            
        Returns:
            Standardized finish result dictionary
        """
        return {
            "action": "finish",
            "tool": None,
            "parameters": {},
            "result": {"success": True, "result": "No action needed"},
            "complete": True,
            "reason": reason
        }
    
    @staticmethod
    def tool_not_found_result(
        tool_name: str,
        parameters: Dict[str, Any],
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Build tool not found error result.
        
        Args:
            tool_name: Name of tool that was not found
            parameters: Parameters that were attempted
            available_tools: List of available tools
            
        Returns:
            Standardized tool not found error result
        """
        from .react_constants import ErrorMessages
        
        return ResultBuilder.error_result(
            tool=tool_name,
            parameters=parameters,
            error=ErrorMessages.TOOL_NOT_FOUND.format(tool_name=tool_name),
            available_tools=available_tools
        )



