"""
Tool call parsing utilities.

Refactored to consolidate tool call parsing logic into a dedicated class.
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ToolCallParser:
    """
    Tool call parsing utilities.
    
    Responsibilities:
    - Parse tool calls from LLM responses
    - Validate tool call format
    - Extract tool call JSON
    
    Single Responsibility: Handle all tool call parsing operations.
    """
    
    @staticmethod
    def parse_tool_call(generated_text: str) -> Dict[str, Any]:
        """
        Parse tool call from generated text.
        
        Args:
            generated_text: Text containing tool call
            
        Returns:
            Parsed tool call dictionary
            
        Raises:
            ValueError: If tool call cannot be parsed
        """
        if "<tool>" not in generated_text:
            raise ValueError(f"Generated text does not contain <tool> tag: {generated_text}")
        
        tool_call_json_str = (
            generated_text.split("<tool>")[-1]
            .split("</tool>")[0]
            .strip()
            .replace(r"}}}", r"}}")
        )
        
        try:
            tool_call = json.loads(tool_call_json_str)
            return tool_call
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool call: {tool_call_json_str}") from e
    
    @staticmethod
    def format_tool_call_response(tool_call: Dict[str, Any]) -> str:
        """
        Format tool call as assistant message content.
        
        Args:
            tool_call: Tool call dictionary
            
        Returns:
            Formatted tool call string
        """
        return f"<tool>{json.dumps(tool_call)}</tool>"

